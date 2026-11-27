import json
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from search_benchmark import busca_exata, busca_hnsw, buscar_artigos

"""
Exemplo para o membro 5:
from src.rag_llm import responder_pergunta

texto_resposta = responder_pergunta("Qual o tratamento de primeira linha para diabetes tipo 2?")
"""

SYSTEM_PROMPT = """Você é um assistente médico rigoroso.
REGRAS OBRIGATÓRIAS:
1. Responda SEMPRE em português do Brasil, mesmo que as sinopses estejam em inglês.
2. Use APENAS as informações das sinopses fornecidas. Não invente dados.
3. Cite obrigatoriamente o PMID e o Título do artigo usado, no formato: (Fonte: PMID <numero> - <Titulo>).
4. Seja direto e conciso (máximo de 3 frases)."""

# Cache em memória para não recarregar SBERT e FAISS a cada pergunta do front-end
_CACHE_BUSCA = {}

def _carregar_motor_sbert(db_path: str, indices_dir: str):
    """Carrega SQLite, SBERT, vetores memmap e indice HNSW apenas uma vez."""
    chave = (db_path, indices_dir)
    if chave in _CACHE_BUSCA:
        return _CACHE_BUSCA[chave]

    pasta = Path(indices_dir)
    meta = json.loads((pasta / "metadata.json").read_text(encoding="utf-8"))
    n_docs = int(meta["num_docs"])
    dim_sbert = int(meta["sbert_dim"])

    nome_modelo = meta.get("sbert_model", "sentence-transformers/all-MiniLM-L6-v2")
    if not Path(nome_modelo).exists() and "all-MiniLM-L6-v2" in nome_modelo:
        nome_modelo = "sentence-transformers/all-MiniLM-L6-v2"

    sbert = SentenceTransformer(nome_modelo, device="cpu")
    vetores = np.memmap(
        pasta / "sbert_vectors.mmap",
        dtype=np.float32,
        mode="r",
        shape=(n_docs, dim_sbert),
    )
    indice_hnsw = faiss.read_index(str(pasta / "hnsw_sbert.index"), faiss.IO_FLAG_MMAP)
    indice_hnsw.hnsw.efSearch = 64

    motor = {
        "db_path": db_path,
        "sbert": sbert,
        "vetores": vetores,
        "indice_hnsw": indice_hnsw,
    }
    _CACHE_BUSCA[chave] = motor
    return motor

def buscar_artigos_reais(
        pergunta: str,
        metodo_busca: str = "hnsw",
        top_k: int = 3,
        db_path: str = "data/pubmed_10k_sample.db",
        indices_dir: str = "indices/indice_pubmed_10k",
) -> tuple[list[dict], float]:
    """
    Executa a busca semântica real (SBERT + HNSW ou Cosseno Exato) nos arquivos
    do projeto e retorna (lista_de_artigos, tempo_ms).
    """
    inicio = time.perf_counter()
    motor = _carregar_motor_sbert(db_path, indices_dir)

    vetor = motor["sbert"].encode(
        [pergunta], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )[0]
    consulta = np.asarray(vetor, dtype=np.float32)

    if metodo_busca.lower() == "exato":
        ids, scores = busca_exata(consulta, motor["vetores"], top_k)
    else:
        ids, scores = busca_hnsw(consulta, motor["indice_hnsw"], top_k)

    with sqlite3.connect(motor["db_path"]) as conn:
        artigos = buscar_artigos(conn, ids, scores)

    tempo_ms = round((time.perf_counter() - inicio) * 1000, 2)
    return artigos, tempo_ms

def gerar_resposta(pergunta: str, artigos: list[dict], modelo: str = "phi3:mini") -> dict:
    """Envia a pergunta + sinopses recuperadas para a LLM local no Ollama."""
    contexto = ""
    for art in artigos:
        contexto+= (
            f"PMID: {art['pmid']}\n"
            f"Título: {art['titulo']}\n"
            f"Sinopse: {art['sinopse']}\n\n"
        )

    user_prompt = f"SINOPSES:\n{contexto}\nPERGUNTA: {pergunta}\nRESPOSTA EM PORTUGUÊS:"

    inicio = time.perf_counter()
    resposta = ollama.chat(
        model=modelo,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )
    tempo_seg = round(time.perf_counter() - inicio, 2)

    return {
        "resposta": resposta["message"]["content"].strip(),
        "tempo_segundos": tempo_seg,
        "modelo": modelo,
    }

def responder_pergunta(
    pergunta: str,
    metodo_busca: str = "hnsw",
    modelo: str = "phi3:mini",
    top_k: int = 3,
    db_path: str = "data/pubmed_10k_sample.db",
    indices_dir: str = "indices/indice_pubmed_10k",
) -> str:
    """
    FUNÇÃO PRINCIPAL PARA O FRONT-END / API (Membro 5).
    Recebe apenas o texto da pergunta (demais parâmetros são opcionais)
    e retorna uma string pronta com resposta, links PubMed e tempos.
    """
    if not pergunta or not pergunta.strip():
        return "Por favor, digite uma pergunta válida."

    usou_mock = False
    if Path(db_path).is_file() and (Path(indices_dir) / "metadata.json").is_file():
        artigos, tempo_busca_ms = buscar_artigos_reais(
            pergunta=pergunta,
            metodo_busca=metodo_busca,
            top_k=top_k,
            db_path=db_path,
            indices_dir=indices_dir,
        )
    else:
        usou_mock = True
        tempo_busca_ms = 0.0
        artigos = [
            {
                "pmid": "11832527",
                "titulo": "Reduction in the incidence of type 2 diabetes with lifestyle intervention or metformin",
                "sinopse": (
                    "In this clinical trial, metformin reduced the incidence of type 2 diabetes "
                    "by 31% compared with placebo in high-risk adults, while lifestyle intervention "
                    "reduced it by 58%."
                ),
            }
        ]

    try:
        resultado = gerar_resposta(
            pergunta=pergunta, 
            artigos=artigos, 
            modelo=modelo
            )
    except Exception as erro:
        return (
            f"Erro ao gerar resposta com o Ollama ({modelo}). "
            f"Verifique se o Ollama está aberto e o modelo foi baixado. Detalhe: {erro}"
        )

    fontes_md = "\n".join(
        f"- PMID {a['pmid']}: {a['titulo']} (https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/)"
        for a in artigos
        if a.get("pmid")
    )
    aviso_mock = " [MODO TESTE - base 10k não encontrada]" if usou_mock else ""

    return (
        f"{resultado['resposta']}\n\n"
        f"Fontes consultadas:\n{fontes_md}\n\n"
        f"---\n"
        f"Busca: SBERT + {metodo_busca}{aviso_mock} ({tempo_busca_ms} ms) | "
        f"LLM: {resultado['modelo']} ({resultado['tempo_segundos']} s)"
    )

if __name__ == "__main__":
    pergunta_teste = "What are the first-line pharmacological treatments for type 2 diabetes mellitus?"
    print("Testando pipeline RAG completo...\n")
    print(responder_pergunta(pergunta_teste))