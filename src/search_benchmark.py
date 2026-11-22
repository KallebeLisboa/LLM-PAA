"""
Compara as quatro buscas semânticas do projeto (PubMed):

    Word2Vec Average     + cosseno exato -> O(N*d)
    Word2Vec Average     + HNSW -> ~O(log N*d)
    Sentence Embeddings  + cosseno exato -> O(N*d)
    Sentence Embeddings  + HNSW -> ~O(log N*d)

Para cada pergunta médica (em inglês) o script:
1. transforma a pergunta em vetor (Word2Vec e Sentence Embeddings);
2. roda e cronometra as 4 buscas (incluindo variação de efSearch no HNSW);
3. busca PMID, título e sinopse dos artigos no SQLite;
4. compara quantos artigos do HNSW coincidem com os do cosseno exato (Recall@K).

No fim mostra a média dos tempos e salva os gráficos, o relatório em texto,
os artigos em JSON (para o LLM/RAGAS) e as métricas estruturadas (para comparar
escala 10k vs. 10 GB).

Uso, a partir da raiz do repositório:
    python src/search_benchmark.py

Base completa (exemplo):
    python src/search_benchmark.py --db "10gb dados/Arquivo em Sqlite/pubmed_10gb.db" \
        --indices "10gb dados/Resultado_da_Vetorizacao" --saida resultado_5rep --repeticoes 5

Observações:
- Os vetores do projeto já são normalizados (produto interno = cosseno).
- Os tempos são medianas de várias repetições, com cache do sistema aquecido.
- O "cosseno exato" é força bruta em NumPy (multiplicação matricial em blocos
  sobre um memmap), não um IndexFlatIP do FAISS.
- Em bases com mais de 1M de artigos, o HNSW do SBERT usa quantização FP16
  (definida no build_indices.py); o recall medido já inclui esse efeito.
"""

import argparse
import json
import sqlite3
import sys
import textwrap
import time
from pathlib import Path
from statistics import mean, median

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from matplotlib.ticker import FuncFormatter
from sentence_transformers import SentenceTransformer

# Usa o mesmo pré-processamento da construção dos índices.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_indices import compute_w2v_doc_vector, tokenize  # noqa: E402


PERGUNTAS = [
    "What are the first-line pharmacological treatments for type 2 diabetes mellitus?",
    "How do mRNA vaccines induce an immune response against viral infections?",
    "What are the main risk factors and biomarkers for early Alzheimer's disease?",
    "What is the mechanism of action of beta-blockers in treating hypertension?",
    "How effective is immunotherapy in treating non-small cell lung cancer?",
    "What are the neurological complications associated with severe COVID-19?",
    "How does gut microbiota composition influence inflammatory bowel disease?",
    "What are the diagnostic criteria and management for acute myocardial infarction?",
    "What are the common side effects of chemotherapy in breast cancer patients?",
    "How is major depressive disorder treated with antidepressants and psychotherapy?",
]

W2V = "Word2Vec Average"
SBERT = "Sentence Embeddings"

FASES_CONSTRUCAO = [
    ("w2v_train_time_s", "Treino do Word2Vec"),
    ("w2v_vectorize_time_s", "Vetorização Word2Vec dos artigos"),
    ("sbert_vectorize_time_s", "Vetorização SBERT dos artigos"),
    ("hnsw_w2v_build_time_s", "Construção do HNSW (Word2Vec)"),
    ("hnsw_sbert_build_time_s", "Construção do HNSW (SBERT)"),
]


# Formatação em pt-BR

def numero_br(valor, casas=0):
    """Formata número no padrão brasileiro: 6.530.506 e 289,2."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def ms_br(valor, casas=3):
    return numero_br(valor, casas)


# Medição e buscas
def medir(funcao, repeticoes, *args):
    """Roda funcao(*args) várias vezes. Devolve (último resultado, mediana em ms)."""
    tempos = []
    resultado = None
    for _ in range(repeticoes):
        inicio = time.perf_counter()
        resultado = funcao(*args)
        tempos.append((time.perf_counter() - inicio) * 1000)
    return resultado, median(tempos)


def busca_exata(consulta, vetores, k, bloco=100_000):
    """
    Compara a consulta com todos os vetores, um bloco por vez, guardando só
    os k melhores. Custo O(N * d).
    """
    total = vetores.shape[0]
    k = min(k, total)
    melhores_scores = np.full(k, -np.inf, dtype=np.float32)
    melhores_ids = np.full(k, -1, dtype=np.int64)

    for inicio in range(0, total, bloco):
        fim = min(inicio + bloco, total)
        scores = np.asarray(vetores[inicio:fim], dtype=np.float32) @ consulta

        n_locais = min(k, len(scores))
        locais = np.argpartition(scores, -n_locais)[-n_locais:]

        todos_scores = np.concatenate([melhores_scores, scores[locais]])
        todos_ids = np.concatenate([melhores_ids, locais.astype(np.int64) + inicio])
        manter = np.argpartition(todos_scores, -k)[-k:]
        melhores_scores = todos_scores[manter]
        melhores_ids = todos_ids[manter]

    ordem = np.argsort(melhores_scores)[::-1]
    return melhores_ids[ordem], melhores_scores[ordem]


def busca_hnsw(consulta, indice, k):
    """Busca aproximada no índice HNSW (o efSearch já foi definido no índice)."""
    scores, ids = indice.search(consulta.reshape(1, -1).astype(np.float32), k)
    return ids[0], scores[0]


def buscar_artigos(conn, ids, scores):
    """Busca PMID, título e sinopse dos ids no SQLite, mantendo a ordem do ranking."""
    validos = [int(i) for i in ids if i >= 0]
    if not validos:
        return []

    marcadores = ",".join("?" * len(validos))
    linhas = conn.execute(
        f"SELECT doc_id, pmid, title, content FROM articles WHERE doc_id IN ({marcadores})",
        validos,
    ).fetchall()
    por_id = {doc_id: (pmid, titulo, conteudo) for doc_id, pmid, titulo, conteudo in linhas}

    artigos = []
    for posicao, (doc_id, score) in enumerate(zip(ids, scores), start=1):
        if int(doc_id) in por_id:
            pmid, titulo, conteudo = por_id[int(doc_id)]
            artigos.append({
                "rank": posicao,
                "pmid": pmid or "",
                "titulo": titulo or "",
                "sinopse": " ".join((conteudo or "").split()),
                "score": float(score),
            })
    return artigos


def ids_validos(ids):
    return {int(i) for i in ids if i >= 0}


# Validação dos arquivos
def validar(conn, pasta, meta):
    """Confere se banco e índices são da mesma base. Devolve (N, dim_w2v, dim_sbert)."""
    necessarios = [
        "word2vec.model", "w2v_vectors.mmap", "sbert_vectors.mmap",
        "hnsw_w2v.index", "hnsw_sbert.index",
    ]
    faltando = [nome for nome in necessarios if not (pasta / nome).is_file()]
    if faltando:
        raise FileNotFoundError(f"Faltam arquivos em '{pasta}': {', '.join(faltando)}")

    n_docs = int(meta["num_docs"])
    total, menor, maior = conn.execute(
        "SELECT COUNT(*), MIN(doc_id), MAX(doc_id) FROM articles"
    ).fetchone()
    if total != n_docs:
        raise ValueError(
            f"O banco tem {total:,} artigos, mas o metadata.json diz {n_docs:,}. "
            "Use o banco e os índices da mesma base."
        )
    if total > 0 and (menor != 0 or maior != total - 1):
        raise ValueError(f"Os doc_id deveriam ir de 0 a {total - 1}, mas vão de {menor} a {maior}.")

    dim_w2v = int(meta["w2v_dim"])
    dim_sbert = int(meta["sbert_dim"])
    for arquivo, dim in [("w2v_vectors.mmap", dim_w2v), ("sbert_vectors.mmap", dim_sbert)]:
        esperado = n_docs * dim * 4  # float32
        real = (pasta / arquivo).stat().st_size
        if real != esperado:
            raise ValueError(
                f"{arquivo} tem {real:,} bytes, mas eram esperados {esperado:,} "
                f"({n_docs:,} vetores x {dim} dimensões)."
            )
    return n_docs, dim_w2v, dim_sbert


# Avaliação de uma representação (Word2Vec ou SBERT)

def avaliar(nome, codificar, pasta, arquivo_vetores, arquivo_indice, dim, n_docs,
            conn, perguntas, args, efs_ordenados):
    """Roda as perguntas com uma representação e avalia a variação de efSearch."""
    print(f"\n=== {nome} (dimensão {dim}) ===", flush=True)
    k = args.top_k

    vetores = np.memmap(pasta / arquivo_vetores, dtype=np.float32, mode="r",
                        shape=(n_docs, dim))
    indice = faiss.read_index(str(pasta / arquivo_indice), faiss.IO_FLAG_MMAP)
    if indice.ntotal != n_docs or indice.d != dim:
        raise ValueError(
            f"O índice {arquivo_indice} tem {indice.ntotal:,} vetores de dimensão "
            f"{indice.d}; o esperado era {n_docs:,} de dimensão {dim}."
        )

    # a primeira leitura dos arquivos vem do disco e distorceria a medição
    indice.hnsw.efSearch = args.ef
    consulta = codificar("diabetes treatment")
    busca_exata(consulta, vetores, k, args.bloco)
    busca_hnsw(consulta, indice, k)

    resultados = []
    for numero, pergunta in enumerate(perguntas, start=1):
        consulta, t_encoding = medir(codificar, args.repeticoes, pergunta)

        (ids_ex, scores_ex), t_busca_ex = medir(
            busca_exata, args.repeticoes, consulta, vetores, k, args.bloco)
        artigos_ex, t_sqlite_ex = medir(
            buscar_artigos, args.repeticoes, conn, ids_ex, scores_ex)

        ids_ex_set = ids_validos(ids_ex)
        por_ef = {}
        ids_hn_principal = scores_hn_principal = None
        t_busca_hn_principal = 0.0

        for ef_val in efs_ordenados:
            indice.hnsw.efSearch = ef_val
            (ids_hn, scores_hn), t_busca_hn = medir(
                busca_hnsw, args.repeticoes, consulta, indice, k)
            comum_ef = len(ids_ex_set & ids_validos(ids_hn))
            por_ef[ef_val] = {"busca": t_busca_hn, "em_comum": comum_ef}

            if ef_val == args.ef:
                ids_hn_principal = ids_hn
                scores_hn_principal = scores_hn
                t_busca_hn_principal = t_busca_hn

        artigos_hn, t_sqlite_hn = medir(
            buscar_artigos, args.repeticoes, conn, ids_hn_principal, scores_hn_principal)
        em_comum = por_ef[args.ef]["em_comum"]

        resultados.append({
            "numero": numero,
            "pergunta": pergunta,
            "modelo": nome,
            "encoding": t_encoding,
            "em_comum": em_comum,
            "por_ef": por_ef,
            "exato": {"busca": t_busca_ex, "sqlite": t_sqlite_ex,
                      "ids": ids_ex_set, "artigos": artigos_ex},
            "hnsw": {"busca": t_busca_hn_principal, "sqlite": t_sqlite_hn,
                     "ids": ids_validos(ids_hn_principal), "artigos": artigos_hn},
        })
        print(f"  Pergunta {numero}/{len(perguntas)}: exato {t_busca_ex:.2f} ms | "
              f"HNSW (ef={args.ef}) {t_busca_hn_principal:.2f} ms | "
              f"artigos em comum {em_comum}/{k}", flush=True)

    del indice, vetores
    return resultados


# Médias, comparações e persistência
def montar_tabela(resultados, ef):
    """Uma linha por método, com a média (entre as perguntas) de cada fase."""
    linhas = []
    for nome in [W2V, SBERT]:
        rs = [r for r in resultados if r["modelo"] == nome]
        encoding = mean(r["encoding"] for r in rs)
        for chave, rotulo, complexidade in [
            ("exato", "Cosseno Exato", "O(N·d)"),
            ("hnsw", f"HNSW (ef={ef})", "~O(log N·d)"),
        ]:
            busca = mean(r[chave]["busca"] for r in rs)
            sqlite = mean(r[chave]["sqlite"] for r in rs)
            linhas.append({
                "modelo": nome, "metodo": rotulo, "complexidade": complexidade,
                "encoding": encoding, "busca": busca, "sqlite": sqlite,
                "total": encoding + busca + sqlite,
            })
    return linhas


def resumir_por_ef(resultados, efs_ordenados, k):
    """Calcula tempo médio e Recall@K médio para cada valor de efSearch."""
    resumo = {}
    for nome in [W2V, SBERT]:
        rs = [r for r in resultados if r["modelo"] == nome]
        resumo[nome] = []
        for ef_val in efs_ordenados:
            tempo_medio = mean(r["por_ef"][ef_val]["busca"] for r in rs)
            comum_medio = mean(r["por_ef"][ef_val]["em_comum"] for r in rs)
            resumo[nome].append({
                "ef": ef_val,
                "busca_ms": tempo_medio,
                "em_comum": comum_medio,
                "recall_pct": comum_medio / k * 100.0,
            })
    return resumo


def sobreposicao_w2v_sbert(resultados):
    """Quantos artigos Word2Vec e SBERT têm em comum (busca exata) por pergunta."""
    w2v = {r["numero"]: r["exato"]["ids"] for r in resultados if r["modelo"] == W2V}
    sbert = {r["numero"]: r["exato"]["ids"] for r in resultados if r["modelo"] == SBERT}
    return {n: len(w2v[n] & sbert[n]) for n in w2v if n in sbert}


def carregar_referencia_10k(caminho_ref, n_docs_atual):
    """Carrega metricas.json da amostra de 10k quando o benchmark roda na base de 10 GB."""
    if n_docs_atual < 1_000_000 or not caminho_ref:
        return None
    p = Path(caminho_ref)
    if not p.is_file():
        return None
    try:
        dados = json.loads(p.read_text(encoding="utf-8"))
        if dados.get("num_docs", 0) < 1_000_000 and len(dados.get("tabela", [])) == 4:
            return dados
    except (OSError, json.JSONDecodeError):
        return None
    return None


def gerar_texto_resumo(resultados, tabela, resumo_ef, args, n_docs, meta, ref_10k=None):
    """Texto com as médias e comparações (usado no terminal e no relatório)."""
    k = args.top_k
    largura = 86
    linhas = []
    adicionar = linhas.append

    def titulo(texto):
        adicionar("")
        adicionar(texto)
        adicionar("-" * largura)

    n_perguntas = len({r["numero"] for r in resultados})
    adicionar("RESULTADOS DO BENCHMARK DE BUSCA SEMÂNTICA")
    adicionar("=" * largura)
    adicionar(f"Base: {numero_br(n_docs)} artigos  |  Top-{k}  |  efSearch principal = {args.ef}  |  "
              f"{n_perguntas} perguntas")
    adicionar(f"Cada tempo = mediana de {args.repeticoes} repetições; "
              "depois, média entre as perguntas.")

    titulo("1) TEMPO MÉDIO POR MÉTODO (ms)")
    adicionar(f"{'Método':<40}{'Complexid.':<13}{'Encoding':>9}{'Busca':>10}{'SQLite':>9}{'Total':>10}")
    for i, linha in enumerate(tabela):
        rotulo = f"{linha['modelo']} + {linha['metodo']}"
        adicionar(f"{rotulo:<40}{linha['complexidade']:<13}{ms_br(linha['encoding']):>9}"
                  f"{ms_br(linha['busca']):>10}{ms_br(linha['sqlite']):>9}"
                  f"{ms_br(linha['total']):>10}")
        if i % 2 == 1 and i < len(tabela) - 1:
            adicionar("")

    titulo("2) QUANTAS VEZES O HNSW É MAIS RÁPIDO QUE O EXATO (só a busca vetorial)")
    for nome, (exato, hnsw) in zip([W2V, SBERT], [tabela[0:2], tabela[2:4]]):
        ganho = exato["busca"] / max(hnsw["busca"], 1e-9)
        adicionar(f"  {nome:<22}{numero_br(ganho, 1):>9}x   "
                  f"({ms_br(exato['busca'])} ms -> {ms_br(hnsw['busca'])} ms)")

    titulo("3) PESO DA VETORIZAÇÃO DA PERGUNTA NO TEMPO TOTAL (com HNSW)")
    for nome, linha in zip([W2V, SBERT], [tabela[1], tabela[3]]):
        adicionar(f"  {nome:<22}{linha['encoding'] / linha['total'] * 100:>8.0f}%   "
                  f"({ms_br(linha['encoding'])} ms de {ms_br(linha['total'])} ms)")

    titulo(f"4) ARTIGOS DO HNSW QUE COINCIDEM COM O COSSENO EXATO (ef={args.ef}, de {k})")
    adicionar(f"  {'Pergunta':<12}{W2V:<24}{SBERT}")
    por_modelo = {nome: {r["numero"]: r["em_comum"] for r in resultados if r["modelo"] == nome}
                  for nome in [W2V, SBERT]}
    for numero in sorted(por_modelo[W2V]):
        adicionar(f"  {numero:<12}{str(por_modelo[W2V][numero]) + '/' + str(k):<24}"
                  f"{por_modelo[SBERT][numero]}/{k}")
    adicionar("")
    for nome in [W2V, SBERT]:
        media = mean(por_modelo[nome].values())
        adicionar(f"  Média {nome:<22}{numero_br(media, 1)} de {k} ({media / k * 100:.0f}%)")

    if len(resumo_ef[W2V]) > 1:
        titulo(f"5) VARIAÇÃO DO PARÂMETRO efSearch NO HNSW (Top-{k})")
        adicionar(f"  {'Representação':<24}{'efSearch':>10}{'Busca (ms)':>14}"
                  f"{'Coincidência':>16}{'Recall@' + str(k):>12}")
        for nome in [W2V, SBERT]:
            for item in resumo_ef[nome]:
                coinc = f"{numero_br(item['em_comum'], 1)}/{k}"
                rec = f"{numero_br(item['recall_pct'], 1)}%"
                adicionar(f"  {nome:<24}{item['ef']:>10}{ms_br(item['busca_ms']):>14}"
                          f"{coinc:>16}{rec:>12}")
            if nome == W2V:
                adicionar("")

    sobre = sobreposicao_w2v_sbert(resultados)
    if sobre:
        media = mean(sobre.values())
        titulo(f"6) ARTIGOS EM COMUM ENTRE WORD2VEC E SBERT (busca exata, de {k})")
        adicionar("  Por pergunta: " + ", ".join(str(sobre[n]) for n in sorted(sobre)))
        adicionar(f"  Média: {numero_br(media, 1)} de {k} (quanto menor, mais diferentes "
                  "são os resultados)")

    fases = [(rotulo, meta[chave]) for chave, rotulo in FASES_CONSTRUCAO
             if isinstance(meta.get(chave), (int, float))]
    if fases:
        titulo("7) TEMPO DE CONSTRUÇÃO (registrado no metadata.json)")
        for rotulo, segundos in fases:
            adicionar(f"  {rotulo:<38}{numero_br(segundos, 2):>10} s")

    if ref_10k:
        n_10k = ref_10k["num_docs"]
        fator_n = n_docs / max(n_10k, 1)
        titulo(f"8) ESCALABILIDADE: {numero_br(n_10k)} vs. {numero_br(n_docs)} ARTIGOS "
               f"({numero_br(fator_n, 0)}x mais dados)")
        adicionar(f"  {'Método':<36}{'10k (ms)':>12}{'10 GB (ms)':>13}{'Crescimento':>14}")
        for l_10k, l_10g in zip(ref_10k["tabela"], tabela):
            rotulo = f"{l_10g['modelo']} + {l_10g['metodo']}"
            cresc = l_10g["busca"] / max(l_10k["busca"], 1e-9)
            adicionar(f"  {rotulo:<36}{ms_br(l_10k['busca']):>12}{ms_br(l_10g['busca']):>13}"
                      f"{numero_br(cresc, 1) + 'x':>14}")

    return "\n".join(linhas)


def escrever_relatorio(caminho, resumo, resultados, n_artigos=5, max_chars=0):
    """Salva o resumo e os artigos retornados por cada método, pergunta por pergunta."""
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(resumo + "\n\n")
        f.write("=" * 90 + f"\nARTIGOS ENCONTRADOS (até {n_artigos} por método)\n" + "=" * 90 + "\n")

        perguntas = sorted({(r["numero"], r["pergunta"]) for r in resultados})
        for numero, pergunta in perguntas:
            f.write(f"\n[Pergunta {numero}] {pergunta}\n")
            for nome in [W2V, SBERT]:
                r = next(x for x in resultados if x["numero"] == numero and x["modelo"] == nome)
                for chave, rotulo in [("exato", "Cosseno Exato"), ("hnsw", "HNSW")]:
                    f.write(f"\n  {nome} + {rotulo}\n")
                    for artigo in r[chave]["artigos"][:n_artigos]:
                        f.write(f"    #{artigo['rank']} PMID {artigo['pmid']} | "
                                f"cos={artigo['score']:.4f} | {artigo['titulo']}\n")
                        sinopse = artigo["sinopse"]
                        if max_chars and len(sinopse) > max_chars:
                            sinopse = sinopse[:max_chars].rstrip() + "..."
                        f.write(textwrap.fill(sinopse, width=100, initial_indent="       ",
                                              subsequent_indent="       ") + "\n")
            f.write("\n" + "-" * 90 + "\n")


def salvar_artigos_json(caminho, resultados):
    """Salva todos os artigos retornados (com sinopse completa) em JSON, para reuso no LLM/RAGAS."""
    saida = []
    for r in resultados:
        for chave, metodo in [("exato", "Cosseno Exato"), ("hnsw", "HNSW")]:
            saida.append({
                "pergunta_numero": r["numero"],
                "pergunta": r["pergunta"],
                "representacao": r["modelo"],
                "metodo": metodo,
                "artigos": r[chave]["artigos"],
            })
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)


def salvar_metricas_json(caminho, n_docs, args, tabela, resumo_ef, resultados):
    """Salva os números agregados em JSON para permitir comparação automática entre escalas."""
    por_modelo = {nome: {r["numero"]: r["em_comum"] for r in resultados if r["modelo"] == nome}
                  for nome in [W2V, SBERT]}
    payload = {
        "num_docs": n_docs,
        "top_k": args.top_k,
        "ef_principal": args.ef,
        "repeticoes": args.repeticoes,
        "tabela": tabela,
        "resumo_por_ef": resumo_ef,
        "recall_medio": {
            nome: mean(por_modelo[nome].values()) / args.top_k * 100.0
            for nome in [W2V, SBERT]
        },
    }
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



# Gráficos
def gerar_graficos(pasta_saida, tabela, resumo_ef, resultados, n_docs, k, ef,
                   sufixo="", ref_10k=None):
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    cores = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    # Gráfico 1: tempo médio da busca vetorial nos 4 métodos.
    nomes = ["Word2Vec\nCosseno Exato\nO(N·d)", "Word2Vec\nHNSW\n~O(log N·d)",
             "Sentence Emb.\nCosseno Exato\nO(N·d)", "Sentence Emb.\nHNSW\n~O(log N·d)"]
    tempos = [linha["busca"] for linha in tabela]
    rotulos = []
    for posicao, tempo in enumerate(tempos):
        texto = f"{ms_br(tempo, 3)} ms"
        if posicao % 2 == 1:
            ganho = tempos[posicao - 1] / max(tempo, 1e-9)
            texto += f"\n{numero_br(ganho, 1)}x mais rápido"
        rotulos.append(texto)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    barras = ax.bar(nomes, tempos, color=cores, width=0.55)
    ax.bar_label(barras, labels=rotulos, padding=3, fontsize=9)
    ax.set_title(
        f"Tempo médio de busca vetorial\n"
        f"(N = {numero_br(n_docs)} artigos | Top-{k} | ef = {ef})",
        fontweight="bold")
    if min(tempos) > 0 and max(tempos) / min(tempos) > 50:
        ax.set_yscale("log")
        ax.set_ylabel("Tempo médio (ms) - escala log")
        ax.set_ylim(min(tempos) * 0.5, max(tempos) * 6)
    else:
        ax.set_ylabel("Tempo médio (ms)")
        ax.set_ylim(0, max(tempos) * 1.3)
    fig.text(0.5, 0.005, "Tempos com cache aquecido (mediana das repetições).",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    fig.savefig(pasta_saida / f"grafico_1_tempo_busca{sufixo}.png", dpi=200)
    plt.close(fig)

    # Gráfico 2: quantos artigos do HNSW coincidem com o cosseno exato.
    numeros = sorted({r["numero"] for r in resultados})
    em_comum = {nome: {r["numero"]: r["em_comum"] for r in resultados if r["modelo"] == nome}
                for nome in [W2V, SBERT]}
    medias = {nome: mean(em_comum[nome].values()) for nome in [W2V, SBERT]}
    x = np.arange(len(numeros))
    largura = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - largura / 2, [em_comum[W2V][n] for n in numeros], largura, color=cores[1],
                label=f"Word2Vec + HNSW (média {numero_br(medias[W2V], 1)} de {k})")
    b2 = ax.bar(x + largura / 2, [em_comum[SBERT][n] for n in numeros], largura, color=cores[3],
                label=f"Sentence Emb. + HNSW (média {numero_br(medias[SBERT], 1)} de {k})")
    ax.bar_label(b1, fmt="%d", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%d", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{n}" for n in numeros])
    ax.set_yticks(range(0, k + 1))
    ax.set_ylim(0, k * 1.3)
    ax.set_xlabel("Pergunta")
    ax.set_ylabel(f"Artigos em comum com o cosseno exato (de {k})")
    ax.set_title(f"HNSW vs. cosseno exato: artigos coincidentes no Top-{k} (ef={ef})",
                 fontweight="bold")
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(pasta_saida / f"grafico_2_hnsw_vs_exato{sufixo}.png", dpi=200)
    plt.close(fig)

    # Gráfico 3: quanto tempo cada etapa de uma consulta leva (pipeline com HNSW).
    etapas = [("encoding", "Vetorizar\na pergunta", "#4C72B0"),
              ("busca", "Buscar os\nartigos", "#55A868"),
              ("sqlite", "Ler título e\nsinopse (SQLite)", "#DD8452")]
    pipelines = [tabela[1], tabela[3]]

    fig, eixos = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, linha in zip(eixos, pipelines):
        valores = [linha[chave] for chave, _, _ in etapas]
        barras = ax.bar([nome for _, nome, _ in etapas], valores,
                        color=[cor for _, _, cor in etapas], width=0.6)
        ax.bar_label(barras, padding=3, fontsize=10,
                     labels=[f"{ms_br(v, 3)} ms" if v < 1 else f"{ms_br(v, 1)} ms"
                             for v in valores])

        mais_lenta = max(range(len(etapas)), key=lambda i: valores[i])
        nome_etapa = etapas[mais_lenta][1].replace("\n", " ")
        percentual = valores[mais_lenta] / linha["total"] * 100
        modelo = linha["modelo"].replace(" Average", "")

        ax.set_title(f"{modelo} + HNSW\nTotal: {ms_br(linha['total'], 2)} ms",
                     fontweight="bold")
        ax.set_xlabel(f"Etapa mais lenta: {nome_etapa} ({percentual:.0f}% do total)")
        ax.set_ylabel("Tempo (ms)")
        ax.set_ylim(0, max(valores) * 1.2)
        casas = 2 if max(valores) < 1 else 1
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, c=casas: numero_br(v, c)))

    fig.suptitle("Tempo de cada etapa de uma consulta", fontweight="bold", fontsize=13)
    fig.text(0.5, 0.01,
             "Atenção: cada gráfico tem a sua própria escala de tempo. "
             "Tempos com cache aquecido; não incluem a geração da resposta pelo LLM.",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(pasta_saida / f"grafico_3_fases_consulta{sufixo}.png", dpi=200)
    plt.close(fig)

    # Gráfico 4: impacto do parâmetro efSearch no tempo e no Recall@K.
    if len(resumo_ef[W2V]) > 1:
        efs = [item["ef"] for item in resumo_ef[W2V]]
        x_ef = np.arange(len(efs))
        fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(11, 4.8))

        for nome, cor, rotulo in [(W2V, cores[1], "Word2Vec + HNSW"),
                                  (SBERT, cores[3], "Sentence Emb. + HNSW")]:
            tempos_ef = [item["busca_ms"] for item in resumo_ef[nome]]
            recalls_ef = [item["recall_pct"] for item in resumo_ef[nome]]

            ax_t.plot(x_ef, tempos_ef, marker="o", linewidth=2, color=cor, label=rotulo)
            for xi, yi in zip(x_ef, tempos_ef):
                ax_t.annotate(f"{ms_br(yi, 3)} ms", (xi, yi), textcoords="offset points",
                              xytext=(0, 7), ha="center", fontsize=8.5)

            ax_r.plot(x_ef, recalls_ef, marker="s", linewidth=2, color=cor, label=rotulo)
            for xi, yi in zip(x_ef, recalls_ef):
                ax_r.annotate(f"{numero_br(yi, 1)}%", (xi, yi), textcoords="offset points",
                              xytext=(0, 7), ha="center", fontsize=8.5)

        ax_t.set_title("Latência da busca HNSW", fontweight="bold")
        ax_t.set_xticks(x_ef)
        ax_t.set_xticklabels([f"ef={v}" for v in efs])
        ax_t.set_ylabel("Tempo médio (ms)")
        max_t = max(max(x["busca_ms"] for x in resumo_ef[W2V]),
                    max(x["busca_ms"] for x in resumo_ef[SBERT]))
        ax_t.set_ylim(0, max_t * 1.25)
        ax_t.legend(fontsize=9)

        ax_r.set_title(f"Recall@{k} vs. Cosseno Exato", fontweight="bold")
        ax_r.set_xticks(x_ef)
        ax_r.set_xticklabels([f"ef={v}" for v in efs])
        ax_r.set_ylabel(f"Recall@{k} (%)")
        min_r = min(min(x["recall_pct"] for x in resumo_ef[W2V]),
                    min(x["recall_pct"] for x in resumo_ef[SBERT]))
        ax_r.set_ylim(max(0, min_r - 15), 108)
        ax_r.legend(fontsize=9)

        fig.suptitle(f"Trade-off do parâmetro efSearch (N = {numero_br(n_docs)} artigos)",
                     fontweight="bold", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(pasta_saida / f"grafico_4_variacao_ef{sufixo}.png", dpi=200)
        plt.close(fig)

    # Gráfico 5: comparação de escalabilidade 10k vs. 10 GB.
    if ref_10k:
        n_10k = ref_10k["num_docs"]
        tempos_10k = [l["busca"] for l in ref_10k["tabela"]]
        tempos_10g = [l["busca"] for l in tabela]
        x_esc = np.arange(4)
        w_esc = 0.36

        fig, ax = plt.subplots(figsize=(10, 5.4))
        b_10k = ax.bar(x_esc - w_esc / 2, tempos_10k, w_esc, color="#7F8C8D",
                       label=f"Amostra ({numero_br(n_10k)} artigos)")
        b_10g = ax.bar(x_esc + w_esc / 2, tempos_10g, w_esc, color="#2E86C1",
                       label=f"Base completa ({numero_br(n_docs)} artigos)")

        ax.bar_label(b_10k, labels=[f"{ms_br(v, 3)} ms" for v in tempos_10k], padding=3, fontsize=8)
        rot_10g = [
            f"{ms_br(v_g, 3)} ms\n({numero_br(v_g / max(v_k, 1e-9), 1)}x)"
            for v_k, v_g in zip(tempos_10k, tempos_10g)
        ]
        ax.bar_label(b_10g, labels=rot_10g, padding=3, fontsize=8)

        ax.set_yscale("log")
        ax.set_xticks(x_esc)
        ax.set_xticklabels(nomes)
        ax.set_ylabel("Tempo médio de busca (ms) - escala log")
        ax.set_ylim(min(tempos_10k + tempos_10g) * 0.4, max(tempos_10k + tempos_10g) * 8)
        ax.set_title(
            f"Escalabilidade: {numero_br(n_10k)} vs. {numero_br(n_docs)} artigos "
            f"({numero_br(n_docs / max(n_10k, 1), 0)}x mais vetores)",
            fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(pasta_saida / "grafico_5_escala_10k_vs_10gb.png", dpi=200)
        plt.close(fig)


def ler_argumentos():
    p = argparse.ArgumentParser(
        description="Compara cosseno exato e HNSW com Word2Vec Average e Sentence Embeddings.")
    p.add_argument("--db", default="data/pubmed_10k_sample.db",
                   help="Banco SQLite (padrão: amostra de 10 mil artigos)")
    p.add_argument("--indices", default="indices/indice_pubmed_10k",
                   help="Pasta criada pelo build_indices.py")
    p.add_argument("--saida", "--out", dest="saida", default="results/sample_10k",
                   help="Pasta onde os gráficos e o relatório serão salvos")
    p.add_argument("--top-k", type=int, default=5, help="Artigos retornados por pergunta")
    p.add_argument("--ef", "--ef-search", "--ef-principal", dest="ef", type=int, default=64,
                   help="Valor principal de efSearch do HNSW (padrão: 64)")
    p.add_argument("--efs", type=int, nargs="+", default=[16, 64, 128],
                   help="Valores de efSearch para análise de sensibilidade (padrão: 16 64 128)")
    p.add_argument("--ref-10k", default="results/sample_10k/metricas.json",
                   help="JSON de métricas da amostra de 10k para comparar escala na base 10 GB")
    p.add_argument("--repeticoes", type=int, default=5,
                   help="Repetições de cada medição (usa-se a mediana, para reduzir ruído)")
    p.add_argument("--perguntas", default=None,
                   help="Arquivo .txt opcional, com uma pergunta em inglês por linha")
    p.add_argument("--pergunta", "--query", dest="pergunta", nargs="+", default=None,
                   help="Uma ou mais perguntas em inglês direto pelo terminal")
    p.add_argument("--bloco", type=int, default=100_000,
                   help="Vetores por bloco na busca exata (reduza se faltar RAM)")
    p.add_argument("--artigos-relatorio", type=int, default=5,
                   help="Quantos artigos de cada método aparecem no relatório (padrão: 5)")
    p.add_argument("--sinopse-chars", type=int, default=0,
                   help="Corta a sinopse no relatório (0 = sinopse completa)")
    p.add_argument("--dispositivo", default="cpu",
                   help="Dispositivo do SBERT para vetorizar a pergunta (cpu ou cuda)")
    args = p.parse_args()

    if min(args.top_k, args.ef, args.repeticoes, args.bloco) < 1 or any(v < 1 for v in args.efs):
        p.error("top-k, ef, efs, repeticoes e bloco devem ser maiores que zero.")
    return args


def main():
    args = ler_argumentos()
    pasta = Path(args.indices)
    saida = Path(args.saida)
    efs_ordenados = sorted(set(args.efs) | {args.ef})

    if not Path(args.db).is_file():
        raise FileNotFoundError(f"Banco não encontrado: {args.db}")
    if not (pasta / "metadata.json").is_file():
        raise FileNotFoundError(f"metadata.json não encontrado em '{pasta}'")

    with open(pasta / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    conn = sqlite3.connect(args.db)
    n_docs, dim_w2v, dim_sbert = validar(conn, pasta, meta)

    if args.pergunta:
        perguntas = [q.strip() for q in args.pergunta if q.strip()]
    elif args.perguntas:
        with open(args.perguntas, "r", encoding="utf-8-sig") as f:
            perguntas = [linha.strip() for linha in f if linha.strip()]
    else:
        perguntas = PERGUNTAS

    print(f"Base: {n_docs:,} artigos | Top-{args.top_k} | efSearch principal = {args.ef} "
          f"(variação: {efs_ordenados}) | {args.repeticoes} repetições")

    # Word2Vec primeiro. Perguntas sem nenhuma palavra do vocabulário são descartadas,
    # porque o vetor delas seria zero e o resultado não teria sentido.
    print("Carregando Word2Vec...", flush=True)
    w2v = Word2Vec.load(str(pasta / "word2vec.model"))
    if w2v.vector_size != dim_w2v:
        raise ValueError("A dimensão do modelo Word2Vec não bate com o metadata.json.")

    validas = []
    for pergunta in perguntas:
        if any(token in w2v.wv for token in tokenize(pergunta)):
            validas.append(pergunta)
        else:
            print(f"Aviso: pergunta ignorada (nenhuma palavra no vocabulário): {pergunta}")
    if not validas:
        raise ValueError("Nenhuma pergunta tem palavras conhecidas pelo Word2Vec.")
    perguntas = validas

    def codificar_w2v(texto):
        return compute_w2v_doc_vector(tokenize(texto), w2v.wv, dim_w2v)

    resultados = avaliar(W2V, codificar_w2v, pasta, "w2v_vectors.mmap", "hnsw_w2v.index",
                         dim_w2v, n_docs, conn, perguntas, args, efs_ordenados)
    del w2v

    # Depois o SBERT.
    nome_modelo = meta.get("sbert_model", "sentence-transformers/all-MiniLM-L6-v2")
    if not Path(nome_modelo).exists() and "all-MiniLM-L6-v2" in nome_modelo:
        nome_modelo = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\nCarregando SBERT ({nome_modelo}) em {args.dispositivo}...", flush=True)
    sbert = SentenceTransformer(nome_modelo, device=args.dispositivo)

    try:
        dim_modelo = sbert.get_embedding_dimension()
    except AttributeError:
        dim_modelo = sbert.get_sentence_embedding_dimension()
    if dim_modelo != dim_sbert:
        raise ValueError("A dimensão do modelo SBERT não bate com o metadata.json.")

    def codificar_sbert(texto):
        vetor = sbert.encode([texto], normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False)[0]
        return np.asarray(vetor, dtype=np.float32)

    resultados += avaliar(SBERT, codificar_sbert, pasta, "sbert_vectors.mmap",
                          "hnsw_sbert.index", dim_sbert, n_docs, conn, perguntas, args, efs_ordenados)
    del sbert
    conn.close()

    # Resultados
    saida.mkdir(parents=True, exist_ok=True)
    sufixo = "_10gb" if n_docs >= 1_000_000 else ""
    tabela = montar_tabela(resultados, args.ef)
    resumo_ef = resumir_por_ef(resultados, efs_ordenados, args.top_k)
    ref_10k = carregar_referencia_10k(args.ref_10k, n_docs)
    resumo = gerar_texto_resumo(resultados, tabela, resumo_ef, args, n_docs, meta, ref_10k)

    escrever_relatorio(saida / f"relatorio{sufixo}.txt", resumo, resultados,
                       args.artigos_relatorio, args.sinopse_chars)
    salvar_artigos_json(saida / f"artigos_encontrados{sufixo}.json", resultados)
    salvar_metricas_json(saida / f"metricas{sufixo}.json", n_docs, args, tabela, resumo_ef, resultados)
    gerar_graficos(saida, tabela, resumo_ef, resultados, n_docs, args.top_k, args.ef, sufixo, ref_10k)

    print("\n" + resumo)
    print("=" * 86)
    arquivos_gerados = [
        f"grafico_1_tempo_busca{sufixo}.png",
        f"grafico_2_hnsw_vs_exato{sufixo}.png",
        f"grafico_3_fases_consulta{sufixo}.png",
    ]
    if len(efs_ordenados) > 1:
        arquivos_gerados.append(f"grafico_4_variacao_ef{sufixo}.png")
    if ref_10k:
        arquivos_gerados.append("grafico_5_escala_10k_vs_10gb.png")
    arquivos_gerados += [
        f"relatorio{sufixo}.txt",
        f"artigos_encontrados{sufixo}.json",
        f"metricas{sufixo}.json",
    ]
    print(f"\nSalvos em '{saida}': {', '.join(arquivos_gerados)}")


if __name__ == "__main__":
    main()