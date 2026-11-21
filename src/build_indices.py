import argparse
import json
import os
import re
import sqlite3
import time
import numpy as np
import faiss
import torch
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

# Lista enxuta de stopwords em inglês para evitar que palavras genéricas dominem a média do Word2Vec
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "there", "these", "they", "this", "to", "was",
    "were", "which", "who", "will", "with", "we", "our", "can", "may", "also"
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokeniza texto em minúsculas removendo stopwords e tokens de 1 caractere. Complexidade: O(L)."""
    return [
        tok for tok in TOKEN_PATTERN.findall(text.lower())
        if len(tok) > 1 and tok not in STOPWORDS
    ]


class SqliteCorpusIterator:
    """Iterador em streaming sobre o banco SQLite para treinar Word2Vec com consumo O(1) de RAM."""
    def __init__(self, db_path: str, limit: int | None = None):
        self.db_path = db_path
        self.limit = limit

    def __iter__(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        query = "SELECT title, content FROM articles ORDER BY doc_id"
        if self.limit:
            query += f" LIMIT {int(self.limit)}"
        cur.execute(query)
        for title, content in cur:
            yield tokenize(f"{title} {content}")
        conn.close()


def get_num_docs(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM articles;")
    count = cur.fetchone()[0]
    conn.close()
    return count


def compute_w2v_doc_vector(tokens: list[str], kv, dim: int) -> np.ndarray:
    """Calcula o vetor médio (Word2Vec Average) normalizado L2 para um documento."""
    vecs = [kv[w] for w in tokens if w in kv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    mean_vec = np.mean(vecs, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm
    return mean_vec


def build_hnsw_chunked(mmap_array: np.memmap, dim: int, hnsw_m: int, ef_construction: int, out_path: str, use_fp16: bool = False, chunk_size: int = 50000):
    """Constrói o índice HNSW em blocos para não duplicar a matriz inteira na memória RAM."""
    num_docs = mmap_array.shape[0]
    if use_fp16:
        # Scalar Quantizer FP16 corta o consumo de RAM do HNSW pela metade (~5.5 GB para 6.3M vetores)
        index = faiss.IndexHNSWSQ(dim, faiss.ScalarQuantizer.QT_fp16, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        sample_n = min(num_docs, 100000)
        index.train(np.array(mmap_array[:sample_n], dtype=np.float32))
    else:
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction

    t0 = time.perf_counter()
    for start in range(0, num_docs, chunk_size):
        end = min(start + chunk_size, num_docs)
        chunk = np.array(mmap_array[start:end], dtype=np.float32)
        index.add(chunk)
        if end % 500000 == 0 or end == num_docs:
            elapsed = time.perf_counter() - t0
            print(f"    HNSW inseridos {end:,}/{num_docs:,} ({elapsed:.1f}s)...", flush=True)

    faiss.write_index(index, out_path)
    return round(time.perf_counter() - t0, 2)


def build_all_indices(
    db_path: str,
    out_dir: str,
    w2v_dim: int = 100,
    w2v_epochs: int = 3,
    sbert_model_name: str = "models/all-MiniLM-L6-v2",
    hnsw_m: int = 16,
    ef_construction: int = 64,
    batch_size: int = 1024,
    encode_batch: int = 256,
    use_fp16_hnsw: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_docs = get_num_docs(db_path)
    # Em bases grandes (> 1M artigos), ativa FP16 no HNSW automaticamente para caber com folga em 16 GB RAM
    if num_docs > 1_000_000:
        use_fp16_hnsw = True

    print(f"=== Construção de Índices para {num_docs:,} artigos ({db_path}) ===", flush=True)
    print(f"    Dispositivo PyTorch: {device.upper()} | HNSW FP16: {use_fp16_hnsw} | M={hnsw_m}", flush=True)

    stats = {
        "db_path": db_path,
        "num_docs": num_docs,
        "device": device,
        "w2v_dim": w2v_dim,
        "w2v_epochs": w2v_epochs,
        "sbert_model": sbert_model_name,
        "sbert_dim": 384,
        "hnsw_M": hnsw_m,
        "hnsw_efConstruction": ef_construction,
        "hnsw_fp16": use_fp16_hnsw,
    }

    # -------------------------------------------------------------------------
    # 1. Treinamento do Modelo Word2Vec + Vetores Médios (Word2Vec Average)
    # -------------------------------------------------------------------------
    w2v_model_path = os.path.join(out_dir, "word2vec.model")
    w2v_mmap_path = os.path.join(out_dir, "w2v_vectors.mmap")

    if os.path.exists(w2v_model_path):
        print("\n[1/4] Carregando modelo Word2Vec já treinado do checkpoint...", flush=True)
        w2v_model = Word2Vec.load(w2v_model_path)
        stats["w2v_vocab_size"] = len(w2v_model.wv)
        print(f"  Word2Vec carregado | Vocabulário: {stats['w2v_vocab_size']:,} termos.", flush=True)
    else:
        print(f"\n[1/4] Treinando modelo Word2Vec ({w2v_epochs} épocas) via streaming do SQLite...", flush=True)
        t0 = time.perf_counter()
        corpus_iter = SqliteCorpusIterator(db_path)
        w2v_model = Word2Vec(
            sentences=corpus_iter,
            vector_size=w2v_dim,
            window=5,
            min_count=5 if num_docs > 100_000 else 3,
            workers=max(1, (os.cpu_count() or 4) - 1),
            epochs=w2v_epochs,
        )
        w2v_model.save(w2v_model_path)
        stats["w2v_train_time_s"] = round(time.perf_counter() - t0, 2)
        stats["w2v_vocab_size"] = len(w2v_model.wv)
        print(f"  Word2Vec treinado em {stats['w2v_train_time_s']}s | Vocabulário: {stats['w2v_vocab_size']:,} termos.", flush=True)

    expected_w2v_bytes = num_docs * w2v_dim * 4
    w2v_mmap = None
    if os.path.exists(w2v_mmap_path) and os.path.getsize(w2v_mmap_path) == expected_w2v_bytes:
        cand = np.memmap(w2v_mmap_path, dtype=np.float32, mode="r+", shape=(num_docs, w2v_dim))
        if np.linalg.norm(cand[-1]) > 0:
            w2v_mmap = cand
            print("[2/4] Vetores 'Word2Vec Average' já completos no disco (checkpoint OK).", flush=True)

    if w2v_mmap is None:
        print("[2/4] Calculando vetores 'Word2Vec Average' para todos os documentos (memmap)...", flush=True)
        t0 = time.perf_counter()
        w2v_mmap = np.memmap(w2v_mmap_path, dtype=np.float32, mode="w+", shape=(num_docs, w2v_dim))
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT doc_id, title, content FROM articles ORDER BY doc_id;")
        kv = w2v_model.wv
        for doc_id, title, content in cur:
            tokens = tokenize(f"{title} {content}")
            w2v_mmap[doc_id] = compute_w2v_doc_vector(tokens, kv, w2v_dim)
            if (doc_id + 1) % 500000 == 0:
                w2v_mmap.flush()
                print(f"  Word2Vec vetorizados {doc_id + 1:,}/{num_docs:,} ({time.perf_counter() - t0:.1f}s)...", flush=True)
        w2v_mmap.flush()
        conn.close()
        stats["w2v_vectorize_time_s"] = round(time.perf_counter() - t0, 2)
        print(f"  Vetores Word2Vec Average salvos em {stats['w2v_vectorize_time_s']}s.", flush=True)

    # -------------------------------------------------------------------------
    # 2. Geração de Sentence Embeddings (SBERT) com GPU + FP16 + Checkpoint
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Gerando Sentence Embeddings com '{sbert_model_name}' em {device.upper()}...", flush=True)
    t0 = time.perf_counter()
    sbert_model = SentenceTransformer(sbert_model_name, device=device)
    if device == "cuda":
        sbert_model.half()  # FP16 na GPU (2x-3x mais rápido e metade da VRAM na RX 6600M)

    sbert_dim = sbert_model.get_embedding_dimension()
    stats["sbert_dim"] = sbert_dim

    sbert_mmap_path = os.path.join(out_dir, "sbert_vectors.mmap")
    expected_sbert_bytes = num_docs * sbert_dim * 4

    start_doc_id = 0
    if os.path.exists(sbert_mmap_path) and os.path.getsize(sbert_mmap_path) == expected_sbert_bytes:
        sbert_mmap = np.memmap(sbert_mmap_path, dtype=np.float32, mode="r+", shape=(num_docs, sbert_dim))
        # Busca rápida do ponto de retomada em blocos para não estourar RAM em 6.3M linhas
        for blk_start in range(0, num_docs, 100000):
            blk_end = min(blk_start + 100000, num_docs)
            norms = np.linalg.norm(sbert_mmap[blk_start:blk_end], axis=1)
            zeros = np.where(norms < 1e-5)[0]
            if len(zeros) > 0:
                start_doc_id = blk_start + int(zeros[0])
                break
        else:
            start_doc_id = num_docs

        if start_doc_id > 0:
            print(f"  Checkpoint detectado: retomando do artigo {start_doc_id:,}/{num_docs:,}...", flush=True)
    else:
        sbert_mmap = np.memmap(sbert_mmap_path, dtype=np.float32, mode="w+", shape=(num_docs, sbert_dim))

    if start_doc_id < num_docs:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT doc_id, title, content FROM articles WHERE doc_id >= ? ORDER BY doc_id;",
            (start_doc_id,),
        )
        batch_texts = []
        batch_ids = []
        processed_now = 0

        for doc_id, title, content in cur:
            batch_ids.append(doc_id)
            batch_texts.append(f"{title}. {content}")
            if len(batch_texts) >= batch_size:
                emb = sbert_model.encode(
                    batch_texts,
                    batch_size=encode_batch,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                sbert_mmap[batch_ids[0]:batch_ids[-1] + 1] = emb.astype(np.float32)
                sbert_mmap.flush()
                processed_now += len(batch_texts)
                batch_texts.clear()
                batch_ids.clear()
                if (doc_id + 1) % 20000 == 0 or (num_docs <= 20000 and (doc_id + 1) % 2000 == 0):
                    elapsed = time.perf_counter() - t0
                    rate = processed_now / max(elapsed, 1e-6)
                    rem_s = (num_docs - (doc_id + 1)) / max(rate, 1e-6)
                    print(
                        f"  Codificados {doc_id + 1:,}/{num_docs:,} ({rate:,.0f} docs/s | ETA: {rem_s/60:.1f} min)",
                        flush=True,
                    )

        if batch_texts:
            emb = sbert_model.encode(
                batch_texts,
                batch_size=encode_batch,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            sbert_mmap[batch_ids[0]:batch_ids[-1] + 1] = emb.astype(np.float32)
            sbert_mmap.flush()
            batch_texts.clear()
            batch_ids.clear()

        conn.close()

    stats["sbert_vectorize_time_s"] = round(time.perf_counter() - t0, 2)
    print(f"  Sentence Embeddings concluídos em {stats['sbert_vectorize_time_s']}s.", flush=True)

    # -------------------------------------------------------------------------
    # 3. Construção dos Índices HNSW (Em blocos para respeitar limite de 16 GB RAM)
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Construindo índices HNSW (M={hnsw_m}, efConstruction={ef_construction}, fp16={use_fp16_hnsw})...", flush=True)

    hnsw_w2v_path = os.path.join(out_dir, "hnsw_w2v.index")
    if not os.path.exists(hnsw_w2v_path):
        print("  -> Construindo HNSW para Word2Vec Average...", flush=True)
        stats["hnsw_w2v_build_time_s"] = build_hnsw_chunked(
            w2v_mmap, w2v_dim, hnsw_m, ef_construction, hnsw_w2v_path, use_fp16=False
        )
        print(f"  HNSW (Word2Vec) construído em {stats['hnsw_w2v_build_time_s']}s.", flush=True)
    else:
        print("  HNSW (Word2Vec) já existe (checkpoint OK).", flush=True)

    hnsw_sbert_path = os.path.join(out_dir, "hnsw_sbert.index")
    if not os.path.exists(hnsw_sbert_path):
        print("  -> Construindo HNSW para Sentence Embeddings...", flush=True)
        stats["hnsw_sbert_build_time_s"] = build_hnsw_chunked(
            sbert_mmap, sbert_dim, hnsw_m, ef_construction, hnsw_sbert_path, use_fp16=use_fp16_hnsw
        )
        print(f"  HNSW (SBERT) construído em {stats['hnsw_sbert_build_time_s']}s.", flush=True)
    else:
        print("  HNSW (SBERT) já existe (checkpoint OK).", flush=True)

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n=== Tudo Pronto! Metadados salvos em {meta_path} ===", flush=True)


if __name__ == "__main__":
    default_sbert = (
        "models/all-MiniLM-L6-v2"
        if os.path.isdir("models/all-MiniLM-L6-v2")
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    parser = argparse.ArgumentParser(description="Constrói Word2Vec Average, Sentence Embeddings e índices HNSW.")
    parser.add_argument("--db", default="data/pubmed_sample.db", help="Caminho do banco SQLite")
    parser.add_argument("--out", default="indices/sample_10k", help="Diretório de saída dos índices")
    parser.add_argument("--sbert-model", default=default_sbert, help="Modelo SentenceTransformer (local ou HF)")
    parser.add_argument("--w2v-epochs", type=int, default=3, help="Épocas de treino do Word2Vec (padrão: 3)")
    parser.add_argument("--hnsw-m", type=int, default=16, help="Parâmetro M do grafo HNSW (padrão: 16)")
    parser.add_argument("--ef-construction", type=int, default=64, help="efConstruction do HNSW (padrão: 64)")
    parser.add_argument("--encode-batch", type=int, default=256, help="Tamanho do lote na GPU/CPU para SBERT")
    args = parser.parse_args()

    build_all_indices(
        args.db,
        args.out,
        w2v_epochs=args.w2v_epochs,
        sbert_model_name=args.sbert_model,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        encode_batch=args.encode_batch,
    )
