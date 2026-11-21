import argparse
import json
import os
import sqlite3
import time


def build_sqlite(jsonl_path: str, db_path: str, batch_size: int = 10000):
    if os.path.exists(db_path):
        os.remove(db_path)

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Otimizações de escrita em massa para SQLite
    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA temp_store = MEMORY;")
    cur.execute("PRAGMA cache_size = -64000;")  # ~64 MB de cache

    cur.execute("""
        CREATE TABLE articles (
            doc_id INTEGER PRIMARY KEY,
            orig_id TEXT,
            pmid TEXT,
            title TEXT,
            content TEXT
        );
    """)

    start_time = time.perf_counter()
    batch = []
    doc_id = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            batch.append((
                doc_id,
                str(record.get("id", "")),
                str(record.get("PMID", "")),
                record.get("title", ""),
                record.get("content", "")
            ))
            doc_id += 1

            if len(batch) >= batch_size:
                cur.executemany(
                    "INSERT INTO articles (doc_id, orig_id, pmid, title, content) VALUES (?, ?, ?, ?, ?);",
                    batch
                )
                conn.commit()
                batch.clear()
                if doc_id % 100000 == 0:
                    elapsed = time.perf_counter() - start_time
                    print(f"  Inseridos {doc_id:,} artigos ({elapsed:.1f}s)...")

    if batch:
        cur.executemany(
            "INSERT INTO articles (doc_id, orig_id, pmid, title, content) VALUES (?, ?, ?, ?, ?);",
            batch
        )
        conn.commit()
        batch.clear()

    # Cria índice para busca rápida por PMID caso necessário
    cur.execute("CREATE INDEX idx_articles_pmid ON articles(pmid);")
    conn.commit()
    conn.close()

    total_time = time.perf_counter() - start_time
    db_size_mb = os.path.getsize(db_path) / (1024 ** 2)
    print(f"\n[SQLite Concluído]")
    print(f"  Arquivo de origem : {jsonl_path}")
    print(f"  Banco gerado      : {db_path} ({db_size_mb:.2f} MB)")
    print(f"  Total de artigos  : {doc_id:,}")
    print(f"  Tempo total       : {total_time:.2f} s ({doc_id / max(total_time, 1e-6):,.0f} docs/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte arquivo JSONL do PubMed para SQLite indexado por doc_id.")
    parser.add_argument("--input", default="data/amostra_10k.jsonl", help="Caminho do arquivo .jsonl de entrada")
    parser.add_argument("--db", default="data/pubmed_sample.db", help="Caminho do banco SQLite de saída")
    args = parser.parse_args()

    build_sqlite(args.input, args.db)
