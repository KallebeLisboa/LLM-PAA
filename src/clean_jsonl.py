import argparse
import html
import json
import os
import re
import time

# MedRAG/pubmed removeu o caractere '>' de várias tags XML originais (ex: <bResults, <sub3</sub, <ip, <mml:miN</mml:mi).
# As regex abaixo removem apenas as marcas XML/HTML mantendo o conteúdo numérico/matemático intacto (ex: N = 65, H0, p < 0.05).
COMPLETE_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:-]*\s*/?>")
MML_OPEN_ATTR_RE = re.compile(r'<mml:math(?:\s+xmlns:mml="[^"]*")?')
MML_TAG_RE = re.compile(
    r"</?mml:(?:msub|msup|msubsup|mrow|mfrac|msqrt|mroot|mover|munder|munderover|mtable|mtr|mtd|mspace|mtext|mstyle|math|mi|mn|mo)/?"
)
BROKEN_MEDRAG_TAG_RE = re.compile(r"</?(?:sub|sup|strong|em|br|i|b|u)")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove tags HTML/XML/MathML (completas ou quebradas) preservando valores e fórmulas."""
    text = html.unescape(text)
    text = COMPLETE_TAG_RE.sub("", text)
    text = MML_OPEN_ATTR_RE.sub("", text)
    text = MML_TAG_RE.sub("", text)
    text = BROKEN_MEDRAG_TAG_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_jsonl(input_path: str, output_path: str, min_content_len: int = 250):
    t0 = time.perf_counter()
    total_in = 0
    total_out = 0
    skipped = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total_in += 1
            doc = json.loads(line)

            title = clean_text(doc.get("title", ""))
            content = clean_text(doc.get("content", ""))

            if not title or len(content) < min_content_len:
                skipped += 1
                continue

            clean_doc = {
                "id": doc.get("id"),
                "PMID": doc.get("PMID"),
                "title": title,
                "content": content,
            }
            fout.write(json.dumps(clean_doc, ensure_ascii=False) + "\n")
            total_out += 1

            if total_in % 500000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  Processados {total_in:,} artigos ({elapsed:.1f}s)...", flush=True)

    elapsed = time.perf_counter() - t0
    out_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"\n[Limpeza Concluída em {elapsed:.2f}s]")
    print(f"  Entrada : {input_path} ({total_in:,} artigos)")
    print(f"  Saída   : {output_path} ({total_out:,} artigos | {out_gb:.2f} GiB)")
    print(f"  Descartados (< {min_content_len} chars após limpeza): {skipped:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limpa tags HTML/XML quebradas e entidades em arquivos JSONL do PubMed.")
    parser.add_argument("--input", required=True, help="Caminho do .jsonl bruto")
    parser.add_argument("--output", required=True, help="Caminho do .jsonl limpo")
    parser.add_argument("--min-len", type=int, default=250, help="Tamanho mínimo da sinopse em caracteres")
    args = parser.parse_args()

    clean_jsonl(args.input, args.output, min_content_len=args.min_len)
