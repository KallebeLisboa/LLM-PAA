<!-- ./src/README.md -->

# `./src/` Folder

# Buscas Semânticas e Benchmarks (`src/search_benchmark.py`)

O script `src/search_benchmark.py` executa e avalia a recuperação de artigos biomédicos comparando **Word2Vec Average** e **Sentence Embeddings (`all-MiniLM-L6-v2`)** sob **Similaridade de Cosseno Exata** e **HNSW (FAISS)**, gerando métricas de tempo, _Recall@5_, relatórios e gráficos.

---

## Como rodar

### A) Rodar a bateria completa de benchmarks (10 perguntas padrão)

Substitua os caminhos entre `<...>` pelos diretórios onde você salvou os arquivos baixados do Google Drive:

**(Linux / macOS):**

```bash
python src/search_benchmark.py \
    --db "<CAMINHO_PARA_O_BANCO.db>" \
    --indices "<PASTA_DOS_INDICES>" \
    --efs 16 32 64 128 256 \
    --saida "<PASTA_DE_SAIDA>"
```

_(Para comparar a base completa de `10 GB` com os resultados da amostra `10k`, adicione `--ref-10k "<CAMINHO_PARA_metricas.json>"`.)_

**Exemplos (Windows PowerShell):**

```powershell
# 1) Roda a amostra de 10k e salva tudo em results\sample_10k
python src/search_benchmark.py `
    --db "data\pubmed_10k_sample.db" `
    --indices "indices\indice_pubmed_10k" `
    --efs 16 32 64 128 256 `
    --saida "results\sample_10k"

# 2) Roda a base completa de 10 GB, cruza com a de 10k e salva em results\full_10gb
python src/search_benchmark.py `
    --db "10gb dados\Arquivo em Sqlite\pubmed_10gb.db" `
    --indices "10gb dados\Resultado_da_Vetorizacao" `
    --efs 16 32 64 128 256 `
    --ref-10k "results\sample_10k\metricas.json" `
    --saida "results\full_10gb"
```

---

### B) Testar perguntas médicas avulsas pelo terminal

Use a flag `--pergunta` (ou `--query`) para passar uma ou mais perguntas diretamente na linha de comando, ou `--perguntas` para apontar para um arquivo `.txt` (uma pergunta em inglês por linha):

**Windows PowerShell:**

```powershell
python src/search_benchmark.py `
    --db "data\pubmed_10k_sample.db" `
    --indices "indices\indice_pubmed_10k" `
    --efs 64 `
    --pergunta "What are the cardiovascular risks associated with chronic kidney disease?" `
    --saida "results\teste"
```

**Linux / macOS:**

```bash
python src/search_benchmark.py \
    --db "data/pubmed_10k_sample.db" \
    --indices "indices/indice_pubmed_10k" \
    --efs 64 \
    --pergunta "What are the cardiovascular risks associated with chronic kidney disease?" \
    --saida "results/teste"
```

---

### C) Sugestões de perguntas biomédicas para testar

- _"How does chronic sleep deprivation affect insulin resistance and type 2 diabetes risk?"_
- _"What are the neuroprotective mechanisms of regular aerobic exercise against Alzheimer's disease?"_
- _"Which biomarkers predict early acute kidney injury after cardiac surgery?"_
- _"How do immune checkpoint inhibitors cause myocarditis in cancer patients?"_
- _"What is the efficacy of monoclonal antibodies in treating severe asthma?"_
