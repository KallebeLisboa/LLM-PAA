# Projeto de Disciplina — PAA 2/2026 (UnB)

**Sistema de Perguntas e Respostas com Busca Semântica e LLM Local (RAG)**

Este repositório reúne o código do projeto de **Projeto e Análise de Algoritmos (2/2026)**. O sistema responde perguntas em linguagem natural sobre **Medicina e Ciências da Saúde** usando **6.530.506 artigos do PubMed (`MedRAG/pubmed`, 10,2 GB de texto puro)**, comparando algoritmos de busca semântica e formatando a resposta com citação obrigatória (`PMID` + título) via LLM local.

---

## Team

- > **Name**: Gabriel de Castro Dias
  > **Matrícula**: 211055432
- > **Name**: Kaillany Pereira Santos  
  > **Matrícula**: 211055488
- > **Name**: Luiz Otávio Damasceno Maciel  
  > **Matrícula**: 160134951
- > **Name**: Rian Kallebe da Silva Lisboa
  > **Matrícula**: 242012000
- > **Name**: Suyanne Sara Miranda Silva  
  > **Matrícula**: 222006445

---

## 📦 1. Onde baixar os Bancos e Vetores prontos (Google Drive)

Como o GitHub não suporta arquivos grandes (> 100 MB), os bancos SQLite e os vetores já processados estão disponíveis no link abaixo:

- **Amostra Leve (10.000 artigos):** _(recomendada para desenvolvimento e testes rápidos no seu PC)_
- **Base Completa (10,2 GB / 6.530.506 artigos):**
- **Resultados da Vetorização:**
  - Google Drive: `https://drive.google.com/drive/folders/1w6ol7FVonv32KxHI1a3Qq8dNEi3A-4pN?usp=sharing`

---

## ⚙️ 2. Como configurar o ambiente Python

1. **Clone o repositório e crie o ambiente virtual:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   # .venv\Scripts\activate       # Windows
   ```

2. **Instale o PyTorch** (versão CPU é suficiente para rodar buscas/avaliação/interface):

   ```bash
   pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

   _(Para GPU AMD no Linux com ROCm: `pip install torch --index-url https://download.pytorch.org/whl/rocm6.4`)_

3. **Instale as dependências do projeto:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🗂️ 3. Arquivos já implementados no repositório (Pipeline de Dados)

| Arquivo                    | O que faz                                                                                                                                                                                                |
| :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/download_datasets.py` | Baixa seletivamente os shards mais recentes do `MedRAG/pubmed` até atingir ~10,2 GB (~6,5M artigos).                                                                                                     |
| `src/clean_jsonl.py`       | Remove tags HTML/XML/MathML quebradas do PubMed preservando fórmulas (`p < 0.05`, `N = 65`), decodifica entidades e normaliza espaços em streaming ($O(1)$ RAM).                                         |
| `src/build_sqlite.py`      | Converte o `.jsonl` limpo em um banco SQLite (`articles`) com chave primária `doc_id` (Árvore B+, busca $O(\log N)$) e índice por `pmid`.                                                                |
| `src/build_indices.py`     | Treina o modelo **Word2Vec** em streaming, calcula os vetores normalizados **Word2Vec Average** e **Sentence Embeddings (`all-MiniLM-L6-v2`)** em `numpy.memmap`, e constrói os grafos **HNSW** (FAISS). |

### 📊 Números reais já gerados (para os slides e relatórios)

| Métrica                                                     |     Amostra (`10k`)     |    Base Completa (`10 GB`)     |
| :---------------------------------------------------------- | :---------------------: | :----------------------------: |
| **Total de artigos (`num_docs`)**                           |        `10.000`         |        **`6.530.506`**         |
| **Vocabulário Word2Vec**                                    |     `32.392` termos     |      **`776.189`** termos      |
| **Vetores `w2v_vectors.mmap` (`float32`, dim `100`)**       |        `3,9 MB`         |         **`2,5 GiB`**          |
| **Vetores `sbert_vectors.mmap` (`float32`, dim `384`)**     |         `15 MB`         |         **`9,4 GiB`**          |
| **Índice `hnsw_w2v.index` (`M=16`, `efConstruction=64`)**   |   `5,2 MB` (`0,31s`)    |         **`3,4 GiB`**          |
| **Índice `hnsw_sbert.index` (`M=16`, `efConstruction=64`)** | `17 MB` (`0,21s`, FP32) | **`5,6 GiB`** (`452,8s`, FP16) |

---

## 🔌 4. Contrato dos Dados (Como usar os arquivos prontos no seu código)

Todo o grupo trabalha sobre o mesmo padrão de `doc_id` (`0` até `num_docs - 1`):

1. **Banco SQLite (`pubmed_10k_sample.db` ou `pubmed_10gb.db`):**
   ```sql
   -- Tabela: articles
   -- Colunas: doc_id (INTEGER PRIMARY KEY), orig_id (TEXT), pmid (TEXT), title (TEXT), content (TEXT)
   SELECT pmid, title, content FROM articles WHERE doc_id = ?;
   ```
2. **Matrizes de Vetores (`.mmap` — já normalizados L2, ou seja, produto interno = Similaridade de Cosseno!):**
   ```python
   import numpy as np
   # Para 10k: n = 10000 | Para 10GB: n = 6530506
   w2v_vecs   = np.memmap("w2v_vectors.mmap",   dtype=np.float32, mode="r", shape=(n, 100))
   sbert_vecs = np.memmap("sbert_vectors.mmap", dtype=np.float32, mode="r", shape=(n, 384))
   ```
3. **Índices HNSW (`FAISS`):**
   ```python
   import faiss
   index = faiss.read_index("hnsw_sbert.index", faiss.IO_FLAG_MMAP)
   index.hnsw.efSearch = 64
   scores, doc_ids = index.search(query_vector, k=5)
   ```
   > ⚠️ **Atenção à RAM na base de 10 GB:** `hnsw_w2v.index` (`3,4 GiB`) e `hnsw_sbert.index` (`5,6 GiB`) juntos consomem ~9 GiB de RAM. Em notebooks com 8 GB de RAM, carregue **um índice por vez** (ou desenvolva na amostra `10k` e rode o benchmark final dos 10 GB em máquina com 16 GB de RAM).

---

## 👥 5. Guia do que cada membro deve fazer (Entrega: 30/11)

### 2. Membro 2 — Análise de Complexidade (PAA), Banco e Licenças _(Slides 2, 3 e 6)_

- **O que fazer:**
  1. Ler arquivos (por exemplo:`src/clean_jsonl.py`, `src/build_sqlite.py` e `src/build_indices.py`) e formalizar a **complexidade de tempo e espaço** de cada etapa (limpeza $O(N \cdot L)$ com RAM $O(1)$, Árvore B+ do SQLite $O(\log N)$, normalização L2 antecipada, construção do HNSW).
  2. Levantar as **licenças** de todas as linguagens, bibliotecas e modelos utilizados (ver `requirements.txt` + Ollama + RAGAS/DeepEval + Streamlit/FastAPI).

### 3. Membro 3 — Buscas Semânticas e Benchmarks _(Slides 7 e 11)_

- **O que fazer:**
  1. Criar o script de busca/benchmark (ex.: `src/search_benchmark.py`) usando os arquivos `.mmap`, `.index` e `.db` descritos na **Seção 4**.
  2. Comparar os **4 cenários**:
     - `Word2Vec Average` + **Similaridade de Cosseno Exata** ($O(N \cdot d)$)
     - `Word2Vec Average` + **HNSW Search** ($O(\log N \cdot d)$)
     - `Sentence Embeddings` + **Similaridade de Cosseno Exata** ($O(N \cdot d)$)
     - `Sentence Embeddings` + **HNSW Search** ($O(\log N \cdot d)$)
  3. Medir tempo de busca (`ms`), _Recall@K_ (HNSW vs. Cosseno Exato), variar parâmetros (`efSearch`) e gerar tabelas/gráficos comparando `10k` vs. `10 GB`.

### 4. Membro 4 — LLMs Locais (Ollama) e Avaliação (RAGAS / DeepEval) _(Slides 8, 12 e 13)_

- **O que fazer:**
  1. Instalar o **[Ollama](https://ollama.com/)** e baixar modelos leves citados no PDF (`smollm2:1.7b`, `tinyllama`, `deepseek-r1:1.5b`, `phi3` ou `mistral`).
  2. Criar o módulo de RAG (ex.: `src/rag_llm.py`) que recebe os _Top-K_ artigos retornados pela busca semântica e monta o _prompt_ exigindo **citação explícita de `PMID` e Título**.
  3. Montar um conjunto de perguntas médicas de teste e rodar **RAGAS** ou **DeepEval** para medir acurácia, fidelidade à fonte (_faithfulness_) e tempo de inferência entre os modelos.

### 5. Membro 5 — Servidor / Interface Web e Entrega Final _(Slides 1, 4, 9 e 14)_

- **O que fazer:**
  1. Criar uma interface web simples em **Streamlit** ou **FastAPI** onde o usuário digita a pergunta, escolhe o algoritmo de busca e vê a resposta da LLM com citações clicáveis (`https://pubmed.ncbi.nlm.nih.gov/<PMID>/`) e tempo de resposta.
  2. Coordenar a montagem dos slides de 10 minutos (original + PDF) e gerar o `.zip` final (**até 100 MB**, contendo o código + amostra `10k` + link para os 10 GB) para envio no Aprender3.

> **Documentação Técnica e Execução:** Para detalhes sobre como rodar os benchmarks, testes e scripts individuais, consulte [`src/README.md`](src/README.md).

---
