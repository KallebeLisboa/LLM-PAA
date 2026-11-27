
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag_llm import responder_pergunta
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="PubMed Assistant API",
    description="API para o sistema de perguntas e respostas com RAG.",
    version="1.0.0"
)

# Permite a comunicação com o frontend durante o desenvolvimento.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PerguntaRequest(BaseModel):
    pergunta: str = Field(min_length=1)
    metodo_busca: str = "hnsw"
    modelo: str = "phi3:mini"
    top_k: int = Field(default=3, ge=1, le=10)


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


@app.get("/chat", include_in_schema=False)
def pagina_chat():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/perguntar")
def perguntar(request: PerguntaRequest):

    if not request.pergunta.strip():
        raise HTTPException(
            status_code=400,
            detail="A pergunta não pode estar vazia."
        )

    resposta = responder_pergunta(
        pergunta=request.pergunta.strip(),
        metodo_busca=request.metodo_busca,
        modelo=request.modelo,
        top_k=request.top_k
    )

    return {"resposta": resposta}
