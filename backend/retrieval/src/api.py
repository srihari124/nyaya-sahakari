import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from rag.src.service import generate_answer, preload_rag_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Nyaya-Sahakari API",
    description="Legal AI Search & Answer System",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
def preload_models_on_startup():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    logger.info("Startup preload started")
    preload_rag_service()
    logger.info("Startup preload completed")


@app.get("/")
def health():
    return {"status": "ok", "message": "Nyaya-Sahakari API is running"}


@app.post("/search")
def search(request: QueryRequest):
    if not request.query or not request.query.strip():
        return {
            "error": "Query cannot be empty"
        }

    result = generate_answer(request.query)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "meta": {
            "source_count": len(result["sources"])
        }
    }
