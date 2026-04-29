"""
retriever.py

Conecta o ChromaDB ao agente via LangChain BaseRetriever.

A deteccao de empresa e feita por regex sobre a query atual.
Se nenhuma empresa for identificada, a busca roda sem filtro de metadado.
"""

import os
import re
from pathlib import Path
from typing import Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import (AsyncCallbackManagerForRetrieverRun,CallbackManagerForRetrieverRun)
from langchain_community.embeddings import OllamaEmbeddings
from chromadb import PersistentClient
from pydantic import Field


# Configuracoes ****************************************************************


EMBEDDING_MODEL = "qwen3-embedding:4b"
EMBEDDING_NUM_CTX = int(2048)
EMBEDDING_NUM_THREAD = str(os.cpu_count() or 4)
COLLECTION_NAME = "financebench"
BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = str(BASE_DIR / "vector_store")
N_RESULTS = 5

# Secao de deteccao de empresa na query
KNOWN_COMPANIES = [
    "APPLE", "AMAZON", "BOEING", "MICROSOFT", "NIKE"
]

_COMPANY_PATTERN = re.compile(
    r"\b(" + "|".join(KNOWN_COMPANIES) + r")\b",
    re.IGNORECASE,
)

def detect_company(query: str) -> Optional[str]:
    """
    Retorna a empresa mencionada na query, ou None se nenhuma for encontrada.
    O match e feito contra o nome canonico (mesmo valor do metadado no ChromaDB).
    """
    match = _COMPANY_PATTERN.search(query)
    if match:
        return match.group(1).upper()
    return None

# Retriever *************************************************************************

class FinancialRetriever(BaseRetriever):
    """
    Retriever que busca chunks no ChromaDB usando embeddings locais.
    Se a query mencionar uma empresa conhecida, filtra os resultados por ela.
    Caso contrario, busca em toda a colecao.
    """

    collection: object = Field(repr=False)
    embeddings_model: object = Field(repr=False)
    n_results: int= Field(default=N_RESULTS)

    class Config:
        arbitrary_types_allowed = True

    def _search(self, query: str) -> list[Document]:
        query_vector = self.embeddings_model.embed_query(query)
        company      = detect_company(query)

        query_params = {
            "query_embeddings": [query_vector],
            "n_results":self.n_results,
            "include":["documents", "metadatas", "distances"],
        }

        if company:
            query_params["where"] = {"company": company}

        results = self.collection.query(**query_params)

        documents = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            meta["similarity_score"] = round(1 - distance, 4)
            documents.append(Document(page_content=text, metadata=meta))

        return documents

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun
            ) -> list[Document]:
        return self._search(query)

    async def _aget_relevant_documents(self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._search(query)

# Build para o Retriever *************************************************************************

def build_retriever(
    vector_store_dir: str = VECTOR_STORE_DIR,
    collection_name: str = COLLECTION_NAME,
    n_results: int = N_RESULTS,
) -> FinancialRetriever:
    """
    Instancia o retriever conectado ao ChromaDB persistido em disco.
    """
    embeddings_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        num_ctx=EMBEDDING_NUM_CTX,
        num_thread=EMBEDDING_NUM_THREAD,
    )

    chroma = PersistentClient(path=vector_store_dir)
    collection = chroma.get_or_create_collection(collection_name)

    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB esta vazio. Execute a ingestao:\n"
            "  python src/ingestion.py"
        )

    return FinancialRetriever(
        collection=collection,
        embeddings_model=embeddings_model,
        n_results=n_results,
    )