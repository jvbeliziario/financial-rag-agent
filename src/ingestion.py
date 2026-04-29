"""
ingestion.py

Lê os PDFs do FinanceBench, aplica chunking com separadores financeiros,
gera embeddings via Qwen3-embedding (Ollama) e persiste no ChromaDB.

"""

import re
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from chromadb import PersistentClient
from tqdm import tqdm

# Configurações ****************************************************************

EMBEDDING_MODEL = "qwen3-embedding:4b"
COLLECTION_NAME = "financebench"
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
VECTOR_STORE_DIR = str(BASE_DIR / "vector_store")
CHUNK_SIZE = 800  # Pensando em parágrafos e tabelas de relatorios financeiros
CHUNK_OVERLAP = 150  # cerca de 20% de overlap para preservar contexto entre chunks
BATCH_SIZE = 20  # chunks por chamada ao Ollama [qwen3-embeddings](levando em conta uso de RAM do modelo)
RESET_INDEX = ("RESET_INDEX", "0") == "1"  # se True, deleta a coleção do ChromaDB antes de indexar,
# caso contrario continua a partir do ultimo id adicionado ao vectorstore.

# Separadores em ordem de prioridade para documentos financeiros:
SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

# Seção de metadados **************************************************************

# Padrões esperados nos nomes de arquivo do FinanceBench:
_FILENAME_RE = re.compile(
    r"^(?P<company>[A-Z]+)_(?P<year>\d{4})(?:Q\d)?_?(?P<doc_type>\w+)?",
    re.IGNORECASE,
)

def extract_metadata_from_path(pdf_path: Path) -> dict:
    """
    Extrai company, year e doc_type do nome do arquivo.

    Metadados essenciais para filtragem no retriever:
    - company: permite filtrar por empresa mencionada na query
    - year: permite filtrar por período relevante
    - doc_type: permite filtrar por tipo de documento (10-K, earnings, etc)
    """
    stem = pdf_path.stem.upper()
    match = _FILENAME_RE.match(stem)

    if match:
        return {
            "company": match.group("company"),
            "year": match.group("year") or "unknown",
            "doc_type": (match.group("doc_type") or "unknown").upper(),
            "source": str(pdf_path),
            "filename": pdf_path.name,
        }

    # Caso não consiga parsear o nome mas ainda indexa, sem filtros
    return {
        "company": "unknown",
        "year": "unknown",
        "doc_type": "unknown",
        "source": str(pdf_path),
        "filename": pdf_path.name,
    }

# PDF Loading e pré-processamento *******************************************************

def load_documents() -> list[Document]:
    """
    Lê todos os PDFs em PDF_DIR com PyMuPDF e injeta metadados financeiros.
    """
    pdfs = list(PDF_DIR.rglob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em {PDF_DIR}.\n"
        )

    documents: list[Document] = []

    for pdf_path in tqdm(pdfs, desc="Carregando PDFs"):
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()
        meta = extract_metadata_from_path(pdf_path)

        for page in pages:
            text = (page.page_content or "").strip()
            if not text:
                continue

            # Mescla metadados do LangChain
            combined_meta = {**page.metadata, **meta}
            documents.append(Document(page_content=text, metadata=combined_meta))

    print(f"\n{len(documents)} páginas carregadas de {len(pdfs)} PDFs")
    return documents


# Chunking ***********************************************************************

def split_documents(documents: list[Document]) -> list[Document]:
    """
    Aplica RecursiveCharacterTextSplitter com separadores calibrados para
    documentos financeiros.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks gerados | tamanho alvo: {CHUNK_SIZE} chars | overlap: {CHUNK_OVERLAP}")
    return chunks


# Embeddings e armazenamento no ChromaDB ************************************************

def build_embeddings_model() -> OllamaEmbeddings:
    """
    OllamaEmbeddings roteia para o servidor Ollama local.
    Nenhuma API key necessária — apenas o Ollama rodando em OLLAMA_BASE_URL.
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )


def embed_and_store(chunks: list[Document], embeddings_model: OllamaEmbeddings) -> None:
    """
    Gera embeddings em batches e persiste no ChromaDB via upsert.
    """
    chroma = PersistentClient(path=VECTOR_STORE_DIR)

    if RESET_INDEX:
        existing = [c.name for c in chroma.list_collections()]
        if COLLECTION_NAME in existing:
            chroma.delete_collection(COLLECTION_NAME)
            print(f"Coleção '{COLLECTION_NAME}' resetada")

    collection = chroma.get_or_create_collection(COLLECTION_NAME)
    failed_count = 0
    skipped_chunks = 0

    # Se o script crashar no meio e você reexecutar com RESET_INDEX=0, basicamente seguirá de onde parou,
    # pulando os chunks que já existem no ChromaDB
    has_existing = (not RESET_INDEX) and (collection.count() > 0)

    print(f"\nIndexando {len(chunks)} chunks em batches de {BATCH_SIZE}...")

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.page_content for c in batch]
        metas = [c.metadata for c in batch]
        # IDs sequenciais simples — upsert garante registro pra re-execuções
        ids = [f"chunk_{i + j}" for j in range(len(batch))]

        if has_existing:
            try:
                existing = collection.get(ids=ids)
                existing_ids = set(existing.get("ids", []) or [])
            except Exception:
                existing_ids = set()

            if existing_ids:
                filtered = [
                    (doc_id, text, meta)
                    for (doc_id, text, meta) in zip(ids, texts, metas)
                    if doc_id not in existing_ids
                ]

                if not filtered:
                    skipped_chunks += len(ids)
                    continue

                ids, texts, metas = zip(*filtered)
                ids = list(ids)
                texts = list(texts)
                metas = list(metas)
                skipped_chunks += (len(batch) - len(ids))

        try:
            vectors = embeddings_model.embed_documents(texts)
            collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metas,
            )
        except Exception as e:
            # continue preserva o progresso dos batches anteriores
            failed_count += 1
            print(f"\nBatch {i}–{i + len(batch)} falhou (ignorado): {e}")
            continue

    total = collection.count()
    print(f"\nIndexação concluída: {total} chunks no ChromaDB")
    print(f"Vector store: {VECTOR_STORE_DIR}")
    print(f"Coleção: {COLLECTION_NAME}")

    if failed_count:
        print(
            f"\n{failed_count} batch(es) falharam e foram ignorados.\n"
            "   Reexecute com RESET_INDEX=0 para tentar apenas os chunks faltantes."
        )

    if skipped_chunks:
        print(f"\n{skipped_chunks} chunks já existiam e foram pulados (sem recalcular embeddings)")


# main ************************************************************************

if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings_model = build_embeddings_model()
    embed_and_store(chunks, embeddings_model)

    print("\nIngestão completa!")
