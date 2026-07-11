"""End-to-end local RAG pipeline for Amazon quarterly financial reports.

This script downloads multiple Amazon investor relations PDF reports, extracts
text with PyPDF, chunks the content using multiple strategies, builds
embeddings with the open-source `BAAI/bge-small-en-v1.5` model, and compares
cosine-similarity retrieval against an HNSW index via ChromaDB. The pipeline is
self-contained so it can be executed locally or in environments like Google
Colab.
#pip install pypdf sentence-transformers chromadb python-dotenv requests
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


try:
    import chromadb
    #from chromadb.config import Settings
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'chromadb' package is required. Install it with 'pip install chromadb'."
    ) from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("amazon_rag")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    logger.info("Loaded OPENAI_API_KEY from environment (value not displayed).")
else:
    logger.warning("OPENAI_API_KEY is not set. It's not required for this local pipeline.")

CHROMA_DIR = Path("chroma_store")
CHROMA_DIR.mkdir(exist_ok=True)

COMPANY_NAME = "Amazon"
DEFAULT_TOP_K = 3

LOCAL_PDF_DIR = Path("amazon_annual_report")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DocumentChunk:
    """Represents a single chunk of text and its metadata."""

    chunk_id: str
    text: str
    page_number: int
    chunking_method: str
    report_name: str
    company_name: str
    source_url: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Return a filesystem-friendly slug for the provided text."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "chunk"


def read_local_pdf(source: str) -> bytes:
    """Read PDF bytes from a local filesystem path."""

    source_path = Path(source)
    if source_path.exists() and source_path.is_file():
        logger.info("Reading local PDF: %s", source_path)
        return source_path.read_bytes()
    raise FileNotFoundError(f"Local PDF source not found: {source}")


def discover_local_pdf_sources(pdf_dir: Path) -> Dict[str, str]:
    """Build source mapping from local PDF directory."""

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        return {}

    mapping: Dict[str, str] = {}
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        report_name = pdf_path.stem.replace("-", " ")
        mapping[report_name] = str(pdf_path)
    return mapping


def extract_pdf_pages(pdf_bytes: bytes) -> List[str]:
    """Extract text from each page of a PDF."""

    reader = PdfReader(BytesIO(pdf_bytes))

    pages: List[str] = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.replace("\u2019", "'").replace("\u2014", "-")
        pages.append(text)
        logger.debug("Extracted %d characters from page %d", len(text), index + 1)
    return pages


def sliding_window_chunk(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks using a fixed window size."""

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    tokens = cleaned.split(" ")
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        if end >= len(tokens):
            break
        start = max(start + chunk_size - overlap, start + 1)
    return chunks


def recursive_character_chunk(
    text: str,
    *,
    chunk_size: int = 600,
    min_chunk_size: int = 200,
) -> List[str]:
    """Approximate recursive character splitter without external dependencies."""

    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    output: List[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            output.append(paragraph)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        buffer = ""
        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    output.append(buffer)
                if len(sentence) <= chunk_size:
                    buffer = sentence
                else:
                    for idx in range(0, len(sentence), chunk_size):
                        segment = sentence[idx : idx + chunk_size]
                        if len(segment) < min_chunk_size and output:
                            output[-1] += " " + segment
                        else:
                            output.append(segment)
                    buffer = ""
        if buffer:
            output.append(buffer)
    return output


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------


def build_corpus(
    pdf_sources: Dict[str, str],
    chunkers: Sequence[tuple[str, Callable[[str], List[str]]]],
) -> tuple[List[DocumentChunk], List[str]]:
    """Download PDFs, extract text, and create chunks using provided chunkers."""

    chunks: List[DocumentChunk] = []
    diagnostics: List[str] = []

    for report_name, source in pdf_sources.items():
        try:
            pdf_bytes = read_local_pdf(source)
            pages = extract_pdf_pages(pdf_bytes)
        except Exception as exc:  # pragma: no cover - network dependent
            message = f"Failed to process {report_name}: {exc}"
            logger.error(message)
            diagnostics.append(message)
            continue

        extracted_chars = sum(len(page.strip()) for page in pages)
        logger.info(
            "Extracted %d characters from %s across %d pages",
            extracted_chars,
            report_name,
            len(pages),
        )
        if extracted_chars == 0:
            diagnostics.append(
                f"{report_name}: extracted zero text from PDF pages (likely scanned/image-only or unavailable text layer)."
            )

        slug = slugify(report_name)
        report_chunk_counter = 0
        for page_index, page_text in enumerate(pages):
            page_number = page_index + 1
            for chunking_method, chunker in chunkers:
                cleaned_page = page_text.strip()
                if not cleaned_page:
                    continue
                method_chunks = chunker(cleaned_page)
                for chunk_idx, chunk_text in enumerate(method_chunks):
                    if not chunk_text.strip():
                        continue
                    chunk_id = f"{slug}-{chunking_method}-p{page_number}-{chunk_idx}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            page_number=page_number,
                            chunking_method=chunking_method,
                            report_name=report_name,
                            company_name=COMPANY_NAME,
                            source_url=source,
                        )
                    )
                    report_chunk_counter += 1
        logger.info("Prepared %d chunks from %s", report_chunk_counter, report_name)
        if report_chunk_counter == 0:
            diagnostics.append(
                f"{report_name}: no chunks created after extraction/chunking."
            )

    logger.info("Total chunks prepared: %d", len(chunks))
    return chunks, diagnostics


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


def embed_chunks(
    model: SentenceTransformer,
    chunks: Sequence[DocumentChunk],
) -> np.ndarray:
    """Generate normalized embeddings for each chunk."""

    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------


def cosine_retrieval(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunks: Sequence[DocumentChunk],
    top_k: int,
) -> List[Dict[str, object]]:
    """Retrieve top-k chunks using cosine similarity on pre-computed embeddings."""

    scores = chunk_embeddings @ query_embedding
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results: List[Dict[str, object]] = []
    for idx in ranked_indices:
        chunk = chunks[int(idx)]
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "score": float(scores[idx]),
                "text": chunk.text,
                "page_number": chunk.page_number,
                "report_name": chunk.report_name,
                "company_name": chunk.company_name,
                "chunking_method": chunk.chunking_method,
            }
        )
    return results


def build_chroma_collection(
    chunks: Sequence[DocumentChunk],
    embeddings: np.ndarray,
    persist_directory: Path,
) -> chromadb.api.models.Collection.Collection:
    """Create (or recreate) a Chroma collection backed by HNSW."""

    if persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_directory))

    collection_name = "amazon_reports"
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "page_number": chunk.page_number,
                "report_name": chunk.report_name,
                "company_name": chunk.company_name,
                "chunking_method": chunk.chunking_method,
                "source_url": chunk.source_url,
            }
            for chunk in chunks
        ],
    )

    return collection


def chroma_hnsw_retrieval(
    collection: chromadb.api.models.Collection.Collection,
    query_embedding: np.ndarray,
    top_k: int,
) -> List[Dict[str, object]]:
    """Retrieve top-k chunks using Chroma's HNSW index."""

    response = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results: List[Dict[str, object]] = []
    distances = response.get("distances", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]

    for distance, document, metadata in zip(distances, documents, metadatas):
        similarity = 1.0 - float(distance)
        results.append(
            {
                "score": similarity,
                "text": document,
                "page_number": metadata.get("page_number"),
                "report_name": metadata.get("report_name"),
                "company_name": metadata.get("company_name"),
                "chunking_method": metadata.get("chunking_method"),
                "source_url": metadata.get("source_url"),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_final_answer(
    query: str,
    cosine_results: Sequence[Dict[str, object]],
    hnsw_results: Sequence[Dict[str, object]],
) -> str:
    """Construct a simple answer string based on top retrievals."""

    sections: List[str] = [f"Query: {query}"]

    if cosine_results:
        top = cosine_results[0]
        sections.append(
            (
                "[Cosine similarity]\n"
                f"Report: {top['report_name']} (page {top['page_number']}, {top['chunking_method']})\n"
                f"Snippet: {top['text']}"
            )
        )
    if hnsw_results:
        top = hnsw_results[0]
        sections.append(
            (
                "[Chroma HNSW]\n"
                f"Report: {top['report_name']} (page {top['page_number']}, {top['chunking_method']})\n"
                f"Snippet: {top['text']}"
            )
        )

    if len(sections) == 1:
        sections.append("No supporting information retrieved.")

    return "\n\n".join(sections)


def initialize_pipeline() -> Dict[str, object]:
    """Prepare corpus, embeddings, and index once for interactive querying."""

    logger.info("Preparing chunkers...")
    chunkers: Sequence[tuple[str, Callable[[str], List[str]]]] = (
        (
            "sliding_window",
            lambda text: sliding_window_chunk(text, chunk_size=250, overlap=40),
        ),
        (
            "recursive",
            lambda text: recursive_character_chunk(text, chunk_size=400, min_chunk_size=120),
        ),
    )

    logger.info("Collecting and chunking documents...")
    active_sources = discover_local_pdf_sources(LOCAL_PDF_DIR)
    logger.info("Using %d local PDFs from %s", len(active_sources), LOCAL_PDF_DIR.resolve())

    chunks, diagnostics = build_corpus(active_sources, chunkers)
    if not chunks:
        logger.warning("No chunks available for retrieval.")
        final_answer = (
            "I couldn't find relevant chunks for your query because the document corpus is empty. "
            "Please verify your local amazon_annual_report PDFs are present and contain extractable text."
        )
        if diagnostics:
            logger.warning("Diagnostics:")
            for note in diagnostics:
                logger.warning("- %s", note)
        return {
            "query": query,
            "query_embedding": [],
            "cosine_latency_seconds": 0.0,
            "hnsw_latency_seconds": 0.0,
            "cosine_top_k": [],
            "hnsw_top_k": [],
            "diagnostics": diagnostics,
            "final_answer": final_answer,
        }

    logger.info("Loading embedding model (BAAI/bge-small-en-v1.5)...")
    try:
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    except Exception as exc:
        message = (
            "Embedding model could not be loaded. "
            "This is often caused by blocked internet/model download access."
        )
        logger.warning("%s Error: %s", message, exc)
        diagnostics.append(f"Model load failure: {exc}")
        return {
            "query": query,
            "query_embedding": [],
            "cosine_latency_seconds": 0.0,
            "hnsw_latency_seconds": 0.0,
            "cosine_top_k": [],
            "hnsw_top_k": [],
            "diagnostics": diagnostics,
            "final_answer": (
                "I couldn't complete retrieval because the embedding model failed to load. "
                "Please ensure model access is available (or use a locally cached model path)."
            ),
        }

    logger.info("Encoding %d chunks...", len(chunks))
    chunk_embeddings = embed_chunks(model, chunks)

    logger.info("Building Chroma HNSW index...")
    collection = build_chroma_collection(chunks, chunk_embeddings, CHROMA_DIR)

    return {
        "ready": True,
        "model": model,
        "chunks": chunks,
        "chunk_embeddings": chunk_embeddings,
        "collection": collection,
        "diagnostics": diagnostics,
    }


def answer_query(
    state: Dict[str, object],
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, object]:
    """Run retrieval for a single query using prebuilt state."""

    if not state.get("ready"):
        return {
            "query": query,
            "query_embedding": [],
            "cosine_latency_seconds": 0.0,
            "hnsw_latency_seconds": 0.0,
            "cosine_top_k": [],
            "hnsw_top_k": [],
            "diagnostics": state.get("diagnostics", []),
            "final_answer": "Pipeline is not ready. Check diagnostics and fix setup issues.",
        }

    model = state["model"]
    chunks = state["chunks"]
    chunk_embeddings = state["chunk_embeddings"]
    collection = state["collection"]
    diagnostics = list(state.get("diagnostics", []))

    logger.info("Encoding query using the same embedding model...")
    query_embedding = model.encode(query, normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    logger.info("Query embedding dimensionality: %d", query_embedding.shape[0])
    logger.info(
        "Query embedding preview (first 8 dims): %s",
        np.array2string(query_embedding[:8], precision=4, separator=", "),
    )

    cosine_start = time.perf_counter()
    cosine_results = cosine_retrieval(query_embedding, chunk_embeddings, chunks, top_k)
    cosine_latency = time.perf_counter() - cosine_start

    hnsw_start = time.perf_counter()
    hnsw_results = chroma_hnsw_retrieval(collection, query_embedding, top_k)
    hnsw_latency = time.perf_counter() - hnsw_start

    final_answer = build_final_answer(query, cosine_results, hnsw_results)

    result_payload = {
        "query": query,
        "query_embedding": query_embedding.tolist(),
        "cosine_latency_seconds": round(cosine_latency, 4),
        "hnsw_latency_seconds": round(hnsw_latency, 4),
        "cosine_top_k": cosine_results,
        "hnsw_top_k": hnsw_results,
        "diagnostics": diagnostics,
        "final_answer": final_answer,
    }

    logger.info("\nCosine similarity retrieval (latency %.4f s):", cosine_latency)
    for idx, item in enumerate(cosine_results, start=1):
        logger.info(
            "%d) score=%.4f | page=%s | report=%s | method=%s",
            idx,
            item["score"],
            item["page_number"],
            item["report_name"],
            item["chunking_method"],
        )

    logger.info("\nChroma HNSW retrieval (latency %.4f s):", hnsw_latency)
    for idx, item in enumerate(hnsw_results, start=1):
        logger.info(
            "%d) score=%.4f | page=%s | report=%s | method=%s",
            idx,
            item["score"],
            item["page_number"],
            item["report_name"],
            item["chunking_method"],
        )

    logger.info("\nFinal answer:\n%s", final_answer)
    return result_payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _print_stepwise_instructions() -> None:
    """Display step-by-step instructions for running the pipeline."""

    instructions = [
        "1. Ensure dependencies are installed: pip install pypdf sentence-transformers chromadb python-dotenv requests",
        "2. Place a .env file alongside this script containing OPENAI_API_KEY=<your-key> (optional for this local pipeline).",
        "3. Run the script: python RAG_AMAZON.py",
        "4. Modify the QUERY constant below or pass a custom query via input.",
        "5. Review the logged retrieval comparison and final answer.",
    ]
    print("\n".join(instructions))


def main() -> None:
    _print_stepwise_instructions()

    logger.info("\nInitializing RAG pipeline (one-time setup)...")
    state = initialize_pipeline()

    if not state.get("ready"):
        output_path = Path("rag_results.json")
        output_path.write_text(json.dumps(state, indent=2))
        logger.info("\nInitialization failed. Saved diagnostics to %s", output_path.resolve())
        return

    logger.info("Initialization complete. You can now ask multiple queries.")
    logger.info("Type 'exit', 'quit', or 'q' to stop.")

    default_query = "Summarize Amazon's operating income trends in recent quarters."

    while True:
        user_query = input("\nEnter your financial research query (press enter for default):\n> ").strip()
        if user_query.lower() in {"exit", "quit", "q"}:
            logger.info("Exiting interactive query loop.")
            break

        query = user_query or default_query
        logger.info("\nRunning retrieval for query: %s", query)
        results = answer_query(state, query, top_k=DEFAULT_TOP_K)

        output_path = Path("rag_results.json")
        output_path.write_text(json.dumps(results, indent=2))
        logger.info("Saved detailed results to %s", output_path.resolve())


if __name__ == "__main__":
    main()
