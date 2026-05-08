"""Ingest cybersecurity PDFs into a local ChromaDB collection.

Usage:
    python ingest.py
    python ingest.py --reset
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import chromadb
import requests
from dotenv import load_dotenv
from pypdf import PdfReader

try:
    from llama_index.core.node_parser import SentenceSplitter
except ImportError:  # pragma: no cover
    SentenceSplitter = None

ROOT = Path(__file__).parent
PAPERS_DIR = ROOT / "papers"
DEFAULT_CHROMA_PATH = ROOT / "chroma_db"

TOPIC_KEYWORDS = {
    "malware": ["malware", "ransomware", "trojan", "botnet", "worm", "virus"],
    "intrusion detection": ["intrusion", "ids", "anomaly detection", "network detection"],
    "phishing": ["phishing", "credential", "spoof", "social engineering"],
    "vulnerability": ["vulnerability", "exploit", "cve", "patch", "zero-day"],
    "privacy": ["privacy", "anonym", "differential privacy", "data leakage"],
    "cryptography": ["cryptography", "encryption", "cipher", "key exchange", "signature"],
    "web security": ["web", "xss", "csrf", "sql injection", "browser"],
    "machine learning security": ["adversarial", "poisoning", "model", "classifier", "neural"],
}


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def infer_topic(text: str) -> str:
    haystack = text.lower()
    scores = {
        topic: sum(1 for keyword in keywords if keyword in haystack)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    best_topic, best_score = max(scores.items(), key=lambda item: item[1])
    return best_topic if best_score else "general cybersecurity"


def split_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> Iterable[str]:
    if SentenceSplitter is not None:
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        yield from splitter.split_text(text)
        return

    words = text.split()
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start = max(end - overlap, start + 1)


def pdf_chunks(pdf_path: Path) -> List[Chunk]:
    reader = PdfReader(str(pdf_path))
    chunks: List[Chunk] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue

        topic = infer_topic(text)
        for chunk_index, chunk_text in enumerate(split_text(text), start=1):
            digest = hashlib.sha1(
                f"{pdf_path.name}:{page_index}:{chunk_index}:{chunk_text[:80]}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=f"{pdf_path.stem}-{page_index}-{chunk_index}-{digest}",
                    text=chunk_text,
                    metadata={
                        "file_name": pdf_path.name,
                        "page_number": page_index,
                        "topic": topic,
                        "chunk_index": chunk_index,
                    },
                )
            )

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
    response = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": texts},
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "Ollama embedding request failed. Make sure Ollama is running and the "
            f"'{model}' model is pulled. Response: {response.text}"
        )
    data = response.json()
    return data["embeddings"]


def ingest(reset: bool = False) -> int:
    load_dotenv(ROOT / ".env")
    chroma_path = Path(os.getenv("CHROMA_PATH", str(DEFAULT_CHROMA_PATH)))
    if not chroma_path.is_absolute():
        chroma_path = ROOT / chroma_path
    collection_name = os.getenv("COLLECTION_NAME", "securescholar_papers")

    client = chromadb.PersistentClient(path=str(chroma_path))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PAPERS_DIR}. Add 5-20 papers, then rerun ingestion.")
        return 0

    all_chunks: List[Chunk] = []
    for pdf in pdfs:
        chunks = pdf_chunks(pdf)
        print(f"{pdf.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No extractable text found in PDFs.")
        return 0

    batch_size = 64
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        embeddings = embed_texts([chunk.text for chunk in batch])
        collection.upsert(
            ids=[chunk.id for chunk in batch],
            documents=[chunk.text for chunk in batch],
            embeddings=embeddings,
            metadatas=[chunk.metadata for chunk in batch],
        )
        print(f"Stored chunks {start + 1}-{start + len(batch)} of {len(all_chunks)}")

    print(f"Done. Collection '{collection_name}' now contains {collection.count()} chunks.")
    return len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into SecureScholar.")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the Chroma collection.")
    args = parser.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
