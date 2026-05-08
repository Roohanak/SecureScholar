from __future__ import annotations

import os
from pathlib import Path
from typing import List

import chromadb
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).parent
PAPERS_DIR = ROOT / "papers"
DEFAULT_CHROMA_PATH = ROOT / "chroma_db"

PROMPTS = {
    "Summarize this paper": (
        "Summarize the selected paper in a professional research style. Include the research problem, "
        "method, findings, security relevance, and limitations. Cite pages inline."
    ),
    "Extract threat model": (
        "Extract the threat model in a structured format: assets, attacker goals, attacker capabilities, "
        "assumptions, trust boundaries, and security implications. Cite pages inline."
    ),
    "Extract attack method": (
        "Describe the attack method clearly and professionally. Include prerequisites, attack steps, "
        "target system, impact, and evidence from the paper. Cite pages inline."
    ),
    "Extract defense/mitigation": (
        "Extract the proposed defenses or mitigations. Explain what each mitigation addresses, how it works, "
        "and any limitations or deployment concerns. Cite pages inline."
    ),
    "Find limitations": (
        "Identify the paper's limitations, evaluation gaps, assumptions, and threats to validity. "
        "Separate directly stated limitations from limitations inferred from the evidence. Cite pages inline."
    ),
    "Compare two papers": (
        "Compare the relevant papers using a professional research format: problem, method, data/evaluation, "
        "main findings, strengths, weaknesses, and practical implications. Cite pages inline."
    ),
    "Generate mini literature review": (
        "Write a concise mini literature review. Synthesize themes across the retrieved papers, compare methods, "
        "identify research gaps, and end with future research directions. Cite pages inline."
    ),
}


def load_config() -> None:
    load_dotenv(ROOT / ".env")


def get_collection():
    chroma_path = Path(os.getenv("CHROMA_PATH", str(DEFAULT_CHROMA_PATH)))
    if not chroma_path.is_absolute():
        chroma_path = ROOT / chroma_path
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection_name = os.getenv("COLLECTION_NAME", "securescholar_papers")
    return client.get_or_create_collection(name=collection_name)


def embed_query(query: str) -> List[float]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma")
    response = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": query},
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "Ollama embedding request failed. Make sure Ollama is running and the "
            f"'{model}' model is pulled. Response: {response.text}"
        )
    return response.json()["embeddings"][0]


def retrieve(query: str, selected_papers: List[str], top_k: int):
    collection = get_collection()
    if collection.count() == 0:
        return None

    query_args = {
        "query_embeddings": [embed_query(query)],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if selected_papers:
        query_args["where"] = {"file_name": {"$in": selected_papers}}

    return collection.query(**query_args)


def build_context(results) -> str:
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    context_blocks = []

    for index, (doc, meta) in enumerate(zip(docs, metadatas), start=1):
        title = Path(meta.get("file_name", "unknown")).stem.replace("_", " ")
        source = (
            f"{index}. {title}, page {meta.get('page_number', '?')}, "
            f"topic: {meta.get('topic', 'unknown')}"
        )
        context_blocks.append(f"[{source}]\n{doc}")

    return "\n\n".join(context_blocks)


def answer_question(question: str, context: str) -> str:
    system_prompt = (
        "You are SecureScholar, a cybersecurity research assistant. "
        "Write in a polished, professional research-assistant style. "
        "Answer only from the provided context. Use clear headings when useful. "
        "Cite evidence inline using the paper title and page number, for example "
        "(6Sense Internet Wide IPv6 Scanning, p. 18). "
        "Do not write citations as generic Source 1 or Source 2 unless the title is unavailable. "
        "Do not mention retrieval distances. "
        "If the context is weak or mismatched, say that the indexed papers do not provide enough evidence."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}"

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(
            "Ollama generation request failed. Make sure Ollama is running and the "
            f"'{model}' model is pulled. Response: {response.text}"
        )
    return response.json().get("response", "No answer returned.")


def local_paper_names() -> List[str]:
    return sorted(path.name for path in PAPERS_DIR.glob("*.pdf"))


def indexed_paper_names() -> List[str]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    data = collection.get(include=["metadatas"], limit=10000)
    return sorted({meta.get("file_name") for meta in data.get("metadatas", []) if meta.get("file_name")})


def main() -> None:
    load_config()
    st.set_page_config(page_title="SecureScholar", page_icon="SS", layout="wide")
    st.title("SecureScholar")
    st.caption("Cybersecurity paper search, synthesis, and source-grounded Q&A.")

    indexed_sources = indexed_paper_names()
    local_sources = local_paper_names()

    with st.sidebar:
        st.header("Papers")
        if local_sources:
            st.write(f"{len(local_sources)} PDFs in papers/")
            selected_papers = st.multiselect("Limit search to", indexed_sources or local_sources)
        else:
            selected_papers = []
            st.info("Add 5-20 cybersecurity PDFs to papers/, then run python ingest.py.")

        st.header("Examples")
        examples = [
            "What attack does this paper propose?",
            "What defense is suggested?",
            "What are the limitations?",
            "Which papers discuss malware detection?",
            "Compare the methods in these papers.",
            "Generate a short literature review on intrusion detection.",
        ]
        chosen_example = st.selectbox("Try a question", [""] + examples)
        top_k = st.slider("Sources to retrieve", min_value=3, max_value=10, value=int(os.getenv("TOP_K", "5")))

    template = st.radio("Prompt template", ["Custom"] + list(PROMPTS.keys()), horizontal=True)
    default_question = PROMPTS.get(template, chosen_example)
    question = st.text_area(
        "Question",
        value=default_question,
        height=120,
        placeholder="Ask about attacks, defenses, limitations, datasets, or compare papers...",
    )

    if st.button("Ask SecureScholar", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
            return

        with st.spinner("Retrieving sources and generating answer..."):
            try:
                results = retrieve(question, selected_papers, top_k)
                if not results:
                    st.warning("No indexed chunks found. Add PDFs to papers/ and run python ingest.py.")
                    return

                context = build_context(results)
                answer = answer_question(question, context)
            except Exception as exc:
                st.error(str(exc))
                return

        st.subheader("Answer")
        st.write(answer)

        st.subheader("References Used")
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        seen_references = []
        for meta in metadatas:
            title = Path(meta.get("file_name", "Unknown paper")).stem.replace("_", " ")
            page = meta.get("page_number", "?")
            reference = f"- {title}, p. {page}"
            if reference not in seen_references:
                seen_references.append(reference)
        st.markdown("\n".join(seen_references))

        with st.expander("Retrieved context"):
            for index, (doc, meta, distance) in enumerate(zip(docs, metadatas, distances), start=1):
                title = Path(meta.get("file_name", "Unknown paper")).stem.replace("_", " ")
                label = (
                    f"{index}. {title}, p. {meta.get('page_number')} | "
                    f"{meta.get('topic')} | relevance distance {distance:.3f}"
                )
                st.markdown(
                    f"**{label}**\n\n"
                    f"{doc}"
                    "\n\n---"
                )

    with st.expander("Setup checklist"):
        st.markdown(
            """
            1. Add 5-20 cybersecurity PDFs to `papers/`.
            2. Install Ollama and run `ollama pull llama3.2` plus `ollama pull embeddinggemma`.
            3. Run `python ingest.py --reset`.
            4. Start the app with `streamlit run app.py`.
            """
        )


if __name__ == "__main__":
    main()
