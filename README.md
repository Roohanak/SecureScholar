# SecureScholar

SecureScholar is a small local-AI Streamlit MVP for asking source-grounded questions over computer security research papers.

## Setup

```powershell
cd SecureScholar
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install Ollama, then pull one chat model and one embedding model:

```powershell
ollama pull llama3.2
ollama pull embeddinggemma
```

The default `.env` points to local Ollama:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=embeddinggemma
```

## Add Papers

Put 5-20 cybersecurity research PDFs in `papers/`.

## Ingest

```powershell
python ingest.py --reset
```

The ingestion script:

- loads PDFs
- extracts text
- splits text into chunks
- creates embeddings
- stores chunks in ChromaDB
- saves file name, page number, inferred topic, and chunk index

## Run

```powershell
streamlit run app.py
```

No OpenAI or Gemini API key is required.

## Test Questions

- What attack does this paper propose?
- What defense is suggested?
- What are the limitations?
- Which papers discuss malware detection?
- Compare the methods in these papers.
- Generate a short literature review on intrusion detection.

## Prompt Buttons

The app includes templates for:

- Summarize this paper
- Extract threat model
- Extract attack method
- Extract defense/mitigation
- Find limitations
- Compare two papers
- Generate mini literature review
