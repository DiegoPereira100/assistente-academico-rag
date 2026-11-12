from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "documentos_academicos"

# Configuração inicial para o primeiro experimento.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Segunda configuração para comparação experimental.
EXPERIMENT_CONFIG = {
    "chunk_size": 500,
    "chunk_overlap": 80,
}

# Modelo de linguagem local.
OLLAMA_MODEL = "qwen2.5:3b"