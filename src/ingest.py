from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PDF_DIR,
)


def load_pdfs():
    documents = []
    pdf_files = sorted(Path(PDF_DIR).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em {PDF_DIR}. Adicione documentos à pasta data/pdfs."
        )

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        # Mantém o nome do arquivo como metadado para exibir a fonte depois.
        for page in pages:
            page.metadata["source_file"] = pdf_path.name

        documents.extend(pages)

    return documents


def main():
    documents = load_pdfs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Recria a coleção a cada ingestão para evitar duplicar os mesmos chunks.
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    existing = vector_store.get()
    if existing and existing.get("ids"):
        vector_store.delete(ids=existing["ids"])

    vector_store.add_documents(chunks)

    print(f"PDFs lidos: {len(list(Path(PDF_DIR).glob('*.pdf')))}")
    print(f"Páginas/documentos carregados: {len(documents)}")
    print(f"Chunks armazenados: {len(chunks)}")
    print(f"Banco vetorial: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
