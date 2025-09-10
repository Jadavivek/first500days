import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    JSONLoader,
)

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Category → Folder or File mapping
CATEGORY_FOLDERS = {
    "medical": r"E:\rag_first500\data\medical_faqs.json",  # fixed path
}


def load_single_file(path: str):
    """Load a single file based on its extension."""
    ext = path.lower()
    docs = []
    try:
        if ext.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif ext.endswith((".docx", ".doc")):
            docs.extend(UnstructuredWordDocumentLoader(path).load())
        elif ext.endswith(".xlsx"):
            docs.extend(UnstructuredExcelLoader(path).load())
        elif ext.endswith(".json"):
            docs.extend(JSONLoader(file_path=path, jq_schema=".[]", text_content=False).load())
    except Exception as e:
        print(f"⚠️ Could not load {os.path.basename(path)}: {e}")
    return docs


def load_all_docs_from_path(path: str):
    """Load documents from a folder or single file."""
    if os.path.isfile(path):
        return load_single_file(path)

    docs = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            docs.extend(load_single_file(file_path))
    return docs


def build_faiss_index(category: str, path: str):
    """Build or update a FAISS index for a given category."""
    print(f"📂 Building index for '{category}' from {path}")

    docs = load_all_docs_from_path(path)
    if not docs:
        print(f"⚠️ No documents found in {path}")
        return

    # Split docs into chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    index_path = f"faiss_index_{category}"
    faiss_file = os.path.join(index_path, "index.faiss")

    if os.path.exists(faiss_file):
        print(f"📌 Updating existing FAISS index for '{category}'")
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        print(f"🆕 Creating new FAISS index for '{category}'")
        db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_path, exist_ok=True)
    db.save_local(index_path)
    print(f"✅ Index for '{category}' saved to {index_path}")


if __name__ == "__main__":
    for category, path in CATEGORY_FOLDERS.items():
        if os.path.exists(path):
            build_faiss_index(category, path)
        else:
            print(f"⚠️ Skipping '{category}': path '{path}' not found")
