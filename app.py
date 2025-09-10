# app.py

import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter

# -------------------------------------------------
# Async fix for Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# -------------------------------------------------
load_dotenv()
GOOGLE_API_KEY="AIzaSyCvt3XMP-6BHeFBpjOE_yISNuaMKuDkArw"
# Embeddings + LLM
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

# Paths
JSON_FILE = "data/medical_faqs.json"
INDEX_PATH = "faiss_index_medical"

# -------------------------------------------------
def build_faiss_index():
    """Load JSON, split docs, build/save FAISS index."""
    if not os.path.exists(JSON_FILE):
        st.error(f"❌ Dataset not found: {JSON_FILE}")
        return None

    st.info("📥 Loading dataset...")
    loader = JSONLoader(file_path=JSON_FILE, jq_schema=".[]", text_content=False)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    st.info("⚙️ Building FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    db.save_local(INDEX_PATH)
    return db

def load_or_build_index():
    """Load FAISS if available, otherwise build it."""
    faiss_file = os.path.join(INDEX_PATH, "index.faiss")
    if os.path.exists(faiss_file):
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return build_faiss_index()

# -------------------------------------------------
prompt_template = """
You are a helpful medical assistant. 
Use the retrieved FAQs to answer the question as clearly as possible.
If multiple relevant answers exist, return them as bullet points.

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def query_medical_index(query):
    db = load_or_build_index()
    if db is None:
        return "❌ Could not load index."

    retriever = db.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain.run(query)

# -------------------------------------------------
# ---- Streamlit UI ----
st.set_page_config(page_title="Medical FAQ Chatbot", page_icon="🩺", layout="centered")
st.title("🩺 Medical FAQ Chatbot")

query = st.text_input("Ask a medical question:")

if query:
    with st.spinner("🔎 Searching FAQs..."):
        answer = query_medical_index(query)
        st.write("**Answer:**", answer)
