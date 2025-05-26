import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import pymupdf as fitz
import os


groq_api_key = "your_groq_api_key"
PERSIST_DIR = "chroma_db"
PDF_FILENAME = "resume.pdf"

# ========= PAGE SETUP =========
st.set_page_config(page_title="📄 Resume Chatbot", layout="centered")
st.title("📄 Chat with My Resume")
st.markdown("Ask questions about my experience, skills, projects, and more!")

# ========= SESSION STATE INIT =========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ========= PDF PROCESSING =========
@st.cache_resource
def process_resume(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([full_text])
    
    # Create vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    return vectorstore

# ========= AGENT SETUP =========
def get_rag_response(query, vectorstore):
    results = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in results])
    
    agent = Agent(
        name="Resume RAG",
        model=Groq(id="llama3-8b-8192", api_key=groq_api_key),
        instructions=[
            f"Use the context to answer questions about the user:",
            f"Context:\n{context}",
            "Only answer from context. If unsure, say 'Not found in resume.'",
            "Be accurate and concise."
        ],
        show_tool_calls=False,
        markdown=True
    )
    return agent.run(query).content

# ========= SIDEBAR =========
with st.sidebar:
    st.header("📁 Upload Resume")
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    if uploaded_file:
        with open(PDF_FILENAME, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.vectorstore = process_resume(PDF_FILENAME)
        st.success("✅ Resume loaded successfully!")

# ========= CHAT INTERFACE =========
if not st.session_state.vectorstore:
    st.info("Upload your resume to start chatting.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask me anything from my resume..."):
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        response = get_rag_response(query, st.session_state.vectorstore)

    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
