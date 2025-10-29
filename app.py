import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📘 RAG Chatbot with LangChain + ChromaDB (Offline)")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

# Path to your local Mistral GGUF model
MODEL_PATH = r"C:\Users\seq_arun\Downloads\LLM Project\RAG chatbot with LangChain + ChromaDB\models\mistral-7b-instruct-v0.1.Q2_K.gguf"  # <-- update if in another folder

# ---------------------
# Build RAG system
# ---------------------
def build_qa(pdf_file):
    # Load PDF
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Embeddings (CPU-friendly)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

    retriever = vectordb.as_retriever()

    # Clean natural prompt
    qa_prompt = PromptTemplate(
        template=(
            "You are a helpful assistant.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer in one clear, natural sentence. "
            "If the answer is not in the context, reply: I don’t know from the provided document."
        ),
        input_variables=["context", "question"],
    )

    # LLM (Mistral)
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=8,   # adjust for your CPU
        temperature=0
    )

    # Retrieval QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=False,
    )
    return qa

# ---------------------
# Chat Interface
# ---------------------
if uploaded_file:
    with st.spinner("Processing document..."):
        pdf_path = os.path.join("./", uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        qa = build_qa(pdf_path)

    # Maintain chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input box fixed at bottom
    user_query = st.chat_input("Ask a question about your document...")

    if user_query:
        # Get answer
        response = qa.run(user_query)
        st.session_state.chat_history.append((user_query, response))

    # Show conversation
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
else:
    st.info("👆 Please upload a PDF to get started.")
