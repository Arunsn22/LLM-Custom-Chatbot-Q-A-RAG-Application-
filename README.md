# LLM-Custom-Chatbot-Q-A-RAG-Application-

# 🧠 Learning Log — RAG Chatbot with LangChain + ChromaDB (Offline)

## 1. Objective
To build a **fully offline Retrieval-Augmented Generation (RAG) chatbot** using:
- **LangChain** for orchestration  
- **ChromaDB** for local vector storage  
- **LlamaCpp (via llama-cpp-python)** to run a local **Mistral-7B GGUF** model  
- **Streamlit** for the user interface  

---

## 2. Topics Learned

### ✅ a. Retrieval-Augmented Generation (RAG)
- **Concept:** Combine local document retrieval with an LLM to answer questions based on custom data.  
- **Components:**
  - **Document Loader** → loads PDFs or text into LangChain.
  - **Text Splitter** → breaks long text into smaller chunks.
  - **Embeddings Model** → converts text chunks into numerical vectors.
  - **Vector Store (ChromaDB)** → stores and retrieves embeddings efficiently.
  - **LLM (Mistral via LlamaCpp)** → generates final answers based on retrieved context.

---

### ✅ b. LangChain Components Used
- `PyPDFLoader` — for reading PDFs.  
- `RecursiveCharacterTextSplitter` — for chunking documents.  
- `HuggingFaceEmbeddings` — for generating vector embeddings (using `all-MiniLM-L6-v2`).  
- `Chroma` — for storing and querying document embeddings.  
- `LlamaCpp` — to load and run a local `.gguf` model using `llama-cpp-python`.  
- `PromptTemplate` — to control how the model answers.  
- `RetrievalQA` — integrates retrieval and LLM response generation.  

---

### ✅ c. Streamlit UI Design
- Used `st.chat_message()` to mimic real chat interactions.  
- Used `st.chat_input()` to allow continuous question asking.  
- Maintained `st.session_state` for chat history.  
- Allowed dynamic PDF uploads and automatic processing into ChromaDB.  

---

## 3. Key Issues Encountered & Solutions

| Issue | Cause | Solution |
|--------|--------|-----------|
| `ModuleNotFoundError: langchain_community` | Newer LangChain versions split modules | Installed `langchain-community` with `pip install -U langchain-community` |
| Pydantic Validation Error: `openai_api_key` missing | OpenAI model placeholder in code | Replaced OpenAI model with local `LlamaCpp` model |
| Echoed full prompt instead of natural answer | Prompt formatting confusion | Cleaned up `PromptTemplate` to separate context, question, and short instruction clearly |
| `application/pdf files are not allowed` | Streamlit file uploader type restrictions | Explicitly allowed `type=["pdf"]` in `st.sidebar.file_uploader` |
| No answers or incomplete responses | Prompt too long or unstructured | Simplified the instruction to one-sentence, natural answers |
| `Model path does not exist` error | Incorrect path to GGUF model | Provided correct absolute path and ensured file was in project directory |
| `LlamaCppEmbeddings` load failure | Mistral model cannot be used as embedding model | Switched to `HuggingFaceEmbeddings(all-MiniLM-L6-v2)` for embeddings |
| Model responses too formal or repetitive | Overly strict or verbose prompt | Rewrote prompt to encourage a natural, student-friendly answer style |

---

## 4. Tools and Packages Used
- **LangChain 0.3.27**
- **LangChain-Community**
- **ChromaDB**
- **llama-cpp-python**
- **sentence-transformers (all-MiniLM-L6-v2)**
- **Streamlit**
- **PyPDFLoader**

---

## 5. Final Working Setup
**Architecture:**
```
Streamlit UI  →  LangChain RAG Chain
                 ↳ PDF Loader + Text Splitter
                 ↳ HuggingFace Embeddings (all-MiniLM-L6-v2)
                 ↳ ChromaDB for storage
                 ↳ LlamaCpp (Mistral 7B GGUF) for answer generation
```

**Result:**
- Fully offline chatbot.  
- Natural, one-sentence answers.  
- Supports continuous Q&A chat flow.  
- Works with local PDF documents.

---

## 6. Key Takeaways
- LangChain has modular design — understanding each layer (Loader → Splitter → Embeddings → Retriever → LLM) is essential.  
- `llama-cpp-python` enables running powerful LLMs like Mistral on CPU without internet.  
- Clear prompt design drastically improves output quality.  
- Streamlit is perfect for rapid prototyping and visualization of conversational apps.  
- Embeddings and generation models should be **separate** — one for semantic search, one for language generation.  

