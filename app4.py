import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama   # ✅ Added

# -------------------------------
# Step 1: Page Configuration
# -------------------------------
st.set_page_config(page_title="C++ RAG Chatbot", page_icon="💬")
st.title("💬 C++ RAG Chatbot")
st.write("Ask any question related to C++ introduction")

# -------------------------------
# Step 2: Load Environment Variables
# -------------------------------
load_dotenv()

# -------------------------------
# Step 3: Load & Cache Vector Store
# -------------------------------
@st.cache_resource
def load_vector_store():
    loader = TextLoader("C++_Introduction.txt", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_documents = text_splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(final_documents, embedding)

    return db

db = load_vector_store()

# -------------------------------
# Step 4: Initialize LLM (Only Once)
# -------------------------------
if "llm" not in st.session_state:
    st.session_state.llm = Ollama(model="gemma2:2b")

# -------------------------------
# Step 5: User Query
# -------------------------------
query = st.text_input("Enter your question about C++:")

if query:

    # ---------------------------
    # Step 5A: Check if Question is C++ Related
    # ---------------------------
    classification_prompt = f"""
    You are a classifier.

    Determine if the following question is related to C++ programming.

    Reply with only one word:
    YES
    or
    NO

    Question: {query}
    """

    is_cpp_related = st.session_state.llm.invoke(classification_prompt).strip().upper()

    if is_cpp_related == "NO":
        st.subheader("🤖 AI Answer")
        st.write("This question is outside the scope of this C++ assistant.")
        st.stop()   # 🚀 STOPS execution (no retrieval happens)

    # ---------------------------
    # Step 5B: Retrieve Context (ONLY if C++ question)
    # ---------------------------
    documents = db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in documents])

    prompt = f"""
    You are a helpful C++ assistant.

    Use ONLY the context provided below to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer clearly and concisely:
    """

    with st.spinner("Thinking... 🤔"):
        response = st.session_state.llm.invoke(prompt)

    # Display AI Answer
    st.subheader("🤖 AI Answer")
    st.write(response)

    # Display Retrieved Context
    st.subheader("📒 Retrieved Context")
    for i, doc in enumerate(documents):
        st.markdown(f"**Result {i+1}:**")
        st.write(doc.page_content)