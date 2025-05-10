from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA

import streamlit as st
import os
from dotenv import load_dotenv

# Page configuration
st.set_page_config(page_title="Herbal Remedies Assistant", page_icon="🌿")

# Load environment variables
load_dotenv()
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# Set up langchain API key if available
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key

# Set up embeddings
embeddings = OllamaEmbeddings(model="llama3.2")

# Function to load and split documents
@st.cache_resource
def load_documents():
    try:
        loader = CSVLoader("herbal_remedies_1000plus.csv")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs[:20])  # Limit for performance
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []

# Function to create or load Chroma vector store
@st.cache_resource
def get_vectorstore(_documents):
    if not _documents:
        return None
    
    # Create Chroma vector store
    return Chroma.from_documents(
        documents=_documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

# UI elements
st.title("🌿 Herbal Remedies Assistant")
st.markdown("Ask questions about herbal remedies and get informed answers based on our database.")

# Load documents
with st.spinner("Loading document database..."):
    documents = load_documents()
    if documents:
        st.success(f"✅ Loaded {len(documents)} document chunks")
    else:
        st.error("Failed to load documents. Please check the CSV file.")
        st.stop()

# Create vector store
with st.spinner("Setting up search capabilities..."):
    vectorstore = get_vectorstore(documents)
    if not vectorstore:
        st.error("Failed to create vector store.")
        st.stop()

# Create retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define system template with context
template = """You are an expert in herbal remedies and natural medicine.
Use the following context to answer the user's question. 
If you don't know the answer based on the context, say you don't know.

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Use Ollama as the LLM
llm = Ollama(model="llama3.2")

# Create chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query interface
user_question = st.text_input("What would you like to know about herbal remedies?", 
                             placeholder="Example: What herbs help with anxiety?")

if user_question:
    with st.spinner("Searching for information..."):
        try:
            response = chain.invoke(user_question)
            st.markdown("### Answer")
            st.markdown(response)
            
            # Show retrieved documents for transparency
            with st.expander("View source documents"):
                retrieved_docs = retriever.get_relevant_documents(user_question)
                for i, doc in enumerate(retrieved_docs):
                    st.markdown(f"**Document {i+1}**")
                    st.markdown(doc.page_content)
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error getting response: {e}")

# Sidebar info
with st.sidebar:
    st.markdown("### About")
    st.markdown("This application uses vector search to find relevant information about herbs and natural remedies from a database of over 1000 herbal remedies.")
    st.markdown("Made with Langchain, Chroma DB, and Streamlit.")