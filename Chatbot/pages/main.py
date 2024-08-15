import csv
import streamlit as st
import pandas as pd
import os
import pandas as pd
from langchain.llms import CTransformers
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

st.markdown("<h1 style='text-align: center; color: Black;'>Ask anything from document</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar for file uploads
st.sidebar.title("Upload Files")
pdf_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])
question = st.chat_input("Say something")

def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")

directory_path1 = "C:/Users/sucin/Desktop/project/stlit_login/pages/pdf/"

if st.sidebar.button("Delete"):
    delete_files_in_directory(directory_path1)

        
if pdf_file is not None:
    with open(os.path.join("C:/Users/sucin/Desktop/project/stlit_login/pages/pdf", pdf_file.name), "wb") as f:
        f.write(pdf_file.getbuffer())

    directory = 'C:/Users/sucin/Desktop/project/stlit_login/pages/pdf/'

    def load_docs(directory):
        loader = DirectoryLoader(directory)
        documents = loader.load()
        return documents

    documents = load_docs(directory)

    # Split the documents after loading them
    def split_docs(documents, chunk_size=1000, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        return docs
    
    docs = split_docs(documents)

    # Initialize SentenceTransformer embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Convert documents into embeddings and store in Chroma
    db_pdf = Chroma.from_documents(docs, embeddings)

    # Initialize mistrall model
    llm_pdf = CTransformers(model='C:\\Users\\sucin\\Desktop\\project\\llm\\mistral', 
                    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",   
                    model_type='mistral',
                    temperature=0.8,
                    gpu_layers=0,
                    max_new_tokens = 6000,context_length = 6000)

    # Load QA chain and create RetrievalQA chain
    chain = load_qa_chain(llm_pdf, chain_type="stuff", verbose=True)
    retrieval_chain = RetrievalQA.from_chain_type(llm_pdf, chain_type="stuff", retriever=db_pdf.as_retriever())

    if question:
        st.chat_message("user").markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        full_question = f"Question: {question}"
        try:
            answer = retrieval_chain.run(question)
            #st.write(f"Answer: {answer}")
            with st.chat_message("assistant"):
                st.markdown(answer)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.write(f"An error occurred: {e}")

