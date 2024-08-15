import csv
import streamlit as st
import os
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders.csv_loader import CSVLoader

openai_api_key = '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'
os.environ['OPENAI_API_KEY'] = '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'

st.title('Ask anything from documents')

# Sidebar for file uploads
st.sidebar.title("Upload Files")
csv_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
pdf_file = st.sidebar.file_uploader("Upload PDF file", type=["pdf"])
question = st.text_input("Enter your question:")

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

directory_path1 = "C:/Users/sucin/Desktop/project/P5/pdfDir/"
directory_path2 = "C:/Users/sucin/Desktop/project/P5/csvDir/"

if st.sidebar.button("Delete"):
    delete_files_in_directory(directory_path1)
    delete_files_in_directory(directory_path2)

if csv_file is not None:
    with open(os.path.join("csvDir",csv_file.name),"wb") as f: 
        f.write(csv_file.getbuffer())

    path= f"C:/Users/sucin/Desktop/project/P5/csvDir/{csv_file.name}"

    agent = create_csv_agent(OpenAI(temperature=0), csv_file, verbose=True)

    if question is not None and question != "":
        with st.spinner(text="In progress..."):
            st.write(agent.run(question))

elif pdf_file is not None:
    with open(os.path.join("pdfDir", pdf_file.name), "wb") as f:
        f.write(pdf_file.getbuffer())

    directory = 'C:/Users/sucin/Desktop/project/P5/pdfDir/'

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

    # Initialize ChatOpenAI model
    openai_api_key = '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'
    llm_pdf = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, api_key=openai_api_key)

    # Load QA chain and create RetrievalQA chain
    chain = load_qa_chain(llm_pdf, chain_type="stuff", verbose=True)
    retrieval_chain = RetrievalQA.from_chain_type(llm_pdf, chain_type="stuff", retriever=db_pdf.as_retriever())

    if st.button("Submit"):
        full_question = f"Question: {question}"
        try:
            answer = retrieval_chain.run(question)
            st.write(f"Answer: {answer}")

        except Exception as e:
            st.write(f"An error occurred: {e}")