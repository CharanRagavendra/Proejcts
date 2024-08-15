from langchain import LLMMathChain, OpenAI, SQLDatabase, GoogleSearchAPIWrapper, LLMChain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from decouple import config
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
import time
from datetime import timedelta
import json
import sys
import pyodbc
import pandas as pd

openai_api_key = 'Use ur own key'

os.environ["GOOGLE_CSE_ID"] = "Use ur own key"
os.environ["GOOGLE_API_KEY"] = "Use ur own key"

llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, api_key=openai_api_key)

search = GoogleSearchAPIWrapper()

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

db = SQLDatabase.from_uri("sqlite:///orders1.db")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

tools = [
    Tool(
        name="SearchTool",
        func=search.run,
        description="Useful for answering questions about current events. Ask targeted questions."
    ),
    Tool(
        name="MathTool",
        func=llm_math_chain.run,
        description="Useful for answering questions about math."
    ),
    Tool(
        name="Product_Database",
        func=db_chain.run,
        description="Useful for answering questions about products."
    )
]

agent = initialize_agent(tools=tools, llm=llm, agent_type="zero-shot-react-description", verbose=True)

#loading documents
directory = 'C:/Users/sucin/Desktop/project/Doc_ret/doc'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)

#split the documents atter we load them
def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
#print(len(docs))

#embedding text using langchain
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db1 = Chroma.from_documents(docs, embeddings,persist_directory="paper")
db1.persist()
vector=Chroma(persist_directory="paper",embedding_function=embeddings)

llm1 = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0, api_key=openai_api_key)
template=  """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
Given question and context, return yes if question and context matches, return no if it does not match and also return the answer
Give me two variables where first variable named bool conatins yes or no 
and the other variable named answer contains the answer
context: {context}
Question: {question}
"""
answer_prompt = PromptTemplate(input_variables=["context","question"],template=template)

# chain = load_qa_chain(llm1, chain_type="stuff",verbose=True)

retrieval_chain = RetrievalQA.from_chain_type(llm=llm1, chain_type="stuff", retriever=vector.as_retriever(),chain_type_kwargs={"prompt":answer_prompt})
print(retrieval_chain.run("What is the details of this order 100001"))

st.title('Ask anything from document or databse')

question = st.text_input("Enter your question:")
          
if st.button("Submit"):
    full_question = f"Question: {question}"
    try:
        # Searches both database and document
        document_answer = retrieval_chain.run(question)
        database_answer = agent.invoke(full_question, handle_parsing_errors=True)
        
        if document_answer[0]=="Y":
            st.write(f"Document Answer: {document_answer}")
        elif database_answer:
            st.write(f"Database Answer: {database_answer}")
        else:
            st.write(f"Cant find any answer")
            
    except Exception as e:
        st.write(f"An error occurred: {e}")


