from langchain import LLMMathChain, OpenAI, SQLDatabase, GoogleSearchAPIWrapper, LLMChain
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from decouple import config
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.output_parsers import StrOutputParser
import time
from datetime import timedelta
import json
import sys
import pyodbc
import pandas as pd

os.environ["GOOGLE_CSE_ID"] = "Use ur own key"
os.environ["GOOGLE_API_KEY"] = "Use ur own key"

openai_api_key = 'Use ur own key'

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

st.title('Order Tracking')
st.write("Enter your order ID and a question regarding your order, and I will help you track it.")

question = st.text_input("Enter your question:")
          
def is_valid_order_id(order_id):
    return order_id.isdigit() and len(order_id) == 6

if st.button("Submit"):
        full_question = f"Question: {question}"
        try:
            s="0"
            for i in question.split(" "):
                res = ''.join(filter(lambda i: i.isdigit(), i))
                if len(res)==6:
                     s=res        
            if len(s)==6:
                 
                output = agent.invoke(full_question, handle_parsing_errors=True)
                st.write(f"Answer: {output['output']}")
            else:
                st.write("Please provide your order id.")
                 
            
        except Exception as e:
            st.write(f"An error occurred: {e}")
