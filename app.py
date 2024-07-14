
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os
import json
from langchain.embeddings import HuggingFaceEmbeddings  # or OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document  # Import Document class
from langchain.chains.question_answering import load_qa_chain
import streamlit as st


# Load formatted data
formatted_data = pd.read_json('formatted_data.json')



# EMBEDDING THE FILE

def load_chunk_persist_json() -> Chroma:
    json_file_path = "formatted_data.json"
    
    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create documents using the Document class
    documents = [Document(page_content=entry['input'], metadata={'response': entry['response']}) for entry in data]

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=10)
    chunked_documents = []
    
    for doc in documents:
        input_chunks = text_splitter.split_text(doc.page_content)
        for chunk in input_chunks:
            chunked_documents.append(Document(page_content=chunk, metadata=doc.metadata))

    # Initialize ChromaDB client
    client = chromadb.Client()

    # Create or get collection
   #collection = client.get_or_create_collection("bA0-b")


    # Set up embeddings model (adjust as needed)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Add chunked documents to the Chroma vector store
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embedding_model,
        persist_directory="db"
    )

    # Persist the vector store
    vectordb.persist()

    return vectordb

# Usage
vectordb = load_chunk_persist_json()

#EMBEDDING COMPLETE

#DEFINING CHAT MODEL
chat_model = ChatOllama(model= 'llama3:latest')

#Creating Prompt

template = '''You are an AI customer service assistant for an e-commerce firm XYZ. Answer the question based on the following context

{context}

Question: {question}'''

prompt = ChatPromptTemplate.from_template(template)


# CREATING CHAIN 
def create_agent_chain():
    llm = chat_model
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

#GENERATING RESPONSE
def get_llm_response(query):
    #vectordb = load_chunk_persist_pdf()
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


#SETTING UP STREAMLIT

st.set_page_config(page_title="Assistant Chatbot", page_icon=":robot:")
st.header("Customer Assistant Chatbot")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    st.write(get_llm_response(form_input))
