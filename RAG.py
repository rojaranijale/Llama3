__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from dotenv import load_dotenv
import streamlit as st
from huggingface_hub import login
from huggingface_hub import InferenceApi
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
#from langchain import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import utils as chromautils
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
login(token=token)

loader = UnstructuredExcelLoader("All_ScrapedText_03_06.xlsx")
docs = loader.load()
docs = chromautils.filter_complex_metadata(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

docs = text_splitter.split_documents(docs)

modelPath = "mixedbread-ai/mxbai-embed-large-v1"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs 
)

db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()


prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {question}

Helpful Answer:
"""

PROMPT = PromptTemplate(
 template=prompt_template, input_variables=["context", "question"]
)
inference_api = InferenceApi(repo_id="distilbert-base-uncased-finetuned-sst-2-english")
llm = HuggingFacePipeline(pipeline=inference_api)

st_callback = StreamlitCallbackHandler(st.container())

retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = retrievalQA.invoke(
            {"query": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["result"])
