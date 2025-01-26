import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import OPENAI_API_KEY, ELASTICSEARCH_URL, ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD
from elasticsearch_utils import connect_to_elasticsearch, fetch_documents
from llm_utils import CustomLLM
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SentenceTransformerRerank

# Cài đặt OpenAI API key
import openai
openai.api_key = OPENAI_API_KEY

# Kết nối Elasticsearch
if "documents" not in st.session_state:
    es_client = connect_to_elasticsearch(
        ELASTICSEARCH_URL,
        ELASTICSEARCH_USERNAME,
        ELASTICSEARCH_PASSWORD
    )
    st.session_state.documents = fetch_documents(es_client, "my_index")

# Tạo VectorStoreIndex và retriever nếu chưa tồn tại
if "index" not in st.session_state:
    embedding_model = HuggingFaceEmbedding()
    st.session_state.index = VectorStoreIndex.from_documents(
        st.session_state.documents,
        embedding=embedding_model
    )

if "retriever" not in st.session_state:
    st.session_state.retriever = st.session_state.index.as_retriever(
        similarity_top_k=5,
        node_postprocessors=[
            SentenceTransformerRerank(
                model="mixedbread-ai/mxbai-rerank-xsmall-v1",
                top_n=5,
                keep_retrieval_score=True
            )
        ]
    )

# Prompt hệ thống
system_prompt = """
{
  "instruction": "You are a multilingual and highly intelligent Q&A system specializing in admissions to the University of Information Technology (UIT). Your task is to provide accurate and contextually relevant answers to user queries based on the provided context and rules.",
  "role": "UIT Admissions Assistant",
  "rules": [
    "1. If the query is in Vietnamese, answer in Vietnamese. If the query is in English, answer in English.",
    "2. Use only the provided context for answers; do not rely on assumptions or prior knowledge.",
    "3. If the query is unclear or insufficient, ask for clarification concisely only when related to UIT admissions.",
    "4. Ensure responses are concise, clear, and avoid technical jargon unless necessary.",
    "5. For queries unrelated to UIT admissions, politely inform the user that you can only assist with UIT admissions and do not inquire further.",
    "6. Handle invalid input (e.g., unreadable characters or formats) by requesting a valid query.",
    "7. Keep responses within a maximum of 800 tokens to ensure clarity and efficiency.",
    "8. Maintain a strict focus on UIT admissions context and exclude unrelated topics.",
    "9. Avoid contradictory statements by carefully cross-checking the context provided before formulating responses."
  ]
}
"""

# Khởi tạo CustomLLM nếu chưa có trong session state
if "custom_llm" not in st.session_state:
    llm = OpenAI(model="gpt-3.5-turbo")
    st.session_state.custom_llm = CustomLLM(
        llm=llm,
        system_prompt=system_prompt,
        retriever=st.session_state.retriever,
        rerank_postprocessor=SentenceTransformerRerank(
            model="mixedbread-ai/mxbai-rerank-xsmall-v1",
            top_n=2,
            keep_retrieval_score=True
        )
    )

# Giao diện Streamlit
st.title("UIT Admissions Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input từ người dùng
if prompt := st.chat_input("Đặt câu hỏi về tuyển sinh tại đây"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state.custom_llm.query(prompt)
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
