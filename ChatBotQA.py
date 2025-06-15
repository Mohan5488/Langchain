import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import os
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(
    api_key=api_key,
    model_name="llama3-8b-8192",
    temperature=0.7
)
st.set_page_config(page_title='Conversational Q&A Chatbot')
st.header("Hey, Let's have a conversation")

def get_response(user_input):
    st.session_state['flowmessages'].append(HumanMessage(content=user_input))
    answer = llm(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a helpful assistant. Answer the user's questions in a conversational manner."),
    ]

input_text = st.text_input("Ask me anything:", key="input_text")
if input_text:
    response = get_response(input_text)
    st.write("AI:", response) 