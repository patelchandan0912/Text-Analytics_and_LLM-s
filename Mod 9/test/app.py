# ... (no changes to the initial part of the code)
import streamlit as st
import os
import pinecone
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv


# Styles
def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


# Pinecone
def initialize_vector_store():
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
    index = pinecone.Index(os.getenv('PINECONE_INDEX_NAME'))
    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Pinecone(index, embed_model, "text")
    return vectorstore


# Session State
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    if "conversation" not in st.session_state:
        llm = ChatOpenAI(temperature=0.2, openai_api_key=os.environ["OPENAI_API_KEY"], model_name="gpt-4")
        st.session_state.conversation = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm), verbose=True)


@dataclass
class Message:
    origin: Literal['human', 'ai']
    message: str


# Submit Callback
def on_click_callback():
    with get_openai_callback() as cb:
        human_prompt = st.session_state.human_prompt
        vectorstore = initialize_vector_store()
        similar_docs = vectorstore.similarity_search(human_prompt, k=25)
        prompt = f"""You are a friendly chatbot. \n\nQuery:\n"{human_prompt}" \n\nContext:" \n"{' '.join([doc.page_content for doc in similar_docs])}" \n"""
        print(prompt)
        llm_response = st.session_state.conversation.run(prompt)
        st.session_state.history.append(Message("human", human_prompt))
        st.session_state.history.append(Message("ai", llm_response))
        st.session_state.token_count += cb.total_tokens


# MAIN PROGRAM

# initializations
load_dotenv()  # load environment variables from .env file
load_css()  # load the css
initialize_session_state()  # initialize the history buffer that is stored in UI  

# create the Streamlit UI
st.title("USF BullBot 🐂")
st.markdown("Welcome to USF BullBot! How can I assist you today?")
chat_placeholder = st.container()  # container for chat history
prompt_placeholder = st.form("chat-form")  # form for user prompt
debug_placeholder = st.empty()  # container for debugging information

def clear_chat():
    st.session_state.history = []

with chat_placeholder:  # display chat history
    for chat in st.session_state.history:
        div = f"""
            <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
                <img class="chat-icon" src = "{'https://content.sportslogos.net/logos/34/837/full/south_florida_bulls_logo_secondary_20036111.png' if chat.origin == 'ai' else 'https://www.seekpng.com/png/detail/131-1316604_built-an-enhanced-lab-template-ppp-prd-045.png'}" width=42 height=42>
                <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
                    &#8203;{chat.message}
                </div>
            </div>
        """
        st.markdown(div, unsafe_allow_html=True)
    if st.button("Clear Chat"):
        clear_chat()

with prompt_placeholder:  # chat input field
    col1, col2 = st.columns((6,1))
    col1.text_input(
        "Chat",
        value="" if not st.session_state.history else "Please enter your question here",
        label_visibility="collapsed",
        key="human_prompt",  # key to reference this field in the callback
    )
    col2.form_submit_button(
        "Submit",
        type="primary",
        on_click=on_click_callback,  # set the callback function for the submit button
    )

debug_placeholder.caption(  # display debugging information
    f"""
    Used {st.session_state.token_count} tokens \n
    Debug Langchain.coversation:
    {st.session_state.conversation.memory.buffer}
    """
)
