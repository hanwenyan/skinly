import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- Model Configuration ---
MODEL_NAME = "gemini-2.5-pro"



def get_gemini_response(user_query: str, api_key: str, system_prompt: str):
    """
    Initializes the Gemini model and gets a response for the user's query.

    Args:
        user_query (str): The query from the user.
        api_key (str): The Google API key.
        system_prompt (str): The system prompt to use.

    Returns:
        str: The model's response or None if an error occurs.
    """
    if not api_key:
        st.error("Please provide your Google API Key in the sidebar.")
        return None

    os.environ["GOOGLE_API_KEY"] = api_key
    try:
        # We are using the MODEL_NAME
        model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.5)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]
        response = model.invoke(messages)
        return response.content
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="👩‍⚕️ Skinly", page_icon="👩‍⚕️")
st.title("👩‍⚕️ Skinly")
st.caption(f"Your AI-powered skin health assistant.")

# --- Load System Prompt from Secrets ---
try:
    system_prompt = st.secrets["SYSTEM_PROMPT"]
except (KeyError, FileNotFoundError):
    st.warning("SYSTEM_PROMPT not found in secrets.toml. Using a default prompt.")
    system_prompt = (
        "You are a helpful AI assistant. Respond to the user's queries in a "
        "friendly and professional manner."
    )

# --- Sidebar for API Key ---
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Enter your Google API Key:", type="password", key="google_api_key")
    st.markdown("[Get your Google API key](https://aistudio.google.com/app/apikey)")

# --- Chat History Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am Skinly, your AI Dermatology Assistant. How can I help you today? \n\n**Disclaimer:** I am an AI assistant and not a medical professional. Please consult with a qualified healthcare provider for any medical advice.",
        }
    ]

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Response Handling ---
if prompt := st.chat_input("What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_gemini_response(prompt, google_api_key, system_prompt)
            if response:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})