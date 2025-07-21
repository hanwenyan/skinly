import os
import base64
from typing import Optional
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- Model Configuration ---
MODEL_NAME = "gemini-2.5-pro"


def get_gemini_response(
    user_query: str, api_key: str, system_prompt: str, image: Optional[bytes] = None
):
    """
    Initializes the Gemini model and gets a response for the user's query.

    Args:
        user_query (str): The query from the user.
        api_key (str): The Google API key.
        system_prompt (str): The system prompt to use.
        image (Optional[bytes]): The image data to send to the model.

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

        content = [{"type": "text", "text": user_query}]
        if image:
            image_base64 = base64.b64encode(image).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
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
    st.header("Upload image")
    uploaded_file = st.file_uploader(
        "Add an image of your skin concern (optional)", type=["png", "jpg", "jpeg"]
)

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
        if "image" in message:
            st.image(message["image"], width=200)
        if "content" in message:
            st.markdown(message["content"])

# --- User Input and Response Handling ---
if prompt := st.chat_input("What is your question about the image or skin concern?"):
    image_bytes = uploaded_file.getvalue() if uploaded_file else None

    user_message = {"role": "user", "content": prompt}
    if image_bytes:
        user_message["image"] = image_bytes
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=200)
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_gemini_response(
                prompt, google_api_key, system_prompt, image=image_bytes
            )
            if response:
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )