import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from huggingface_hub import InferenceClient
from openai import OpenAI
import os
import pandas as pd

st.set_page_config(layout="wide")

# API Keys - read from environment variables (see .env.example)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # optional for most models

# Function to load CSV file
def load_csv_file(file_path="reviews2.csv"):
    """Load CSV file and convert to text for RAG"""
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to text format
        text_chunks = []
        
        # Check if 'review' column exists (based on your CSV)
        if 'review' in df.columns:
            # Process the review column specifically
            for idx, row in df.iterrows():
                review_text = str(row['review']).strip()
                if review_text and review_text != 'nan':
                    text_chunks.append(review_text)
        else:
            # Fallback for other CSV structures
            for idx, row in df.iterrows():
                row_text = ""
                for col in df.columns:
                    if pd.notna(row[col]):
                        row_text += f"{col}: {row[col]}. "
                if row_text.strip():
                    text_chunks.append(row_text.strip())
        
        # Return as list for sentence processing
        return text_chunks
    except FileNotFoundError:
        st.error(f"CSV file '{file_path}' not found. Please ensure the file exists.")
        return ["No data available. Please check your CSV file."]
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return ["Error loading data."]

# Load initial data from CSV
my_initial_sentences = load_csv_file("reviews2.csv")
my_initial_rag_text = "\n\n".join(my_initial_sentences)

# Check if the LLM model is not already in the session state
if "my_llm_model" not in st.session_state:
    # Set the default LLM model
    st.session_state["my_llm_model"] = "mistralai/Mistral-7B-Instruct-v0.3"

# Check if the SPACE_ID environment variable is not already in the session state
if "my_space" not in st.session_state:
    st.session_state["my_space"] = os.environ.get("SPACE_ID") 

# Function to update the LLM model client
def update_llm_model():
    model = st.session_state["my_llm_model"]
    
    if model.startswith("gemini-"):
        # Initialize the client for gemini models
        if GOOGLE_API_KEY:
            st.session_state["client"] = OpenAI(
                api_key=GOOGLE_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            st.warning("Please add your Google API key at the top of the code")
    elif model.startswith("gpt-"):
        # Initialize the client for openai models
        if OPENAI_API_KEY:
            st.session_state["client"] = OpenAI(api_key=OPENAI_API_KEY)
        else:
            st.warning("Please add your OpenAI API key at the top of the code")
    else:
        # HuggingFace models
        if st.session_state["my_space"]:
            st.session_state["client"] = InferenceClient(model)
        else:
            if HF_TOKEN:
                st.session_state["client"] = InferenceClient(model, token=HF_TOKEN)
            else:
                # Most HuggingFace models work without token
                st.session_state["client"] = InferenceClient(model)
        
# Check if the client is not already in the session state
if "client" not in st.session_state:
    update_llm_model()

# Check if the embeddings model is not already in the session state
if "embeddings_model" not in st.session_state:
    st.session_state["embeddings_model"] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

MAXIMUM_TOKENS = 512

my_system_instructions = "You are a helpful assistant analyzing product reviews. Be brief and concise. Provide answers based on the review data."

first_message = "Hello! I can help you analyze the product reviews. What would you like to know?"

# Check if the chat messages are not already in the session state
if "my_chat_messages" not in st.session_state:
    st.session_state["my_chat_messages"] = []
    st.session_state["my_chat_messages"].append({"role": "system", "content": my_system_instructions})

def delete_chat_messages():
    for key in st.session_state.keys():
        if key != "my_rag_text":
            del st.session_state[key]

augmented_prompt = ""

# Create two columns with a 1:2 ratio
column_1, column_2 = st.columns([1, 2])

# In the first column
with column_1:
    # Display a disclaimer
    st.expander("Disclaimer", expanded=False).markdown("""Large Language Models may provide inaccurate answers. Please verify the responses.""")
    
    # Add a selectbox for model selection
    model_options = ["mistralai/Mistral-7B-Instruct-v0.3",
                     "Qwen/Qwen2.5-72B-Instruct", 
                     "HuggingFaceH4/zephyr-7b-beta"]
    
    # Add API-dependent models
    if GOOGLE_API_KEY:
        model_options.extend(["gemini-1.5-flash", "gemini-1.5-pro"])
    if OPENAI_API_KEY:
        model_options.extend(["gpt-3.5-turbo", "gpt-4"])
    
    st.selectbox("Select the model to use:",
                model_options, 
                key="my_llm_model", 
                on_change=update_llm_model)
    
    # CSV file upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        sentences = []
        for idx, row in df.iterrows():
            row_text = ""
            for col in df.columns:
                if pd.notna(row[col]):
                    row_text += f"{col}: {row[col]}. "
            if row_text.strip():
                sentences.append(row_text.strip())
        st.session_state["my_sentences"] = sentences
        st.session_state["my_rag_text"] = "\n\n".join(sentences)
        delete_chat_messages()
    
    # Button to reload default CSV
    if st.button("Reload reviews2.csv"):
        sentences = load_csv_file("reviews2.csv")
        st.session_state["my_sentences"] = sentences
        st.session_state["my_rag_text"] = "\n\n".join(sentences)
        delete_chat_messages()
    
    # Add a text area for RAG text input
    st.text_area(
        label="Review Data (from CSV):", 
        value=st.session_state.get("my_rag_text", my_initial_rag_text), 
        height=400, 
        key="my_rag_text", 
        on_change=delete_chat_messages
    )

# Check if the sentences are not already in the session state
if "my_sentences" not in st.session_state:
    # Use the pre-loaded sentences from CSV
    st.session_state["my_sentences"] = my_initial_sentences

# Check if the embeddings are not already in the session state
if "my_embeddings" not in st.session_state:
    with st.spinner("Creating embeddings for review data..."):
        st.session_state["my_embeddings"] = st.session_state["embeddings_model"].encode(st.session_state["my_sentences"])
    
with column_2:
    # Create a container for the messages
    messages_container = st.container(height=500)
    
    # Display the first message from the assistant
    messages_container.chat_message("ai", avatar="🤖").markdown(first_message)
    
    # Iterate through the chat messages
    for message in st.session_state["my_chat_messages"]:
        if message["role"] == "user":
            messages_container.chat_message(message["role"], avatar="👤").markdown(message["content"])
        elif message["role"] == "assistant":
            messages_container.chat_message(message["role"], avatar="🤖").markdown(message["content"])

    # Check if there is a new prompt from the user
    if prompt := st.chat_input("Ask about the reviews..."):
        
        # Encode the user's prompt to get its embedding
        my_question_embedding = st.session_state["embeddings_model"].encode([prompt])
        
        # Calculate the cosine similarity
        similarity_to_question = cosine_similarity(my_question_embedding, st.session_state.my_embeddings).flatten()
        
        # Number of sentences to keep based on similarity
        nof_keep_sentences = min(10, len(st.session_state["my_sentences"]))
        
        # Get the indices of the top similar sentences
        sorted_indices = similarity_to_question.argsort()[::-1][:nof_keep_sentences]
        
        # Retrieve the top similar sentences
        sorted_sentences = [st.session_state["my_sentences"][i] for i in sorted_indices]
        
        # Show similarity scores
        max_similarity = similarity_to_question[sorted_indices[0]]
        st.sidebar.write(f"Max similarity: {max_similarity:.3f}")
        
        # Construct the augmented prompt
        augmented_prompt = "Based on these product reviews:\n\n"
        for i, sentence in enumerate(sorted_sentences):
            augmented_prompt += f"{i+1}. {sentence}\n\n"
        augmented_prompt += "\n" + 20*"-" + "\n\n"
        augmented_prompt += f"Question: {prompt}\n\n"
        augmented_prompt += "Please answer based on the review information above."

        # Display the user's prompt
        messages_container.chat_message("user", avatar="👤").markdown(prompt) 
        
        # Append the augmented prompt to chat messages
        st.session_state["my_chat_messages"].append({"role": "user", "content": augmented_prompt})
        
        # Create response
        with messages_container.chat_message("ai", avatar="🤖"):
            response_placeholder = st.empty()
            response = ""
            
            try:
                if 'client' in st.session_state:
                    # Stream the response
                    for chunk in st.session_state["client"].chat.completions.create(
                        messages=st.session_state["my_chat_messages"], 
                        stream=True, 
                        max_tokens=MAXIMUM_TOKENS
                    ):
                        if chunk.choices[0].delta.content:
                            response += chunk.choices[0].delta.content
                            response_placeholder.markdown(response)
                else:
                    response = "Please configure API keys in environment variables."
                    response_placeholder.markdown(response)
            except Exception as e:
                response = f"Error: {str(e)}"
                response_placeholder.markdown(response)

        # Remove the augmented prompt
        st.session_state["my_chat_messages"].pop()
        
        # Append the user's original prompt
        st.session_state["my_chat_messages"].append({"role": "user", "content": prompt})
        
        # Append the assistant's response
        st.session_state["my_chat_messages"].append({"role": "assistant", "content": response})

    # Display debug information
    st.write("Chat History:")
    st.json(st.session_state["my_chat_messages"], expanded=False)
    
    if augmented_prompt:
        st.write("Last Augmented Prompt:")
        st.text_area("Debug", augmented_prompt, height=200, disabled=True)