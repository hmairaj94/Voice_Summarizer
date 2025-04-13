import streamlit as st
import requests
import json
import tempfile
import os
import datetime
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

st.title("🔥 Voice-Activated Summarizer with Sentiment Insight")

HF_API_URL = "https://api-inference.huggingface.co/models/"
HF_API_KEY = os.getenv("HF_API_KEY", "")  

if 'history' not in st.session_state:
    st.session_state.history = []

if not HF_API_KEY:
    st.warning("⚠️ Hugging Face API key not found in environment variables.")
    st.info("Please create a .env file with HF_API_KEY=your_api_key or enter it below:")
    HF_API_KEY = st.text_input("Enter your Hugging Face API Key:", type="password")

WHISPER_MODEL = "openai/whisper-small"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Helper functions for API calls
def query_huggingface_api(model, data, task="inference"):
    """Generic function to call Hugging Face API"""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    API_URL = f"{HF_API_URL}{model}"
    
    if task == "audio":
        with open(data, "rb") as f:
            response = requests.post(API_URL, headers=headers, data=f)
    else:
        response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code != 200:
        st.error(f"API Error: {response.status_code} - {response.text}")
        return None
        
    return response.json()

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper API"""
    result = query_huggingface_api(WHISPER_MODEL, audio_path, task="audio")
    if result and isinstance(result, dict) and "text" in result:
        return result["text"]
    return "Transcription failed. Please try again with clearer audio."

def summarize_text(text):
    """Summarize text using BART API"""
    params = {
        "inputs": text,
        "parameters": {"max_length": 150, "min_length": 30, "do_sample": False}
    }
    result = query_huggingface_api(SUMMARIZER_MODEL, params)
    if result and isinstance(result, list) and len(result) > 0:
        return result[0].get("summary_text", "Summarization failed.")
    return "Summarization failed. Please try again with different text."

def analyze_sentiment(text):
    """Analyze sentiment using DistilBERT API"""
    params = {"inputs": text}
    result = query_huggingface_api(SENTIMENT_MODEL, params)
    
    if result is None:
        return {"label": "NEUTRAL", "score": 0.5}
    
    if isinstance(result, list):
        if len(result) > 0:
            
            if isinstance(result[0], dict) and "label" in result[0] and "score" in result[0]:
                return result[0]
            
            elif isinstance(result[0], list):
                flattened = [item for sublist in result for item in sublist]
                if flattened and isinstance(flattened[0], dict):
                    return flattened[0]
    
    elif isinstance(result, dict) and "label" in result and "score" in result:
        return result
    
    return {"label": "NEUTRAL", "score": 0.5}



# Audio input section - Upload only
st.subheader("1️⃣ Audio Input")
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None and HF_API_KEY:
    st.subheader("2️⃣ Transcription")
    with st.spinner("Processing audio..."):
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            transcript = transcribe_audio(tmp_file_path)
        
            st.text_area("Transcript", transcript, height=150)
            
            if len(transcript.split()) > 5:
                # Summarization
                st.subheader("3️⃣ Summary")
                with st.spinner("Generating summary..."):
                    # Ensure the text isn't too long for the model
                    max_length = 1024
                    if len(transcript.split()) > max_length:
                        st.warning(f"The transcript is quite long. Summarizing the first {max_length} words.")
                        words = transcript.split()[:max_length]
                        transcript_for_summary = " ".join(words)
                    else:
                        transcript_for_summary = transcript
                    
                    # Generate summary if text is long enough
                    if len(transcript_for_summary.split()) > 10:
                        summary = summarize_text(transcript_for_summary)
                    else:
                        summary = "Text too short to summarize."
                    
                    st.text_area("Summary", summary, height=100)
                
                # Sentiment Analysis
                st.subheader("4️⃣ Sentiment Analysis")
                with st.spinner("Analyzing sentiment..."):
                    try:
                        sentiment_result = analyze_sentiment(transcript)
                        
                        sentiment_label = sentiment_result.get("label", "NEUTRAL")
                        sentiment_score = sentiment_result.get("score", 0.5)
                        
                        if sentiment_label == "POSITIVE":
                            st.markdown(f"<h4 style='color:green'>Sentiment: {sentiment_label} ({sentiment_score:.2f})</h4>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h4 style='color:red'>Sentiment: {sentiment_label} ({sentiment_score:.2f})</h4>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error processing sentiment: {str(e)}")
                        sentiment_label = "ERROR"
                        sentiment_score = 0.0
                
                # Save to history
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append({
                    "timestamp": timestamp,
                    "transcript": transcript,
                    "summary": summary,
                    "sentiment": sentiment_label,
                    "sentiment_score": sentiment_score
                })
            else:
                st.warning("The transcript is too short for meaningful analysis. Please provide longer audio.")
                
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            st.info("Try uploading a different file with clearer audio.")
        
        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass

# History section

# History section
st.subheader("📚 Analysis History")
if st.session_state.history:
    # Debug information
    st.write(f"Number of history items: {len(st.session_state.history)}")
    
    # Use radio buttons instead of selectbox
    timestamps = [item['timestamp'] for item in st.session_state.history]
    selected_timestamp = st.radio(
        "Select a previous analysis:",
        timestamps,
        key="history_radio"
    )
    
    # Find the selected item by timestamp
    selected_item = None
    for item in st.session_state.history:
        if item['timestamp'] == selected_timestamp:
            selected_item = item
            break
    
    if selected_item:
        with st.expander("View details", expanded=True):
            st.write(f"**Timestamp:** {selected_item['timestamp']}")
            st.write(f"**Transcript:** {selected_item['transcript']}")
            st.write(f"**Summary:** {selected_item['summary']}")
            
            sentiment = selected_item.get('sentiment', 'N/A')
            score = selected_item.get('sentiment_score', 0.0)
            st.write(f"**Sentiment:** {sentiment} ({score:.2f})")
else:
    st.info("No analysis history yet. Process an audio file to see results here.")