import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import re
import os
import logging
from openai import OpenAI

# Configure logging to ignore ScriptRunContext warnings
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

# Initialize OpenAI client
api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY", "")
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key) if api_key else None

# ... [Keep AVINASH_CONTEXT unchanged] ...

def sanitize_text(text):
    """Clean text for speech synthesis"""
    text = re.sub(r'[^.!?]*$', '', text)
    return re.sub(r'([$`"\\])', r'\\\1', text)

def generate_response(prompt):
    """Generate response with error handling"""
    if not client:
        return "API configuration error. Please check your secrets."
    
    try:
        response = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[{
                "role": "system",
                "content": f"""Respond as Avinash using: {AVINASH_CONTEXT}
                Guidelines:
                1. 2-3 sentences max
                2. Professional tone
                3. Reference real experience"""
            },{
                "role": "user", 
                "content": prompt
            }],
            temperature=0.65,
            max_tokens=200
        )
        return sanitize_text(response.choices[0].message.content.strip())
    except Exception as e:
        return f"Error processing request: {str(e)}"

def text_to_speech(text):
    """Convert text to speech with error handling"""
    try:
        tts = gTTS(text=text[:500], lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def audio_frame_handler(frame: av.AudioFrame):
    """Process audio input with proper state management"""
    if st.session_state.get("processing_audio"):
        return frame
    
    try:
        st.session_state.processing_audio = True
        recognizer = sr.Recognizer()
        audio_data = frame.to_ndarray().tobytes()
        audio = sr.AudioData(audio_data, frame.sample_rate, 2)
        text = recognizer.recognize_google(audio)
        st.session_state.user_input = text.strip()
    except Exception:
        st.session_state.user_input = ""
    finally:
        st.session_state.processing_audio = False
    return frame

# Streamlit UI Configuration
st.set_page_config(page_title="Avinash Voice Assistant", layout="centered")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'processing_audio' not in st.session_state:
    st.session_state.processing_audio = False

# Main app interface
st.title("Avinash Vikram Singh - AI Voice Assistant")

# WebRTC component with SYNC processing
webrtc_streamer(
    key="voice-chat",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_handler,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True},
    async_processing=False  # Disable async processing
)

# Conversation handling
if st.session_state.user_input:
    user_text = st.session_state.user_input.strip()
    if user_text:
        with st.spinner("Generating response..."):
            response = generate_response(user_text)
            audio_bytes = text_to_speech(response)
            
            st.session_state.conversation.append(("You", user_text))
            st.session_state.conversation.append(("Avinash", response))
            
            # Clear input and force rerun
            st.session_state.user_input = ""
            st.experimental_rerun()

# Display conversation history
for speaker, text in st.session_state.conversation[-6:]:
    st.markdown(f"**{speaker}:** {text}")

# Text input fallback
with st.expander("Type your question"):
    text_input = st.text_input("Text input:")
    if text_input:
        response = generate_response(text_input.strip())
        st.session_state.conversation.append(("You", text_input))
        st.session_state.conversation.append(("Avinash", response))
        audio_bytes = text_to_speech(response)
        if audio_bytes:
            st.audio(audio_bytes, format='audio/mp3')
        st.experimental_rerun()
