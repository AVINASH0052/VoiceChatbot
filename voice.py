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

# Configure logging to ignore warnings
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

# Initialize OpenAI client
api_key = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY", "")
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key) if api_key else None

# Context data for AI responses
AVINASH_CONTEXT = {
    "personal_details": {
        "full_name": "Avinash Vikram Singh",
        "professional_identity": "Data Scientist and AI Engineer",
        "current_location": "Bengaluru, India",
        "contact": {
            "email": "avinashvs0052@gmail.com",
            "phone": "7052985015",
            "linkedin": "linkedin.com/in/avinash-vikram-singh-40b263233",
            "github": "github.com/AVINASH0052"
        }
    },
    "education": {
        "degree": "BSc(Honours) Computer Science",
        "university": "Delhi University",
        "graduation_year": "2023",
        "gpa": "7.5"
    },
    "technical_skills": {
        "core_competencies": ["LLM Fine-tuning", "AI API Development", "Speech Recognition"],
        "programming": ["Python", "Java", "R", "C++"],
        "ml_frameworks": ["TensorFlow", "PyTorch", "Keras"]
    }
}

def sanitize_text(text):
    """Clean text for speech synthesis."""
    text = re.sub(r'[^.!?]*$', '', text)
    return re.sub(r'([$`"\\])', r'\\\1', text)

def generate_response(prompt):
    """Generate AI response with error handling."""
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
        return sanitize_text(response.choices[0].message.content.strip()) if response and response.choices else "No response received."
    except Exception as e:
        return f"Error: {str(e)}"

def text_to_speech(text):
    """Convert text to speech."""
    try:
        tts = gTTS(text=text[:500], lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def audio_frame_handler(frame: av.AudioFrame):
    """Process audio input and convert speech to text."""
    recognizer = sr.Recognizer()
    
    try:
        audio_data = frame.to_ndarray().tobytes()
        with sr.AudioFile(BytesIO(audio_data)) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            if text and ('last_input' not in st.session_state or st.session_state.last_input != text):
                st.session_state.user_input = text
                st.session_state.last_input = text  # Avoid duplicate processing
    except Exception as e:
        st.session_state.user_input = ""
        st.warning(f"Voice recognition error: {str(e)}")

    return frame

# Streamlit UI Configuration
st.set_page_config(page_title="Avinash Voice Assistant", layout="centered")

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Main App UI
st.title("Avinash Vikram Singh - AI Voice Assistant")

# WebRTC component with fixed async handling
ctx = webrtc_streamer(
    key="voice-chat",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_handler,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True},
    async_processing=False  # Disabling async to fix missing ScriptRunContext
)

# Process user input from voice
if ctx and st.session_state.user_input:
    user_text = st.session_state.user_input.strip()
    
    if user_text and ('last_input' not in st.session_state or st.session_state.last_input != user_text):
        with st.spinner("Generating response..."):
            response = generate_response(user_text)
            audio_bytes = text_to_speech(response)

            # Append to chat history
            st.session_state.conversation.append(("You", user_text))
            st.session_state.conversation.append(("Avinash", response))
            st.session_state.last_input = user_text  # Prevent duplicate processing
            
            # Play response audio
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')

            st.session_state.user_input = ""  # Reset input
            st.rerun()

# Display conversation history
for speaker, text in st.session_state.conversation[-6:]:
    st.markdown(f"**{speaker}:** {text}")

# Text input fallback
with st.expander("Type your question"):
    text_input = st.text_input("Text input:")
    if text_input:
        response = generate_response(text_input)
        
        # Append to conversation history
        st.session_state.conversation.append(("You", text_input))
        st.session_state.conversation.append(("Avinash", response))
        
        # Convert response to speech
        audio_bytes = text_to_speech(response)
        if audio_bytes:
            st.audio(audio_bytes, format='audio/mp3')
        
        st.rerun()
