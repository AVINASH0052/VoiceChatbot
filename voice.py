# avinash_voice_assistant.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import re
import os
from openai import OpenAI

# Initialize OpenAI client with secrets
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=st.secrets["NVIDIA_API_KEY"]
)

# Context data (keep original structure)
AVINASH_CONTEXT = {
    "personal_details": {
        "full_name": "Avinash Vikram Singh",
        "professional_identity": "Data Scientist and AI Engineer",
        # ... [rest of original context dictionary] ...
    }
}

def sanitize_text(text):
    """Ensure proper sentence endings for TTS"""
    text = re.sub(r'[^.!?]*$', '', text)
    return re.sub(r'([$`"\\])', r'\\\1', text)

def generate_response(prompt):
    """Generate response with error handling"""
    try:
        response = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[{
                "role": "system",
                "content": f"""Act as Avinash using: {AVINASH_CONTEXT}
                Rules:
                1. 2-3 sentences max
                2. Professional tone
                3. Never mention being AI
                4. Reference real experience"""
            },{
                "role": "user", 
                "content": prompt
            }],
            temperature=0.65,
            max_tokens=250
        )
        
        response_text = response.choices[0].message.content.strip()
        return sanitize_text(response_text[:500])  # Limit response length
        
    except Exception as e:
        return "Let me think differently. Could you rephrase?"

def text_to_speech(text):
    """Convert text to audio bytes with error handling"""
    try:
        tts = gTTS(text=text[:500], lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        st.error(f"Speech synthesis error: {str(e)}")
        return None

def audio_frame_handler(frame: av.AudioFrame):
    """Process audio input with error handling"""
    recognizer = sr.Recognizer()
    try:
        audio_data = frame.to_ndarray().tobytes()
        audio = sr.AudioData(audio_data, frame.sample_rate, 2)
        text = recognizer.recognize_google(audio)
        st.session_state.user_input = text
    except sr.UnknownValueError:
        st.session_state.user_input = ""
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
    return frame

# Streamlit UI Configuration
st.set_page_config(
    page_title="Avinash AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Session state initialization
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# UI Components
st.title("Avinash Vikram Singh - AI Assistant")
st.caption("Voice-enabled Professional Assistant")

# WebRTC Audio Input
webrtc_ctx = webrtc_streamer(
    key="voice-input",
    mode=WebRtcMode.SENDONLY,
    audio_frame_callback=audio_frame_handler,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"audio": True},
)

# Conversation processing
if webrtc_ctx and st.session_state.user_input:
    user_text = st.session_state.user_input.strip()
    
    if user_text and len(user_text) > 3:  # Minimum input length
        with st.spinner("Processing..."):
            bot_response = generate_response(user_text)
            audio_bytes = text_to_speech(bot_response)
            
            # Update conversation history
            st.session_state.conversation.append(("You", user_text))
            st.session_state.conversation.append(("Avinash", bot_response))
            st.session_state.user_input = ""
            
            # Display audio response
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')

# Display conversation history
for speaker, text in st.session_state.conversation[-4:]:  # Last 2 exchanges
    st.markdown(f"**{speaker}:** {text}")
    st.write("---")

# Text input fallback
with st.expander("🔍 Type your question instead"):
    text_input = st.text_input("Text input:", key="text_query")
    if text_input:
        bot_response = generate_response(text_input)
        audio_bytes = text_to_speech(bot_response)
        st.markdown(f"**Response:** {bot_response}")
        if audio_bytes:
            st.audio(audio_bytes, format='audio/mp3')
