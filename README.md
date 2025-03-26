# AI Professional Assistant Chatbot 🤖

A voice-enabled AI assistant representing Avinash Vikram Singh, featuring voice/text input and context-aware responses using NVIDIA's Llama3-70B model.

[![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?logo=flask)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![NVIDIA](https://img.shields.io/badge/NVIDIA_AI-API-76B900?logo=nvidia)]([https://integrate.nvidia.com](https://build.nvidia.com/))


## Features ✨
- 🎙️ Voice input via Web Speech API
- ⌨️ Text input fallback
- 🔊 Text-to-speech with Indian accent preference
- 🛑 Interactive stop button
- 🔐 Secure API key management
- 📱 Responsive design

## Installation 🛠️

### Prerequisites
- NVIDIA API key ([Get here](https://integrate.nvidia.com))
- Python 3.11+

### Local Setup
```bash
git clone https://github.com/yourusername/avinash-chatbot.git
cd avinash-chatbot
pip install -r requirements.txt
export NVIDIA_API_KEY="your_api_key_here"
flask run --port=5000
