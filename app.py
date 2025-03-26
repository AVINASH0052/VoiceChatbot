import os
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize NVIDIA client with environment variable
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ['NVIDIA_API_KEY']  # Will fail if key not set (secure by default)
)

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
    # ... (rest of your existing AVINASH_CONTEXT dictionary) ...
}

def sanitize_text(text):
    """Clean text for speech synthesis"""
    return re.sub(r'([$`"\\])', r'\\\1', text)

def generate_response(prompt):
    """Generate responses using NVIDIA API"""
    try:
        response = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[{
                "role": "system",
                "content": f"""You are Avinash Vikram Singh. Respond conversationally using:
                {AVINASH_CONTEXT}
                
                Response Guidelines:
                1. Complete all sentences properly
                2. Keep responses concise (2-3 sentences)
                3. Maintain professional tone
                4. Never mention being an AI"""
            }, {
                "role": "user", 
                "content": prompt
            }],
            temperature=0.65,
            max_tokens=250,
            frequency_penalty=0.25
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Ensure response ends with punctuation
        if not response_text.endswith(('.', '!', '?')):
            last_sentence = re.findall(r'.*?[.!?]', response_text)
            response_text = last_sentence[0] if last_sentence else response_text + "."
        
        return response_text
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        return "Let me think about that differently. Could you rephrase?"

@app.route('/')
def home():
    """Serve the chat interface"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle chat requests"""
    try:
        user_input = request.json.get('message', '').strip()
        if not user_input:
            return jsonify({'error': 'Empty input'}), 400
            
        response = generate_response(user_input)
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 10000
    port = int(os.environ.get('PORT', 10000))
    # Run with Gunicorn-compatible settings
    app.run(host='0.0.0.0', port=port, debug=False)
