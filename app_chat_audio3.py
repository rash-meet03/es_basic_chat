import os
import time
import asyncio
import uuid
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

import pickle
import numpy as np
import faiss
import requests
from flask import Flask, request, jsonify, render_template
import markdown  # For converting markdown to HTML
from openai import AsyncOpenAI
from concurrent.futures import ThreadPoolExecutor

# Initialize OpenAI instance for TTS with the API key explicitly passed
openai_tts = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
DATA_DIR = "data"
CHAT_HISTORY_FILE = os.path.join(DATA_DIR, "chat_history.pkl")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Load chat history if exists
if os.path.exists(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, "rb") as f:
        chat_history = pickle.load(f)
else:
    chat_history = []

def save_chat_history():
    """Save chat history to disk"""
    with open(CHAT_HISTORY_FILE, "wb") as f:
        pickle.dump(chat_history, f)

def post_with_retries(url, headers, json_data, max_retries=3, backoff_factor=2):
    """
    Makes a POST request to the given URL with retries if a 429 status is encountered.
    Exponential backoff is used for waiting between retries.
    """
    attempt = 0
    while attempt < max_retries:
        response = requests.post(url, headers=headers, json=json_data)
        if response.status_code == 429:
            wait_time = backoff_factor ** attempt
            print(f"Received 429 error. Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            attempt += 1
        else:
            return response
    return response  # Return the last response if all retries fail

async def generate_tts_audio(text: str, voice: str = "coral") -> str:
    """
    Uses OpenAI's TTS API to generate audio from the provided text.
    The audio is saved as an MP3 file in the 'static/audio' directory.
    Returns the relative file path for the generated audio.
    """
    tts_instructions = (
        "Voice: Clear, authoritative, and composed, projecting confidence and professionalism.\n\n"
        "Tone: Neutral and informative, maintaining a balance between formality and approachability.\n\n"
        "Punctuation: Structured with commas and pauses for clarity, ensuring information is digestible and well-paced.\n\n"
        "Delivery: Steady and measured, with slight emphasis on key figures and deadlines to highlight critical points."
    )
    
    async with openai_tts.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions=tts_instructions,
        response_format="mp3",  # Use MP3 for directly playable output
    ) as response:
        # Await the complete binary response
        audio_data = await response.read()
    
    audio_dir = os.path.join("static", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    filename = f"audio_{int(time.time())}.mp3"
    filepath = os.path.join(audio_dir, filename)
    with open(filepath, "wb") as f:
        f.write(audio_data)
    
    return f"/static/audio/{filename}"

# Use ThreadPoolExecutor for background task processing.
executor = ThreadPoolExecutor(max_workers=4)
# Dictionary to store task futures keyed by a unique task ID.
task_futures = {}

@app.route('/')
def chat_ui():
    """Serve the chat interface"""
    return render_template('chat3.html')

@app.route('/get_history')
def get_history():
    """Return chat history for initial load"""
    return jsonify(chat_history[-10:])  # Return last 10 interactions

@app.route('/audio_status/<task_id>', methods=['GET'])
def audio_status(task_id):
    """Check the status of the TTS audio generation background task."""
    future = task_futures.get(task_id)
    if not future:
        return jsonify({'state': 'NOT_FOUND'})
    if future.done():
        try:
            result = future.result()
            # Optionally remove the future from the dictionary once done.
            del task_futures[task_id]
            return jsonify({'state': 'SUCCESS', 'audio_url': result})
        except Exception as e:
            return jsonify({'state': 'FAILURE', 'error': str(e)})
    else:
        return jsonify({'state': 'PENDING'})

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests and RAG pipeline with background TTS integration."""
    query = request.json['message']
    
    try:
        # Correct FAISS index loading
        index = faiss.read_index(os.path.join(DATA_DIR, "faiss_index.index"))
        with open(os.path.join(DATA_DIR, "text_chunks.pkl"), "rb") as f:
            chunks = pickle.load(f)
    except FileNotFoundError as e:
        return jsonify({"error": f"Missing files: {str(e)}"}), 400

    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    embedding_payload = {"input": [query], "model": "mistral-embed"}
    embed_response = post_with_retries("https://api.mistral.ai/v1/embeddings", headers, embedding_payload)
    
    if embed_response.status_code != 200:
        return jsonify({"error": "Embedding API call failed"}), 500
    
    query_embedding = embed_response.json()['data'][0]['embedding']
    D, I = index.search(np.array([query_embedding]).astype('float32'), 3)
    context = "\n".join(chunks[i] for i in I[0])
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Provide responses only in plain text without any markdown symbols (such as **, *, etc.) even when listing pointers. Format responses as clear, readable paragraphs or bullet points using plain text only."},
        *[{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, msg in enumerate(chat_history[-10:])],
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    
    chat_payload = {
        "model": "ministral-8b-2410",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    chat_response = post_with_retries("https://api.mistral.ai/v1/chat/completions", headers, chat_payload)
    
    if chat_response.status_code != 200:
        return jsonify({"error": "Chat API call failed"}), 500
    
    answer = chat_response.json()['choices'][0]['message']['content']
    html_answer = markdown.markdown(answer)
    
    # Offload TTS generation to background thread.
    task_id = str(uuid.uuid4())
    future = executor.submit(lambda: asyncio.run(generate_tts_audio(answer)))
    task_futures[task_id] = future
    
    chat_history.extend([query, answer])
    save_chat_history()
    
    return jsonify({
        "answer": html_answer,
        "tts_task_id": task_id,  # Return task ID so client can poll for audio.
        "history": chat_history[-10:]
    })

if __name__ == '__main__':
    app.run(port=5001, debug=False)
