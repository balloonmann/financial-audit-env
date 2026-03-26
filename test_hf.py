import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("HF_TOKEN")
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

headers = {"Authorization": f"Bearer {TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

if not TOKEN:
    print("Error: HF_TOKEN not found in .env")
else:
    print(f"Testing HF Token: {TOKEN[:8]}...")
    try:
        output = query({
            "inputs": "Can you hear me?",
            "parameters": {"max_new_tokens": 10}
        })
        print("Response:", output)
    except Exception as e:
        print("Error connecting to Hugging Face:", e)
