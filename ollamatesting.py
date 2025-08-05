import requests

# Ollama's default endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Choose your model (make sure it's downloaded and running)
model = "llama3"

# Your prompt
prompt = "What is the capital of France?"

# Prepare the request payload
payload = {
    "model": model,
    "prompt": prompt,
    "stream": False  # set to True if you want streaming responses
}

# Send POST request to Ollama
response = requests.post(OLLAMA_URL, json=payload)

# Handle response
if response.status_code == 200:
    result = response.json()
    print("Response from Ollama:")
    print(result['response'])
else:
    print("Error:", response.status_code, response.text)
