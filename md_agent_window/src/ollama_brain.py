import requests
import time
import json

class OllamaBrain:
    """
    A standalone, reusable client for connecting to a local network Ollama server.
    """
    
    def __init__(self, server_ip="10.24.12.85", port=11434, model="glm-4.7-flash:latest"):
        self.server_ip = server_ip
        self.port = port
        self.base_url = f"http://{server_ip}:{port}"
        self.model = model
        
    def generate_response(self, prompt, system_prompt="You are a helpful AI assistant.", temperature=0.7, timeout=120):
        """
        Generates a response from the Ollama model.
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        
        print(f"[OllamaBrain] Sending request to {self.base_url} (Model: {self.model})...", flush=True)
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=timeout) 
            response.raise_for_status()
            duration = time.time() - start_time
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.ConnectionError:
            return f"Error: Could not connect to server at {self.base_url}. Is it running?"
        except Exception as e:
            return f"Error: {str(e)}"
