# Local Network Ollama Server Integration Request

I need to integrate a local network Ollama server (running on an RTX 5090 in the same network) into my Python agent project.

## Server Details
- **IP**: `10.24.12.85`
- **Port**: `11434`
- **Model**: `glm-4.7-flash:latest`

## Requirements

### 1. Create `ollama_brain.py`
Please create a standalone, reusable module named `ollama_brain.py` (in the `src` directory or project root) with the following code. This module handles the connection and timeouts robustly.

```python
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
        
    def generate_response(self, prompt, system_prompt="You are a helpful AI assistant.", temperature=0.7):
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
            response = requests.post(url, json=payload, timeout=120) 
            response.raise_for_status()
            duration = time.time() - start_time
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.ConnectionError:
            return f"Error: Could not connect to server at {self.base_url}. Is it running?"
        except Exception as e:
            return f"Error: {str(e)}"
```

### 2. Update `config.py`
Add the server configuration and register it as a fallback model.

```python
class Config:
    # ... existing config ...
    
    # [NEW] LOCAL NETWORK OLLAMA SERVER (RTX 5090)
    LOCAL_SERVER_IP = "10.24.12.85"
    LOCAL_SERVER_MODEL = "glm-4.7-flash:latest"

    FALLBACK_MODELS = [
        # ... existing models ...
        "direct/local-network-ollama",  # Add this priority entry
        # ...
    ]
```

### 3. Update `llm_client.py`
Modify the `generate_response` method in `LLMClient` to handle the `direct/local-network-ollama` key.

```python
# Inside generate_response method loop or logic:

elif self.model == "direct/local-network-ollama":
    from ollama_brain import OllamaBrain # Adjust import path as needed
    
    ip = getattr(Config, "LOCAL_SERVER_IP", "10.24.12.85")
    model = getattr(Config, "LOCAL_SERVER_MODEL", "glm-4.7-flash:latest")
    
    print(f"   [LLM-Client] Switching to Local Network Ollama ({ip})...")
    brain = OllamaBrain(server_ip=ip, model=model)
    text_response = brain.generate_response(prompt_str, system_prompt=system_instruction)
```

Please apply these changes to ensure the agent can utilize the local high-performance server.
