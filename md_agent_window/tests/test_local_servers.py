import requests
import json
import time
import sys
import os

# Add project root to path to import config if needed, 
# but for this standalone test we can also just hardcode or read from Config class structure
# Let's try to import Config to be dynamic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.config import Config
    IP_5090 = getattr(Config, "LOCAL_SERVER_5090_IP", "10.24.12.85")
    MODEL_5090 = getattr(Config, "LOCAL_SERVER_5090_MODEL", "glm-4.7-flash:latest")
    
    IP_4090 = getattr(Config, "LOCAL_SERVER_4090_IP", "10.24.12.81")
    MODEL_4090 = getattr(Config, "LOCAL_SERVER_4090_MODEL", "qwen3-next:latest")
except ImportError:
    # Fallback if config import fails
    print("Warning: Could not import Config. Using default IPs.")
    IP_5090 = "10.24.12.85"
    MODEL_5090 = "glm-4.7-flash:latest"
    IP_4090 = "10.24.12.81"
    MODEL_4090 = "qwen3-next:latest"

def test_ollama_server(name, ip, model):
    url = f"http://{ip}:11434/api/generate"
    print(f"\n[{name}] Testing Connection...")
    print(f"  Target: {url}")
    print(f"  Model:  {model}")
    
    # 1. Ping Test
    import subprocess
    import platform
    param = '-n' if platform.system().lower()=='windows' else '-c'
    command = ['ping', param, '1', ip]
    
    try:
        if ip not in ["localhost", "127.0.0.1"]:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                print(f"  [OK] Ping to {ip} successful.")
            else:
                print(f"  [FAIL] Ping to {ip} FAILED. Machine might be down or unreachable.")
                return # Stop if ping fails
    except Exception as e:
        print(f"  [WARN] Ping check skipped: {e}")
    
    payload = {
        "model": model,
        "prompt": "Hello! Reply with 'OK' if you are working.",
        "stream": False,
        "keep_alive": "5m"  # Keep in memory for 5 mins
    }
    
    try:
        start_time = time.time()
        # Increased timeout to 300s (5 mins) for heavy model loading
        response = requests.post(url, json=payload, timeout=300) 
        response.raise_for_status()
        data = response.json()
        latency = (time.time() - start_time) * 1000
        
        print(f"  [OK] Success! (Latency: {latency:.2f}ms)")
        print(f"  [OK] Response: {data.get('response', '').strip()}")
        
    except requests.exceptions.ConnectionError:
        print(f"  [FAIL] Failed: Could not connect to {ip}. Is Ollama running and listening on 0.0.0.0?")
        
        # Fallback check for Windows (often binds only to localhost)
        if ip != "127.0.0.1" and ip != "localhost":
            print(f"  [INFO] Trying fallback to localhost...")
            try:
                fallback_url = f"http://127.0.0.1:11434/api/generate"
                requests.post(fallback_url, json=payload, timeout=5)
                print(f"  [HINT] Connection to 127.0.0.1 SUCCEEDED! This means Ollama is running but NOT listening on {ip}.")
                print(f"         Set OLLAMA_HOST=0.0.0.0:11434 environment variable on the server.")
            except:
                print(f"  [INFO] Fallback to localhost also failed.")
                
    except requests.exceptions.ReadTimeout:
        print(f"  [FAIL] Failed: Read Timeout (Model might be loading).")
    except Exception as e:
        print(f"  [FAIL] Error: {e}")

if __name__ == "__main__":
    print("=== Local LLM Server Connectivity Test ===")
    
    # Test Server 1 (5090)
    test_ollama_server("Server 1 (RTX 5090)", IP_5090, MODEL_5090)
    
    # Test Server 2 (4090)
    test_ollama_server("Server 2 (RTX 4090)", IP_4090, MODEL_4090)
    
    print("\n=== Test Complete ===")
