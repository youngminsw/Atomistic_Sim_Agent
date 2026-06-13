
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tools_lib import AgentTools
from src.config import Config

def test_model_switching():
    print("=== Testing Model Switching Tools ===")
    
    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../workspace"))
    os.makedirs(work_dir, exist_ok=True)
    
    tools = AgentTools(work_dir)
    
    # 1. Test Reading Registry
    print("\n[Test 1] Reading Model Registry...")
    registry_content = tools.read_model_registry()
    if "direct/local-network-ollama" in registry_content and "Low" in registry_content:
        print("✓ Registry read successfully.")
        print(f"snippet: {registry_content[:150]}...")
    else:
        print("X Registry read failed or content missing.")
        return

    # 2. Test Switching Model (Simulated)
    # Note: We won't actually change the file if we want to preserve state, 
    # but the tool modifies config.py. 
    # Let's switch to 'verification-test-model' (will fail) then back to 'direct/local-network-ollama'
    
    print("\n[Test 2] Switching to invalid model...")
    result = tools.switch_sim_agent_model("invalid-model")
    print(f"Result: {result}")
    if "Error" in result:
        print("✓ Invalid model correctly rejected.")
    else:
        print("X Invalid model was NOT rejected.")
        
    print("\n[Test 3] Switching to 'direct/local-network-ollama'...")
    # This should succeed as it is in registry
    result = tools.switch_sim_agent_model("direct/local-network-ollama")
    print(f"Result: {result}")
    
    # Verify config.py actually changed (or stayed same if it was already that)
    from src.config import Config
    # Reload config parsing manually to check file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "config.py")
    with open(config_path, "r") as f:
        content = f.read()
    
    if 'SIM_MODEL_NAME = "direct/local-network-ollama"' in content:
        print("✓ config.py confirmed to have correct model.")
    else:
        print("X config.py does not match expected state.")

if __name__ == "__main__":
    test_model_switching()
