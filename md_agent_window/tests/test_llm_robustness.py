
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.llm_client import LLMClient
from src.agent_core import AgentEngine
from src.inspection_agent import InspectionAgent

class TestLLMRobustness(unittest.TestCase):
    
    def test_model_separation(self):
        """Verify Sim Agent and Inspector use different models/configs."""
        print("\n[Test] Verifying Model Separation...")
        
        # Mock Config to ensure we have distinct names
        with patch("src.config.Config.SIM_MODEL_NAME", "sim-model-v1"), \
             patch("src.config.Config.INSPECTION_MODEL_NAME", "inspection-model-v1"):
             
            agent = AgentEngine("test_workdir")
            inspector = InspectionAgent("test_workdir")
            
            print(f"  Sim Model: {agent.client.model}")
            print(f"  Inspector Model: {inspector.client.model}")
            
            self.assertEqual(agent.client.model, "sim-model-v1")
            self.assertEqual(inspector.client.model, "inspection-model-v1")
            self.assertNotEqual(agent.client.model, inspector.client.model)
            print("  [OK] Models are separated.")

    @patch("src.antigravity_client.AntigravityClient")
    def test_timeout_propagation(self, mock_ag_client):
        """Verify timeout is passed to AntigravityClient."""
        print("\n[Test] Verifying Timeout Propagation...")
        
        mock_instance = MagicMock()
        mock_ag_client.return_value = mock_instance
        mock_instance.generate_content.return_value.text = "OK"
        
        with patch("src.config.Config.LLM_TIMEOUT", 42):
            client = LLMClient(model_name="test-model")
            client.generate_response([{"role": "user", "content": "hi"}])
            
            # Check init call args
            mock_ag_client.assert_called_with(model_name="test-model", timeout=42)
            print("  [OK] Timeout 42s passed to AntigravityClient.")

    def test_looping_fallback(self):
        """Verify fallback logic cycles through models."""
        print("\n[Test] Verifying Looping Fallback...")
        
        fallback_models = ["model-A", "model-B"]
        
        with patch("src.config.Config.FALLBACK_MODELS", fallback_models), \
             patch("src.config.Config.MAX_FAILOVER_LOOPS", 2):
            
            client = LLMClient(model_name="initial-model")
            
            # 1. First failover -> model-A
            success = client._try_failover()
            self.assertTrue(success)
            self.assertEqual(client.model, "model-A")
            print(f"  Step 1: Switched to {client.model}")
            
            # 2. Second failover -> model-B
            success = client._try_failover()
            self.assertTrue(success)
            self.assertEqual(client.model, "model-B")
            print(f"  Step 2: Switched to {client.model}")
            
            # 3. Third failover -> model-A (Loop!)
            success = client._try_failover()
            self.assertTrue(success)
            self.assertEqual(client.model, "model-A")
            print(f"  Step 3: Looped back to {client.model}")
            
            # 4. Fourth failover -> model-B
            success = client._try_failover()
            self.assertTrue(success)
            self.assertEqual(client.model, "model-B")
            print(f"  Step 4: Looped back to {client.model}")
            
            # 5. Fifth failover -> Exhausted (2 loops * 2 models = 4 switches allowed? 
            # Actually limit check is >= total_limit. count starts at 0.
            # 0 -> A (count becomes 1)
            # 1 -> B (count becomes 2)
            # 2 -> A (count becomes 3)
            # 3 -> B (count becomes 4)
            # 4 -> Limit reached?
            # total_limit = 2 * 2 = 4.
            # If attempts (4) >= 4, it should fail.
            
            success = client._try_failover()
            self.assertFalse(success)
            print("  Step 5: Failover exhausted limits correctly.")

if __name__ == "__main__":
    unittest.main()
