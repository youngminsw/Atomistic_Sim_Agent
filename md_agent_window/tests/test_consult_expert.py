"""
Test for consult_expert A2A feature.
Simulates the Inspection Agent giving advice when Sim Agent is stuck.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inspection_agent import InspectionAgent

def test_consult_expert():
    """Test that consult_expert returns structured advice."""
    print("=" * 60)
    print("Testing consult_expert (A2A Consultation)")
    print("=" * 60)
    
    # Create agent with a dummy work directory
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    agent = InspectionAgent(work_dir)
    
    # Simulate a stuck scenario
    problem = "Consecutive failures: Last tool was read_file - file not found 3 times"
    context = """
    Called: read_file
    Tool Result: File not found.
    Called: read_file
    Tool Result: File not found.
    Called: read_file
    Tool Result: File not found.
    """
    
    print(f"\nProblem: {problem}")
    print(f"Context: {context[:100]}...")
    print("\nCalling consult_expert...")
    
    result = agent.consult_expert(problem, context)
    
    print("\n--- Expert Response ---")
    print(f"Diagnosis: {result.get('diagnosis', 'N/A')}")
    print(f"Advice: {result.get('advice', 'N/A')}")
    print(f"Ask User: {result.get('ask_user', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    
    # Validate structure
    assert "diagnosis" in result, "Missing 'diagnosis' in response"
    assert "advice" in result, "Missing 'advice' in response"
    assert "ask_user" in result, "Missing 'ask_user' in response"
    
    print("\n✅ Test PASSED: consult_expert returns structured advice")
    return True

if __name__ == "__main__":
    try:
        test_consult_expert()
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
