import sys
import os
import shutil
import argparse

# Ensure we can import modules from src/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.agent_core import AgentEngine

def solve_user_request(request_dict):
    print(f"\n>>> [AGENT] Initializing ReAct Engine...")
    
    # 1. Setup Work Directory inside /app/results for bind mount access
    job_id_str = f"{request_dict['ion']}_{request_dict['sub']}_{request_dict['events']}evts"
    
    # Use 'results' folder for Docker bind mount compatibility
    results_base = os.path.join(os.getcwd(), "results")
    os.makedirs(results_base, exist_ok=True)
    
    work_dir = os.path.join(results_base, f"run_{job_id_str}")
    
    if os.path.exists(work_dir):
        try:
            shutil.rmtree(work_dir)
        except Exception as e:
            print(f">>> [Warning] Could not clear '{work_dir}': {e}")
            import time
            work_dir = f"{work_dir}_{int(time.time())}"
            print(f">>> [Info] Switched to new work directory: {work_dir}")
            
    os.makedirs(work_dir, exist_ok=True)
    
    # 2. Construct Natural Language Goal
    goal = f"""
    I want to simulate {request_dict['ion']} ion bombardment on {request_dict['sub']} substrate.
    
    Parameters:
    - Ion: {request_dict['ion']}
    - Substrate: {request_dict['sub']}
    - Energy: {request_dict['energy']} eV
    - Events: {request_dict['events']}
    
    Work Directory: {work_dir}
    
    Task:
    1. Research the materials.
    2. Build the simulation domain.
    3. Generate LAMMPS input.
    4. Run the simulation.
    5. If it fails, fix the error and retry until success.
    """
    
    # 3. Launch Engine
    # AgentEngine expects work_dir path string
    engine = AgentEngine(work_dir)
    engine.run(goal)
    print(f"\n[OUTPUT_DIR] {work_dir}")
    return work_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous MD Agent")
    parser.add_argument("--ion", type=str, required=True)
    parser.add_argument("--sub", type=str, required=True)
    parser.add_argument("--energy", type=float, required=True)
    parser.add_argument("--events", type=int, default=100)
    
    args = parser.parse_args()
    REQ = {
        "ion": args.ion, 
        "sub": args.sub, 
        "energy": args.energy, 
        "events": args.events
    }
    solve_user_request(REQ)
