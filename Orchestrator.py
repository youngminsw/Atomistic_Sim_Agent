import os
import sys
import argparse
import subprocess
import json
import shutil

# Add sub-projects to path
sys.path.append(os.path.join(os.getcwd(), "md_agent_window"))
sys.path.append(os.path.join(os.getcwd(), "ML_KMC_Model"))

class Orchestrator:
    def __init__(self):
        self.work_dir = os.getcwd()
        self.model_lib_dir = os.path.join(self.work_dir, "md_agent_window", "Reference", "mdn_model_library")
        if not os.path.exists(self.model_lib_dir):
            os.makedirs(self.model_lib_dir, exist_ok=True)
        
    def log(self, step, message):
        print(f"\n[ORCHESTRATOR] [{step}] {message}")

    def _get_lib_path(self, ion, sub):
        return os.path.join(self.model_lib_dir, f"{ion}_{sub}")

    def manage_model_library(self, action, ion, sub):
        """
        Manages the persistence of trained models.
        action: 'save' (ML -> Lib) or 'load' (Lib -> ML)
        """
        lib_path = self._get_lib_path(ion, sub)
        # Artifacts to manage
        artifacts = [
            ("checkpoints/best_mdn_model.pt", os.path.join("checkpoints", "best_mdn_model.pt")),
            ("x_scaler.pkl", "x_scaler.pkl"),
            ("y_scaler.pkl", "y_scaler.pkl"),
            ("mdn_output.csv", "mdn_output.csv") # Optional: save data too
        ]

        if action == 'save':
            self.log("Library", f"Saving model artifacts to Library: {lib_path}")
            os.makedirs(os.path.dirname(os.path.join(lib_path, "checkpoints/dummy")), exist_ok=True) # Ensure subdir exists
            
            count = 0
            for src_rel, dst_rel in artifacts:
                # Source is ML_KMC_Model dir, Dest is Library path
                src = os.path.join(self.ml_dir, src_rel)
                dst = os.path.join(lib_path, dst_rel)
                
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    count += 1
            self.log("Library", f"Saved {count} artifacts.")
            return True

        elif action == 'load':
            self.log("Library", f"Loading model artifacts from: {lib_path}")
            if not os.path.exists(lib_path):
                self.log("Library", "Model not found in library.")
                return False
                
            count = 0
            for src_rel, dst_rel in artifacts:
                # Source is Library, Dest is ML_KMC_Model
                src = os.path.join(lib_path, dst_rel)
                dst = os.path.join(self.ml_dir, src_rel)
                
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    count += 1
            
            if count > 0:
                self.log("Library", f"Loaded {count} artifacts.")
                return True
            else:
                self.log("Library", "No valid artifacts found in library folder.")
                return False

    def run_md_simulation(self, ion, substrate, energy, events):
        self.log("MD", f"Starting MD Simulation: {ion} -> {substrate} @ {energy}eV ({events} events)")
        
        # Note: sys.executable ensures we use the same python interpreter (and thus same conda env/paths)
        # as this Orchestrator is running in. This propagates the 'mss_agent' environment correctly.
        cmd = [
            sys.executable, 
            "md_agent_window/autonomous_agent.py",
            "--ion", ion,
            "--sub", substrate,
            "--energy", str(energy),
            "--events", str(events)
        ]
        try:
            # Capture output to find result path
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout) # Print MD output to console so user can see progress (or part of it)
            
            # Find output dir
            output_dir = None
            for line in result.stdout.splitlines():
                if "[OUTPUT_DIR]" in line:
                    output_dir = line.split("[OUTPUT_DIR]")[-1].strip()
                    break
            
            if output_dir and os.path.exists(output_dir):
                self.log("MD", f"Simulation Complete. Output: {output_dir}")
                return output_dir
            else:
                self.log("MD", "Simulation finished but could not find [OUTPUT_DIR] tag.")
                return None
                
        except subprocess.CalledProcessError as e:
            self.log("MD", f"Error: {e}")
            if e.stdout: print(e.stdout)
            if e.stderr: print(e.stderr)
            return None

    def run_ml_training(self, ion, sub):
        self.log("ML", "Starting Model Training...")
        # Assume dumps are in ML_KMC_Model/ or current dir.
        
        cmd = [
            sys.executable, 
            "ML_KMC_Model/02_01_Train_Model.py",
            "--x", "ML_KMC_Model/mdn_input.csv",
            "--y", "ML_KMC_Model/mdn_output.csv",
            "--out_dir", "ML_KMC_Model",
            "--epochs", "20"
        ]
        try:
            subprocess.run(cmd, check=True)
            self.log("ML", "Training Complete.")
            
            # Save to Library
            self.manage_model_library('save', ion, sub)
            return True
        except subprocess.CalledProcessError as e:
            self.log("ML", f"Error: {e}")
            return False

    def run_kmc_simulation(self, energy, angle):
        self.log("KMC", f"Starting KMC Simulation @ {energy}eV...")
        
        # We must run in the ML_KMC_Model directory because total_model/Infer_Model use relative paths for checkpoints
        kmc_script = "04_KMC_tool.py" 
        
        cmd = [sys.executable, kmc_script]
        try:
            # Run in the ML_KMC_Model directory
            subprocess.run(cmd, cwd=self.ml_dir, check=True)
            self.log("KMC", "Simulation Complete.")
        except subprocess.CalledProcessError as e:
            self.log("KMC", f"Error: {e}")

            
    def _copy_file(self, src, dst):
        try:
            if os.path.exists(src):
                shutil.copy2(src, dst)
                self.log("Transfer", f"Copied {os.path.basename(src)} -> ML Dir")
                return True
            else:
                self.log("Transfer", f"Missing source file: {src}")
                return False
        except Exception as e:
            self.log("Transfer", f"Copy failed: {e}")
            return False

    def parse_intent(self, user_request):
        """Uses LLM to parse natural language request into structured action."""
        if not self.llm_client:
            print("[Error] LLM Client not available. Cannot parse intent.")
            return {"clarification_needed": True, "question": "LLM Setup Failed. Check logs."}
        
        system_prompt = """
        You are the Orchestrator for a Multi-Scale Simulation Platform.
        Your job is to map user requests to specific pipeline stages.
        
        PIPELINE STAGES:
        1. MD (Molecular Dynamics): Generates atomic data (ion bombardment). Needs: ion, substrate, energy, events.
        2. ML (Machine Learning): Trains the surrogate model. Needs: nothing specific (automates dump2csv + train).
        3. KMC (Kinetic Monte Carlo): Simulates device etching/deposition. Needs: trench_image (optional description), energy.
        
        INSTRUCTIONS:
        - Identify which stage(s) the user wants.
        - Extract parameters (Ion, Substrate, Energy, Events).
        - IMPORTANT: If Ion/Substrate are not explicitly mentioned, infer reasonable defaults (e.g., Ar on Si) BUT ask for clarification if totally ambiguous.
        - If the user implies using EXISTING data/model (e.g. "Run MC with this"), skip MD/ML stages.
        - If the user says "Full simulation", implies ALL stages (MD -> ML -> KMC).
        
        Return JSON format:
        {
            "stages": ["MD", "ML", "KMC"] or ["KMC"] etc.,
            "params": {
                "ion": "Ar", "sub": "Si", "energy": 100.0, "events": 20,
                "trench_desc": "..." 
            },
            "clarification_needed": boolean,
            "question": "..." (if clarification needed)
        }
        """
        
        response = self.llm_client.generate_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request}
        ])
        return response

    def run_interactive(self):
        print("\n" + "="*60)
        print("🤖 Multi-Scale Agent Orchestrator (Interactive Mode)")
        print("="*60, flush=True)
        print("Tell me what you want to do. Examples:", flush=True)
        print(" - 'Run a full simulation for Ar on Si at 100eV'", flush=True)
        print(" - 'I have a model, just run the KMC simulation for this trench'", flush=True)
        print(" - 'Run MD only for 50 events'", flush=True)
        
        while True:
            try:
                user_input = input("\n[User]> ").strip()
                if not user_input: continue
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                print("[Thinking...]", flush=True)
                intent = self.parse_intent(user_input)
                
                if intent.get("clarification_needed"):
                    print(f"[Orchestrator] {intent.get('question', 'Could you clarify?')}", flush=True)
                    continue
                    
                stages = intent.get('stages', [])
                params = intent.get('params', {})
                print(f"[Plan] Stages: {stages}", flush=True)
                print(f"[Plan] Params: {params}", flush=True)
                
                confirm = input("[Proceed?] (y/n) > ").strip()
                if confirm.lower() != 'y':
                    print("Cancelled.", flush=True)
                    continue
                    
                self.execute_pipeline(intent)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[Error] {e}", flush=True)
                import traceback
                traceback.print_exc() 

    def execute_pipeline(self, intent):
        stages = intent.get("stages", [])
        params = intent.get("params", {})
        
        self.log("INIT", f"Executing Pipeline: {stages}")
        
        # Defaults
        ion = params.get("ion", "Ar")
        sub = params.get("sub", "Si")
        energy = float(params.get("energy", 100.0))
        events = int(params.get("events", 20))
        
        md_out_dir = None
        
        # 1. MD Stage
        if "MD" in stages:
            md_out_dir = self.run_md_simulation(ion, sub, energy, events)
            if not md_out_dir:
                self.log("Stop", "MD Simulation failed.")
                return
            
            # Transfer Data regardless of ML step (might be needed later)
            inc_dump = os.path.join(md_out_dir, "incident.dump")
            ref_dump = os.path.join(md_out_dir, "reflected.dump")
            target_inc = os.path.join(self.ml_dir, "incident.dump")
            target_ref = os.path.join(self.ml_dir, "reflected.dump")
            self._copy_file(inc_dump, target_inc)
            self._copy_file(ref_dump, target_ref)

        # 2. ML Stage
        if "ML" in stages:
            # Check dependency
            if "MD" not in stages and not (os.path.exists(os.path.join(self.ml_dir, "incident.dump"))):
                self.log("Warn", "ML requested but no dump files found. Training might fail.")
                
            self.log("ML", "Converting Dumps to CSV...")
            try:
                cmd = [sys.executable, "ML_KMC_Model/01_Dump2csv.py", "--workdir", "ML_KMC_Model"]
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                 self.log("ML", f"Dump2CSV failed: {e}")
                 return
                 
            if not self.run_ml_training(ion, sub):
                return
        
        # 3. KMC Stage
        if "KMC" in stages:
            # Check model availability mechanism
            # Case A: We just trained it (ML stage ran) -> It's already in ML_KMC_Model/checkpoints
            # Case B: We skipped ML -> We must LOAD it from Library or check if it exists
            
            model_path = os.path.join(self.ml_dir, "checkpoints", "best_mdn_model.pt")
            
            if "ML" not in stages:
                # Try to load from library if not currently present or if user requested specific material
                # To be safe, ALWAYS try to load from library if ML was skipped, to ensure we use the requested material's model
                self.log("KMC", f"Checking Model Library for {ion}/{sub}...")
                if self.manage_model_library('load', ion, sub):
                    self.log("KMC", "Model loaded from library.")
                else:
                    self.log("Warn", "Model not found in library. Using whatever is currently in local buffer (if any).")
            
            if not os.path.exists(model_path):
                 self.log("Stop", f"KMC requested but no valid model found at {model_path}. Train ML first or check library.")
                 return
            
            # TODO: Handle 'trench_image' param if provided (future)
            self.run_kmc_simulation(energy, 0.0)
        
        self.log("DONE", "Execution Finished.")

if __name__ == "__main__":
    orch = Orchestrator()
    # Check args for non-interactive mode
    if len(sys.argv) > 1:
        # Simple test hardcoded
        req = {
            "stages": ["MD", "ML", "KMC"],
            "params": { "ion": "Ar", "sub": "Si", "energy": 100.0, "events": 20 }
        }
        orch.execute_pipeline(req)
    else:
        orch.run_interactive()
