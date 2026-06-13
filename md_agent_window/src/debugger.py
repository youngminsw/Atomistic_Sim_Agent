from src.llm_client import LLMClient
import os
import re

class SimulationDebugger:
    def __init__(self):
        self.brain = LLMClient()
        
    def analyze_error(self, log_content, context_file):
        """
        Determines the type of error (LAMMPS vs Python) and routes it.
        context_file: The primary file involved (in.sputtering or script.py)
        """
        
        # Check if it's a Python Traceback
        if "Traceback (most recent call last)" in log_content:
            return self._analyze_python_error(log_content)
        else:
            return self._analyze_lammps_error(log_content, context_file)

    def _analyze_lammps_error(self, error_log, input_file_path):
        with open(input_file_path, "r") as f:
            input_content = f.read()
            
        prompt = f"""
        You are an expert in LAMMPS debugging.
        Input Script ({os.path.basename(input_file_path)}):
        ```
        {input_content}
        ```
        Error Log:
        ```
        {error_log}
        ```
        Task: Identify the command causing the error and propose a JSON fix.
        """
        return self._query_llm(prompt, os.path.basename(input_file_path))

    def _analyze_python_error(self, traceback_log):
        # Extract filename from traceback
        # File "d:\...\src\researcher.py", line 10, in ...
        match = re.search(r'File "([^"]+)", line (\d+)', traceback_log)
        if match:
            broken_file = match.group(1)
            line_no = match.group(2)
        else:
            print("[Debugger] Could not parse filename from traceback.")
            return None
            
        if not os.path.exists(broken_file):
            print(f"[Debugger] File {broken_file} not found locally.")
            return None
            
        with open(broken_file, "r") as f:
            code_content = f.read()
            
        prompt = f"""
        You are a Python Expert. The code crashed.
        
        File: {os.path.basename(broken_file)}
        Line: {line_no}
        
        Code:
        ```python
        {code_content}
        ```
        
        Traceback:
        ```
        {traceback_log}
        ```
        
        Task: Fix the Python code to resolve the error.
        
        Output JSON (one of):
        {{ "file": "{os.path.basename(broken_file)}", "action": "replace_line", "line": {line_no}, "new_text": "..." }}
        {{ "file": "{os.path.basename(broken_file)}", "action": "replace_text", "old_text": "...", "new_text": "..." }}
        """
        return self._query_llm(prompt, os.path.basename(broken_file))

    def _query_llm(self, prompt, target_filename):
        messages = [
            {"role": "system", "content": "You are a Code Repair AI. Return JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        plan = self.brain.generate_json(messages, temperature=0.1)
        
        # Default filename if missing in LLM response
        if plan and "file" not in plan:
            plan["file"] = target_filename
            
        print(f"   [Debugger] Proposed Fix: {plan}")
        return plan
