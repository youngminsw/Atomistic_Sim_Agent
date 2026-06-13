
import os
import csv
import time
from datetime import datetime

class TokenTracker:
    def __init__(self, work_dir):
        self.log_file = os.path.join(work_dir, "token_usage.csv")
        self._ensure_header()
    
    def _ensure_header(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Model", "Input_Tokens", "Output_Tokens", "Total_Tokens"])
                
    def log_usage(self, model, input_tokens, output_tokens):
        try:
            timestamp = datetime.now().isoformat()
            total_tokens = input_tokens + output_tokens
            
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, model, input_tokens, output_tokens, total_tokens])
                
        except Exception as e:
            print(f"[TokenTracker] Failed to log usage: {e}")
