import os

class CodePatcher:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        
    def apply_fix(self, fix_plan):
        """
        Applies a fix described by JSON.
        Expected format:
        {
            "file": "in.sputtering",
            "action": "replace_line" | "replace_text",
            "line": 10, (for replace_line)
            "old_text": "...", (for replace_text)
            "new_text": "..."
        }
        """
        target_file = fix_plan.get("file")
        full_path = os.path.join(self.work_dir, target_file)
        
        if not os.path.exists(full_path):
            print(f"[Coder] Error: File {target_file} not found.")
            return False
            
        print(f"[Coder] Applying fix to {target_file}...")
        
        with open(full_path, "r") as f:
            lines = f.readlines()
            
        action = fix_plan.get("action")
        
        if action == "replace_line":
            line_idx = fix_plan.get("line") - 1 # 1-based to 0-based
            if 0 <= line_idx < len(lines):
                original = lines[line_idx]
                lines[line_idx] = fix_plan.get("new_text") + "\n"
                print(f"   Line {fix_plan.get('line')}: '{original.strip()}' -> '{fix_plan.get('new_text').strip()}'")
            else:
                print(f"   [Coder] Invalid line number {fix_plan.get('line')}")
                
        elif action == "replace_text":
            old = fix_plan.get("old_text")
            new = fix_plan.get("new_text")
            content = "".join(lines)
            if old in content:
                content = content.replace(old, new)
                lines = content.splitlines(keepends=True)
                print(f"   Text replaced: '{old}' -> '{new}'")
            else:
                print(f"   [Coder] Old text not found.")
                
        with open(full_path, "w") as f:
            f.writelines(lines)
            
        return True
