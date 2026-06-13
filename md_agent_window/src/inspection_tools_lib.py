"""
Inspection Tools Library - Read-only tools for Inspection Agent.
Inherits from AgentTools but restricts write operations.
"""

import os
import warnings
from typing import List, Dict, Any

class InspectionTools:
    """
    Read-only tools for Inspection Agent.
    Provides file reading, searching, and OVITO rendering capabilities.
    """
    
    def __init__(self, work_dir: str = "."):
        self.work_dir = work_dir
    
    def read_file(self, file_path: str = None, filename: str = None, 
                  max_lines: int = 0, start_line: int = 0) -> Dict[str, Any]:
        """
        Reads a file from the filesystem.
        Args:
            file_path: Absolute path to the file
            filename: Relative path (work_dir based)
            max_lines: Max lines to read (0 = all)
            start_line: Line to start from (0-indexed)
        """
        # Determine path
        if file_path:
            path = file_path
        elif filename:
            if os.path.isabs(filename):
                path = filename
            else:
                path = os.path.join(self.work_dir, filename)
        else:
            return {"success": False, "error": "Either file_path or filename required"}
        
        if not os.path.exists(path):
            return {"success": False, "error": f"File not found: {path}"}
        
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            if start_line > 0:
                lines = lines[start_line:]
            if max_lines > 0:
                lines = lines[:max_lines]
            
            return {
                "success": True,
                "path": path,
                "content": "".join(lines),
                "total_lines": total_lines,
                "lines_read": len(lines)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def bash(self, command: str, description: str = "") -> Dict[str, Any]:
        """
        Execute read-only shell commands.
        Blocks write operations like rm, mv, cp, >, etc.
        """
        import subprocess
        
        # Block write commands
        BLOCKED = [
            "rm ", "rm\t", "rmdir",
            "mv ", "mv\t", 
            "cp ", "cp\t",
            "> ", ">>",
            "tee ",
            "chmod", "chown",
            "touch ",
            "mkdir ",
            "dd ",
            "wget ", "curl -o", "curl -O"
        ]
        
        for blocked in BLOCKED:
            if blocked in command:
                return {
                    "success": False,
                    "error": f"Write command blocked: '{blocked.strip()}' not allowed in Inspection Agent"
                }
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.work_dir
            )
            
            output = result.stdout + result.stderr
            return {
                "success": result.returncode == 0,
                "output": output[:5000],  # Limit output size
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out (30s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def grep(self, pattern: str, path: str, context_lines: int = 2) -> Dict[str, Any]:
        """
        Search for pattern in file(s).
        """
        import subprocess
        
        try:
            cmd = ["grep", "-n", "-r", f"-C{context_lines}", pattern, path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            matches = []
            for line in result.stdout.split("\n"):
                if line.strip():
                    matches.append(line)
            
            return {
                "success": True,
                "matches": len(matches),
                "results": matches[:50]  # Limit results
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_directory(self, dir_path: str) -> Dict[str, Any]:
        """List contents of a directory."""
        if not os.path.exists(dir_path):
            return {"success": False, "error": f"Directory not found: {dir_path}"}
        
        try:
            items = os.listdir(dir_path)
            files = []
            dirs = []
            
            for item in items:
                full_path = os.path.join(dir_path, item)
                if os.path.isdir(full_path):
                    dirs.append(item)
                else:
                    size = os.path.getsize(full_path)
                    files.append({"name": item, "size": size})
            
            return {
                "success": True,
                "path": dir_path,
                "directories": dirs,
                "files": files
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def render_structure(self, file_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Render structure file to images using OVITO.
        Replaces ASE implementation.
        """
        import os
        try:
            from ovito.io import import_file
            from ovito.vis import Viewport, TachyonRenderer
        except ImportError:
             return {"success": False, "error": "OVITO not installed."}

        if output_dir is None:
            output_dir = os.path.dirname(file_path) or self.work_dir
        
        try:
            # Load Structure
            pipeline = import_file(file_path)
            pipeline.add_to_scene()
            
            basename = os.path.splitext(os.path.basename(file_path))[0]
            images = []
            
            # Define Viewport
            vp = Viewport()
            vp.type = Viewport.Type.Perspective
            vp.fov = 0.6  # Default FOV
            
            # Views with camera directions
            views = {
                "top": (0, 0, -1),      # Top view (looking down Z)
                "front": (0, -1, 0),    # Front view (looking down Y)
                "side": (-1, 0, 0),     # Side view (looking down X)
                "iso": (-1, -1, -1)     # Isometric
            }
            
            for name, direction in views.items():
                out_name = os.path.join(output_dir, f"{basename}_{name}.png")
                
                # Setup Camera
                vp.camera_dir = direction
                vp.zoom_all()
                
                # Render
                # Use TachyonRenderer for better quality (shadows, AO)
                renderer = TachyonRenderer(shadows=False, direct_light_intensity=1.1)
                vp.render_image(filename=out_name, size=(600, 600), renderer=renderer)
                
                images.append(out_name)
            
            pipeline.remove_from_scene()
            
            return {
                "success": True,
                "images": images,
                "source_file": file_path
            }
            
        except Exception as e:
            return {"success": False, "error": f"Rendering failed (OVITO): {str(e)}"}
