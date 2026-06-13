import json
import signal
import sys
import time
import os
from src.llm_client import LLMClient
from src.tools_lib import AgentTools
from src.inspection_client import InspectionClient
from src.planner import TaskPlanner

# Version for sync verification - UPDATE THIS ON EVERY CHANGE
VERSION = "2026.01.22.v38"  # Add: Planning Phase (Plan -> Inspect -> Execute workflow)
DEBUG = True  # Set to False after issues are resolved

class AgentEngine:
    # Token optimization settings (EXPANDED - more context for complex tasks)
    MAX_HISTORY_TURNS = 50  # Keep 50 turns before summarizing (was 25)
    MAX_TOOL_RESULT_CHARS = 2000  # Allow more result chars (was 600)
    
    def __init__(self, work_dir):
        print(f"[Agent] Version: {VERSION}")  # Sync verification
        from src.config import Config
        self.client = LLMClient(
            model_name=Config.SIM_MODEL_NAME,
            api_key=Config.SIM_API_KEY
        )
        self.tools_lib = AgentTools(work_dir, agent_handle=self)
        self.inspector = InspectionClient(work_dir)  # For param validation
        self.planner = TaskPlanner(work_dir, self.client)  # For task planning
        self.log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_trace.log")
        self.history = []
        self.summary = ""  # Running summary of past actions
        self.current_plan = None  # Active task plan
        self._cache = {}  # Cache for repeated queries (e.g., research_crystal)
        
        # Handle Ctrl+C
        signal.signal(signal.SIGINT, self._handle_interrupt)

        # Dynamic Tools Loading
        self.dynamic_handlers = {}
        try:
            # Fetch tools from Inspection Agent
            dynamic_tools, handlers = self.inspector.get_tools_definitions()
            self.dynamic_handlers = handlers
        except Exception as e:
            print(f"[Agent] Warning: Could not load dynamic tools: {e}")
            dynamic_tools = []
            
        # [Fallback] Manual registration of 'consult' if missing
        has_consult = any(t['function']['name'] == 'consult' for t in dynamic_tools)
        if not has_consult:
            print("[Agent] 'consult' tool not found dynamically. Adding fallback registration.")
            consult_tool = {
                "type": "function",
                "function": {
                    "name": "consult",
                    "description": "Conversational interface to Inspection Agent for validation and advice.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "request": {"type": "string", "description": "Natural language request describing what to analyze."},
                            "info": {"type": "object", "description": "Context dictionary with paths/params."},
                            "history": {"type": "string", "description": "Optional history context."}
                        },
                        "required": ["request"]
                    }
                }
            }
            dynamic_tools.append(consult_tool)
            # Add handler to dynamic_handlers
            self.dynamic_handlers["consult"] = lambda args: self.inspector.consult(args.get("request"), args.get("info"), args.get("history"))
        
        # Define Tool Schemas
        self.tools_schema = dynamic_tools + [

            {
                "type": "function",
                "function": {
                    "name": "research_crystal",
                    "description": "Get crystal structure parameters for a material",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "formula": {"type": "string", "description": "Chemical formula e.g. Si, SiO2"}
                        },
                        "required": ["formula"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "research_potential",
                    "description": "Recommend forcefield strategy based on materials. Use bash('ls /path/to/potentials') to see available files, then bash('cp file workdir/') to copy.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sub_elements": {"type": "array", "items": {"type": "string"}},
                            "ion_elements": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["sub_elements", "ion_elements"]
                    }
                }
            },
            # NOTE: list_potential_library and copy_potential_file REMOVED
            # LLM should use bash("ls ...") and bash("cp ...") instead
            {
                "type": "function",
                "function": {
                    "name": "download_potential_file",
                    "description": "Download a force field file from a URL (e.g., GitHub raw URL) to work directory. Use when list_potential_library returns empty and you found a URL via search_web.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Direct URL to the potential file (e.g., https://raw.githubusercontent.com/lammps/lammps/develop/potentials/SiO.tersoff)"},
                            "filename": {"type": "string", "description": "Optional: filename to save as. If not provided, extracts from URL."}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "build_substrate",
                    "description": "Generate substrate structure file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "formula": {"type": "string"},
                            "crystal_params": {"type": "object", "description": "JSON object from research_crystal"},
                            "ion_elements": {"type": "array", "items": {"type": "string"}, "description": "List of ion elements to reserve atom types for (e.g. ['Ar'])"}
                        },
                        "required": ["formula", "crystal_params"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_cif_file",
                    "description": "Find/Download a CIF file for a material from local DB or URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "formula": {"type": "string", "description": "Chemical formula e.g. SiO2"},
                            "url": {"type": "string", "description": "Optional direct URL to raw CIF file (e.g. from COD)"}
                        },
                        "required": ["formula"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "build_structure_from_cif",
                    "description": "Generate substrate from a CIF file (replicates to ~30A + adds vacuum).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cif_filename": {"type": "string", "description": "Filename in Reference/cif_database (e.g. 'SiO2.cif')"},
                            "target_size": {"type": "number", "description": "Target size for replication in Angstroms (default 30.0)"},
                            "ion_elements": {"type": "array", "items": {"type": "string"}, "description": "List of ion elements to reserve atom types for (e.g. ['Ar'])"}
                        },
                        "required": ["cif_filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_projectile",
                    "description": "Generate projectile molecule file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ion_formula": {"type": "string"},
                            "type_map": {"type": "object", "description": "Dict mapping Element string to Integer ID"}
                        },
                        "required": ["ion_formula", "type_map"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_lammps_input",
                    "description": "Write LAMMPS input script for ion bombardment simulation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Output filename e.g. in.sputtering"},
                            "substrate_file": {"type": "string", "description": "Path to substrate .lammps file from build_substrate"},
                            "molecule_file": {"type": "string", "description": "Path to molecule .txt file from create_projectile"},
                            "events": {"type": "integer", "description": "Number of bombardment events (default 10)"},
                            "max_energy": {"type": "number", "description": "Max ion energy in eV (default 100)"},
                            "max_angle": {"type": "number", "description": "Max incident angle in degrees (default 10)"},
                            "potential_commands": {"type": "string", "description": "LAMMPS pair_style commands for potentials"}
                        },
                        "required": ["filename", "substrate_file"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_slurm_script",
                    "description": "Write Slurm queue script",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Output filename e.g. run.slurm"},
                            "job_name": {"type": "string", "description": "Slurm job name"},
                            "input_script": {"type": "string", "description": "LAMMPS input script path"},
                            "nodes": {"type": "integer", "description": "Number of nodes (default 1)"},
                            "ntasks": {"type": "integer", "description": "Number of MPI tasks (default 48)"}
                        },
                        "required": ["filename", "input_script"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_simulation",
                    "description": "Submit simulation to Slurm queue (non-blocking). Returns immediately after job submission. Background monitor will track completion automatically. Do NOT call check_simulation_progress immediately after this.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input_script": {"type": "string", "description": "Path to LAMMPS input script"}
                        },
                        "required": ["input_script"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_slurm_jobs",
                    "description": "Cancel SLURM jobs. If job_id is provided, cancels that specific job. If not provided, cancels all previously submitted jobs by this agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string", "description": "Optional specific job ID to cancel. If omitted, cancels all jobs."}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_slurm_jobs",
                    "description": "Get list of currently tracked SLURM job IDs submitted by this agent",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"}
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_user",
                    "description": "Ask the human user a question for clarification or decision making. Input should be a clear question string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The question to ask the user"}
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_inspector",
                    "description": "Consult the Inspection Agent (Expert) for advice when stuck or for strategic guidance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The question to ask the expert"},
                            "context": {"type": "string", "description": "Optional context about the problem"}
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "consult",
                    "description": "Alias for ask_inspector. Consult the Inspection Agent (Expert).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The question to ask the expert"},
                            "context": {"type": "string", "description": "Optional context about the problem"}
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": "Replace a specific string in a file with a new string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Name of the file to edit"},
                            "search_text": {"type": "string", "description": "Exact text to find and replace"},
                            "replace_text": {"type": "string", "description": "New text to insert"}
                        },
                        "required": ["filename", "search_text", "replace_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_files",
                    "description": "List files by extension",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "extension": {"type": "string"}
                        },
                        "required": ["extension"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "modify_structure_code",
                    "description": "Modify structure.py to fix structure generation for a formula. Use when build_substrate returns structure_errors.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "formula": {"type": "string", "description": "The chemical formula being fixed (e.g., SiO2)"},
                            "old_code": {"type": "string", "description": "Exact code block to replace from structure.py"},
                            "new_code": {"type": "string", "description": "Corrected code block with proper crystal structure"}
                        },
                        "required": ["formula", "old_code", "new_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "modify_lammps_gen_code",
                    "description": "Modify lammps_gen.py to fix LAMMPS input generation issues. Use when generate_lammps_input returns validation errors.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "issue": {"type": "string", "description": "Description of the issue being fixed (e.g., 'mass assignment', 'element ordering')"},
                            "old_code": {"type": "string", "description": "Exact code block to replace from lammps_gen.py"},
                            "new_code": {"type": "string", "description": "Corrected code block with proper logic"}
                        },
                        "required": ["issue", "old_code", "new_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "modify_source_code",
                    "description": "Modify ANY Python file in src/ directory to fix code issues. Use this for self-correction when errors occur in agent code (tools_lib.py, agent_core.py, etc.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Target file basename in src/ (e.g., 'tools_lib.py', 'structure.py')"},
                            "reason": {"type": "string", "description": "Why this change is needed"},
                            "old_code": {"type": "string", "description": "Exact code block to replace (copy exactly from file)"},
                            "new_code": {"type": "string", "description": "New corrected code block"}
                        },
                        "required": ["filename", "reason", "old_code", "new_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "request_review",
                    "description": "Ask Inspection Agent to review input file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "params": {"type": "object", "description": "Optional context parameters"}
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information. Use this to find latest research, tutorials, or documentation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_papers",
                    "description": "Search academic papers (Semantic Scholar). Use this to find force field parameters, simulation methods, or material properties from literature.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query for academic papers"},
                            "limit": {"type": "integer", "description": "Number of results (default 5)"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_simulation_progress",
                    "description": "Inspect simulation status. Returns 'COMPLETE', 'RUNNING', or 'INCOMPLETE'. Uses log tail check and Slurm Job ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "context_description": {"type": "string", "description": "Why you are checking (e.g. 'Turn 3 verification')"},
                            "job_id": {"type": "string", "description": "Slurm Job ID from run_simulation output"}
                        },
                        "required": ["context_description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_user",
                    "description": "Ask the human user for input or decision. Use this when you need guidance, approval, or the user needs to provide information you cannot determine automatically.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The question to ask the user"}
                        },
                        "required": ["question"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a shell command. Useful for file operations, running scripts, listing directories. Security: dangerous commands are blocked.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to execute"},
                            "description": {"type": "string", "description": "Optional description of what the command does"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "Search for a pattern in files. Returns matching lines with context. Searches .py, .j2, .txt files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
                            "path": {"type": "string", "description": "File or directory path to search"},
                            "context_lines": {"type": "integer", "description": "Lines of context before/after match (default 2)"}
                        },
                        "required": ["pattern", "path"]
                    }
                }
            }
        ]
    
    def _truncate_tool_result(self, result):
        """Truncate long tool results smartly - preserve paths and key info."""
        result_str = str(result)
        
        if len(result_str) <= self.MAX_TOOL_RESULT_CHARS:
            return result_str
        
        # Smart truncation: keep first 200 chars (often contains paths/status)
        # and last 100 chars (often contains conclusions/errors)
        head = result_str[:200]
        tail = result_str[-100:]
        
        # Extract any file paths for preservation
        import re
        paths = re.findall(r'[/\\][\w./_-]+\.(lammps|txt|data|log|dump|py|in)', result_str)
        path_info = f" [Paths: {', '.join(paths[:3])}]" if paths else ""
        
        return f"{head}...[TRUNCATED]{path_info}...{tail}"
    
    def _summarize_history(self):
        """Summarize old history to save tokens while preserving critical context."""
        if len(self.history) <= self.MAX_HISTORY_TURNS + 2:  # +2 for system+user
            return
        
        # Extract actions from old history to summarize
        old_messages = self.history[2:-self.MAX_HISTORY_TURNS]
        if not old_messages:
            return
            
        actions = []
        errors_found = []  # Track errors to prevent loops
        for msg in old_messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func_name = tc["function"]["name"]
                    args_str = str(tc["function"].get("arguments", ""))[:50]
                    actions.append(f"- {func_name}({args_str}...)")
            elif msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                # Preserve error information
                if "error" in content.lower() or "failed" in content.lower():
                    errors_found.append(content[:200])
                actions.append(f"  → {content[:150]}")
        
        # Harvest Milestones from old messages
        for i, msg in enumerate(old_messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    # Check result in next message
                    result_content = ""
                    if i + 1 < len(old_messages) and old_messages[i+1].get("role") == "tool":
                        result_content = str(old_messages[i+1].get("content", "")).lower()
                    
                    if "error" in result_content or "failed" in result_content:
                        continue
                    
                    # Track Critical Steps
                    args = str(tc["function"].get("arguments", ""))
                    if name == "fetch_cif_file":
                        self.milestones.append(f"[✓] CIF Acquired")
                    elif name in ["build_structure_from_cif", "build_substrate"]:
                        self.milestones.append(f"[✓] Structure Built ({name})")
                    elif name == "generate_lammps_input":
                        self.milestones.append(f"[✓] LAMMPS Input Ready")
                    elif name == "generate_slurm_script":
                        self.milestones.append(f"[✓] Slurm Script Ready")
                    elif name == "create_projectile":
                        self.milestones.append(f"[✓] Projectile Created")
                    elif name == "run_simulation":
                        self.milestones.append(f"[✓] Simulation Submitted")
        
        # Deduplicate
        self.milestones = list(dict.fromkeys(self.milestones))
        
        if actions:
            # Keep more actions (15 instead of 8)
            new_summary = "\n".join(actions[-15:])
            error_section = ""
            if errors_found:
                error_section = "\n\nPREVIOUS ERRORS (DO NOT REPEAT):\n" + "\n".join(errors_found[-3:])
            
            milestone_section = ""
            if self.milestones:
                milestone_section = "COMPLETED MILESTONES:\n" + "\n".join(self.milestones) + "\n\n"
                
            self.summary = f"{milestone_section}Previous actions:\n{new_summary}{error_section}"
        
        # Rebuild history: system, user, summary, recent messages
        system_msg = self.history[0]
        user_msg = self.history[1]
        recent = self.history[-self.MAX_HISTORY_TURNS:]
        
        summary_msg = {"role": "user", "content": f"[CONTEXT]\n{self.summary}"}
        self.history = [system_msg, user_msg, summary_msg] + recent
        
        print(f"   [Agent] History summarized. Now {len(self.history)} messages.")
    
    def get_history_summary(self) -> str:
        """
        Generate history summary for Inspection Agent context.
        Provides user goal + recent actions for informed validation.
        """
        parts = []
        
        # 1. User Goal (from first user message)
        if len(self.history) > 1:
            user_msg = self.history[1]
            goal = user_msg.get("content", "")[:300]
            parts.append(f"USER GOAL:\n{goal}")
        
        # 2. Recent Tool Calls (last 8 actions with results)
        actions = []
        for msg in self.history[-16:]:  # Look at last 16 messages
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"].get("arguments", "{}"))
                        # Summarize key args
                        key_args = {k: str(v)[:50] for k, v in list(args.items())[:3]}
                        actions.append(f"- {name}({key_args})")
                    except:
                        actions.append(f"- {name}(...)")
            elif msg.get("role") == "tool":
                content = str(msg.get("content", ""))[:100]
                if actions:
                    actions[-1] += f" → {content}"
        
        if actions:
            parts.append("RECENT ACTIONS:\n" + "\n".join(actions[-8:]))
        
        # 3. Current summary if exists
        if self.summary:
            parts.append(f"CONTEXT SUMMARY:\n{self.summary[:300]}")
        
        return "\n\n".join(parts) if parts else "No history available."

    def run(self, user_goal):
        """Main Loop with Planning Phase"""
        print(f"\n[Agent] Starting Mission: {user_goal}")
        
        # === DIRECT EXECUTION (No PLanning Phase) ===
        print("\n" + "="*50)
        print("[PHASE 1] EXECUTION (Standard Workflow)")
        print("="*50)

        # Define Workflow Guide (Standard Procedure)
        workflow_guide = """
WORKFLOW:
1. Research material (Crystal Structure)
   - CRITICAL: You MUST use 'fetch_cif_file(formula)' to get the structure from the database.
   - DO NOT use 'research_crystal' unless 'fetch_cif_file' completely fails (returns error).
   - After getting CIF, use 'build_structure_from_cif'.
2. Create Projectile (if not exists):
   - Use 'create_projectile(ion_formula, type_map)' to generate the molecule file.
3. Prepare Force Field:
   - Check available potentials: bash("ls {ff_library_path}")
   - Strategies:
     A) COPY EXISTING: If you find a suitable file (e.g. Si.tersoff) in the library, COPY it to ./ using bash("cp ..."). 
        IF YOU COPY A FILE, YOU DO NOT NEED TO CALL 'research_potential'. PROCEED DIRECTLY TO STEP 4.
     B) RESEARCH: If no file exists, use 'research_potential' to find parameters.
4. Generate LAMMPS input → 'generate_lammps_input'
   - If you have Si.tersoff and ZBL is needed, use 'potential_style': 'hybrid/overlay tersoff zbl'.
5. Inspect input before running.
6. If uncertain about force field or parameters, use search_papers or search_web first.
7. Collaboration: You can use 'ask_user' ANYTIME you are unsure, need a decision, or feel stuck. Do not waste time in repeated failure loops.
"""
        
        # Use standard workflow as plan context
        plan_context = "Follow the Standard Workflow defined below."
        
        
        system_msg = {
            "role": "system",
            "content": """You are an Autonomous MD Agent using LAMMPS.

CURRENT EXECUTION PLAN:
{plan_context}

IMPORTANT: Follow the plan steps in order. Mark each step as you complete it.
            
IMPORTANT: When you see 'Observation:', it means a tool has completed. DO NOT REPEAT the same tool call. Analyze the observation and proceed to the next step.

CODEBASE MAP (Use these paths for reading/modifying code):
{codebase_map}

RESOURCE PATHS (Use bash to explore and copy files):
- Force Field Library: {ff_library_path}
  → Use: bash("ls {ff_library_path}") to list available potentials
  → Use: bash("cp {ff_library_path}/Si.tersoff ./") to copy to work directory
- CIF Database: {cif_database_path}
  → Use: bash("ls {cif_database_path}") to list available CIF files
  → Or use: fetch_cif_file(formula) to download from Materials Project

{workflow_guide}

CRITICAL RULE - TEMPLATE PROTECTION:
- You may ONLY pass parameters to generate_lammps_input that are defined as template variables.
- Do NOT use apply_patch or write_file to modify timestep, neighbor settings, thermo settings, or other hardcoded template values.
- Only 'potential_commands' allows multi-line custom code.

MANDATORY WORKFLOW RULES:
1. USE PROVIDED TOOLS: DO NOT create separate Python scripts (e.g., build_sio2.py, run_md.py) for tasks that have dedicated tools.
   - Use 'build_substrate' tool, not 'write_file(build.py)'.
   - Use 'run_simulation' tool, not 'subprocess.run(...)'.
2. DO NOT POLL: If a command succeeds silently (like 'cp' or 'mv'), assume success and proceed. Do NOT repeated verify unless an error occurred.

MANDATORY ERROR HANDLING:
1. If build_substrate returns 'structure_errors', you MUST call modify_structure_code() to fix it before proceeding. Do NOT continue with a broken structure.
2. If generate_lammps_input returns 'validation': 'FAILED', you MUST call modify_lammps_gen_code() to fix it before run_simulation.
3. SIMULATION MONITORING:
   - run_simulation() with Slurm returns IMMEDIATELY after job submission (non-blocking).
   - A background monitor process will track the simulation automatically.
   - DO NOT call check_simulation_progress() immediately after run_simulation().
   - Only check progress if the user asks, or after sufficient time has passed (e.g., 10+ minutes for long simulations).
   - When the simulation completes, check log.lammps for errors before declaring DONE.


POTENTIAL STRATEGY FOR SPUTTERING:
- Substrate: Use Tersoff or Vashishta for covalent materials (SiO, SiC).
- Ion-Substrate collision: Always include ZBL via hybrid/overlay for high-energy impacts.
- For complex systems, prefer hybrid/overlay rather than ReaxFF unless user explicitly requests it.

COLLABORATION:
- You have access to an Inspection Agent (via A2A protocol) for validation and review.
- Use 'validate_*' tools to check your files BEFORE running simulations.
- Use 'review_plan' to get feedback on your logic.

DEVELOPMENT MODE - SELF-MODIFICATION RULE ("Read Before Write"):
⚠️ This agent is under active development. You can fix bugs in your own source code!

CRITICAL SAFETY PROTOCOL:
1. BEFORE calling 'modify_source_code', you MUST call 'read_file(filename)' to see the ACTUAL CONTENT.
2. DO NOT GUESS code content. If you guess, the 'old_code' will not match and the patch will FAIL.
3. If you encounter an error, read the file first.
4. If a tool seems broken, read its implementation in 'tools_lib.py' before trying to fix it.

AVAILABLE FILES TO MODIFY: structure.py, lammps_gen.py, tools_lib.py, agent_core.py, and all .py files in src/

WHEN TO ESCALATE TO USER:
- Missing information that cannot be inferred (e.g., user preference for force field)
- API key issues or authentication errors
- After 3 failed attempts to fix an issue yourself
- Use 'ask_user(question)' to request guidance

LOOP DETECTION:
If you've called the same tool 3+ times with similar arguments, STOP and either:
1. Fix the underlying code with 'modify_source_code()'
2. Ask user with 'ask_user()'

AVAILABLE TOOLS:
{tool_names}"""
        }
        
        # Inject dynamic paths and tool names
        src_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(src_dir)  # md_agent root
        
        codebase_map = f"""- structure.py: {os.path.join(src_dir, 'structure.py')}
- lammps_gen.py: {os.path.join(src_dir, 'lammps_gen.py')}
- tools_lib.py: {os.path.join(src_dir, 'tools_lib.py')}
- agent_core.py: {os.path.abspath(__file__)}"""
        
        # Resource paths for LLM to use with bash
        ff_library_path = os.path.join(base_dir, "Reference", "force_field_library", "potentials")
        cif_database_path = os.path.join(base_dir, "Reference", "cif_database")
        
        tool_names = ", ".join([t["function"]["name"] for t in self.tools_schema])
        system_msg["content"] = system_msg["content"].format(
            tool_names=tool_names,
            codebase_map=codebase_map,
            ff_library_path=ff_library_path,
            cif_database_path=cif_database_path,
            plan_context=plan_context,
            workflow_guide=workflow_guide
        )
        
        self.history = [system_msg, {"role": "user", "content": user_goal}]

        # Reflection tracking
        self._consecutive_failures = 0
        self._last_tool_name = None
        self._repeated_tool_count = 0
        self._total_stuck_count = 0  # Track total stuck detections
        MAX_STUCK_RETRIES = 2  # After 2 expert consultations fail, ask user immediately
        self.milestones = []  # Persistent list of completed major steps

        max_turns = 150  # Increased for complex simulations
        for turn in range(max_turns):
            print(f"\n[Turn {turn+1}] Thinking...")
            
            # Summarize history if too long
            self._summarize_history()
            
            # REFLECTION CHECK: Consult Inspection Agent if agent appears stuck
            if self._consecutive_failures >= 3 or self._repeated_tool_count >= 3:
                self._total_stuck_count += 1
                
                # After too many stuck detections, ask user for guidance
                if self._total_stuck_count >= MAX_STUCK_RETRIES:
                    print(f"[Agent] Loop detected ({self._total_stuck_count}x). Suggesting 'ask_user' to Agent...")
                    
                    # Inject system alert to guide LLM
                    intervention_msg = {
                        "role": "user",
                        "content": (
                            f"SYSTEM ALERT: Repeated pattern detected. You have performed the same action or failed {self._total_stuck_count} times in a row. "
                            f"The last tool tried was '{self._last_tool_name}'. "
                            "It seems you are stuck in a loop. "
                            "Please STOP repeating the same action. "
                            "If you are unsure how to proceed, use the 'ask_user' tool to ask for human guidance. "
                            "Explain what you are trying to do and where you are stuck."
                        )
                    }
                    self.history.append(intervention_msg)
                    self._total_stuck_count = 0
                    self._consecutive_failures = 0
                    self._repeated_tool_count = 0
                    continue
                
                print("[Agent] Stuck detected - consulting Inspection Agent...")
                
                # Build context from recent history
                recent_context = []
                for msg in self.history[-6:]:
                    if msg.get("role") == "tool":
                        content = str(msg.get("content", ""))[:200]
                        recent_context.append(f"Tool Result: {content}")
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            recent_context.append(f"Called: {tc['function']['name']}")
                
                problem = f"{'Consecutive failures' if self._consecutive_failures >= 3 else 'Repeated same tool'}: Last tool was {self._last_tool_name}"
                context_str = "\n".join(recent_context[-5:])  # Last 5 actions
                
                try:
                    # Call Inspection Agent for advice (Conversational)
                    consult_req = f"I am stuck. Problem: {problem}. Context: {context_str}. Please diagnose and advise on what to do next."
                    advice_result = self.tools_lib._consult_inspector(
                        request=consult_req,
                        info={"problem": problem, "recent_actions": recent_context}
                    )
                    
                    # Build reflection message with expert advice
                    diagnosis = advice_result.get("analysis", "Check logs for details")
                    advice = advice_result.get("response", "Try a different approach")
                    # Check if advice suggests asking user
                    ask_user = "ask_user" in str(advice).lower() or "human" in str(advice).lower()
                    
                    reflection_content = (
                        f"[EXPERT CONSULTATION]\n"
                        f"Inspection Agent analyzed your situation:\n\n"
                        f"**Diagnosis**: {diagnosis}\n"
                        f"**Advice**: {advice}\n"
                    )
                    if ask_user:
                        reflection_content += "\n⚠️ Consider using ask_user() to get human guidance."
                    
                    print(f"[Agent] Expert advice received: {advice[:100]}...")
                    
                except Exception as e:
                    print(f"[Agent] Consultation failed: {e}. Using fallback.")
                    reflection_content = (
                        "[SYSTEM REFLECTION]\n"
                        f"You have {'failed' if self._consecutive_failures >= 3 else 'repeated the same action'} multiple times. "
                        "STOP and analyze:\n"
                        "1. What exactly is going wrong?\n"
                        "2. Are you searching in the wrong place?\n"
                        "3. Should you try a completely different approach?\n"
                        "4. Do you need to ask the user (ask_user) for help?\n"
                        "Take a moment to think before your next action."
                    )
                
                reflection_msg = {"role": "user", "content": reflection_content}
                self.history.append(reflection_msg)
                
                # Reset counters after injection
                self._consecutive_failures = 0
                self._repeated_tool_count = 0
            
            # Call LLM
            response_msg = self.client.generate_response(self.history, tools=self.tools_schema)
            
            if not response_msg:
                print("[Agent] LLM Error. Stopping.")
                self.save_history()
                return

            # Append Assistant Response
            self.history.append(response_msg)
            
            # Check for generic content
            if response_msg.get("content"):
                thought = response_msg['content']
                print(f"[AI Thought] {thought}", flush=True)
                with open(self.log_path, "a", encoding="utf-8") as log:
                    log.write(f"\n[AI Thought]: {thought}\n")
                
            # Check for Tool Calls
            tool_calls = response_msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    args_str = tc["function"].get("arguments", "{}")
                    if args_str is None: args_str = "{}"
                    
                    # Safe ID retrieval (Ollama often omits 'id')
                    import uuid
                    if "id" not in tc:
                        # INJECT the ID back into the object so history remains consistent
                        tc["id"] = f"call_{func_name}_{uuid.uuid4().hex[:8]}"
                    call_id = tc["id"]
                    
                    # Parse arguments once
                    try:
                        if isinstance(args_str, dict):
                            args = args_str
                            args_display = json.dumps(args, indent=2)
                        else:
                            args = json.loads(args_str)
                            args_display = json.dumps(args, indent=2)
                    except:
                        args = {}
                        args_display = str(args_str)

                    # Log Call (Verbose)
                    print(f"   [Tool Call] {func_name}\nArgs: {args_display}", flush=True)
                    with open(self.log_path, "a", encoding="utf-8") as log:
                        log.write(f"[Tool Call]: {func_name}\nArgs: {args_display}\n")
                    
                    try:
                        # Validate critical tools before execution
                        critical_tools = ["generate_lammps_input", "run_simulation"]
                        if func_name in critical_tools:
                            # Use absolute path to agent_trace.log in md_agent folder
                            validation = self.inspector.validate_params(func_name, args, self.log_path)
                            
                            # Handle case where MCP returns error string instead of dict
                            if isinstance(validation, str):
                                print(f"[Agent] Validation skipped (MCP error): {validation[:100]}")
                                result = self._execute_tool(func_name, args)
                            elif not validation.get("valid", True):
                                # Return validation error to LLM instead of executing
                                result = f"VALIDATION ERROR: {validation.get('suggestion', 'Unknown validation error')}"
                            else:
                                result = self._execute_tool(func_name, args)
                        else:
                            result = self._execute_tool(func_name, args)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        result = f"Error executing tool: {str(e)}"
                    
                    # REFLECTION TRACKING: Detect failures and repeats
                    result_str = str(result).lower()
                    is_failure = any(kw in result_str for kw in ["error", "failed", "not found", "missing", "invalid"])
                    
                    if is_failure:
                        self._consecutive_failures += 1
                    else:
                        self._consecutive_failures = 0
                    
                    if func_name == self._last_tool_name:
                        self._repeated_tool_count += 1
                    else:
                        self._repeated_tool_count = 0
                        self._last_tool_name = func_name
                    # Truncate result to save tokens (but show more in terminal)
                    # For terminal, show up to 2000 chars
                    str_result = str(result)
                    print(f"   [Tool Result] {str_result[:2000]}...", flush=True)
                    if len(str_result) > 2000:
                        print(f"   ... (truncated {len(str_result)-2000} chars) ...")

                    # For LLM context, still truncate to avoid overflow
                    truncated_result = self._truncate_tool_result(result)
                    
                    with open(self.log_path, "a", encoding="utf-8") as log:
                        log.write(f"[Tool Result]: {str_result}\n")
                    
                    # Feed back to LLM (truncated)
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": truncated_result
                    })
                    
                    # [System Override] Check for Simulation Success and Exit Immediately
                    # STRICTER CHECK: Only trigger if the tool was actually checking progress or running sim
                    # Prevents false positives when reading source code containing the string "Total wall time"
                    is_completion_check = func_name in ["check_simulation_progress", "run_simulation"]
                    
                    if is_completion_check and ("Total wall time" in str_result or "Status: COMPLETE" in str_result):
                        print(f"\n" + "="*50)
                        print(f">>> [MISSION COMPLETE] Simulation finished successfully.")
                        print(f"    Automated shutdown triggered.")
                        print(f"="*50 + "\n")
                        
                        with open(self.log_path, "a", encoding="utf-8") as log:
                            log.write(f"\n[System]: Simulation success detected. Mission Complete.\n")
                        self.save_history()
                        return # Exit run() completely

                    # [Constraint] Execute only ONE tool per turn
                    break
            else:
                # No tool calls - check if done
                if "DONE" in str(response_msg.get("content")):
                    with open(self.log_path, "a", encoding="utf-8") as log:
                        log.write(f"\n[Agent]: Job Done.\n")
                    self.save_history()
                    break


    def _execute_tool(self, tool_name, args):
        """Execute a tool by name."""
        try:
            # Check AgentTools
            if hasattr(self.tools_lib, tool_name):
                func = getattr(self.tools_lib, tool_name)
                return func(**args)
            
            return f"Error: Tool '{tool_name}' not found."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error executing '{tool_name}': {e}"

    def _handle_interrupt(self, signum, frame):
        print("\n[Agent] Interrupted! Saving history...")
        self.save_history()
        
        # Auto-cleanup Slurm jobs
        print("[Agent] Cancelling running Slurm jobs...")
        try:
            if hasattr(self, 'tools_lib'):
                self.tools_lib.cancel_slurm_jobs()
        except Exception as e:
            print(f"[Agent] Failed to cancel jobs: {e}")
            
        sys.exit(0)

    def save_history(self):
        """Save conversation history to JSON file."""
        import json
        import datetime
        
        # Create logs directory if not exists
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"history_{timestamp}.json"
        filepath = os.path.join(log_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            print(f"\n[Agent] History saved to: {filepath}")
        except Exception as e:
            print(f"\n[Agent] Failed to save history: {e}")

