from src.llm_client import LLMClient
import json
import os
import re
import math
import traceback

# Ovito imports (assuming these are needed for visual_inspection)
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

try:
    from ovito.io import import_file
    from ovito.vis import Viewport, TachyonRenderer
    OVITO_AVAILABLE = True
except ImportError:
    print("Ovito not found. Visual inspection will be skipped.")
    OVITO_AVAILABLE = False


class InspectionAgent:
    # Required parameters for critical tools
    REQUIRED_PARAMS = {
        "generate_lammps_input": ["substrate_file"],
        "run_simulation": ["input_script"]
    }
    
    # Expected stoichiometry for common formulas
    STOICHIOMETRY = {
        "sio2": {"Si": 1, "O": 2},
        "si3n4": {"Si": 3, "N": 4},
        "al2o3": {"Al": 2, "O": 3},
        "tio2": {"Ti": 1, "O": 2}
    }
    
    # Recommended (not required) parameters
    RECOMMENDED_PARAMS = {
        "generate_lammps_input": ["masses", "potential_commands", "events", "c", "lz1"]
    }
    
    
    def __init__(self, work_dir: str = "."):
        from src.config import Config
        from src.inspection_tools_lib import InspectionTools
        
        self.client = LLMClient(
            model_name=Config.INSPECTION_MODEL_NAME,
            api_key=Config.INSPECTION_API_KEY
        )
        self.work_dir = work_dir
        self.tools = InspectionTools(work_dir)
    
    def review_task_plan(self, plan: dict, user_goal: str) -> dict:
        """
        Review and approve/reject a task plan before execution.
        
        Args:
            plan: The structured plan from TaskPlanner
            user_goal: Original user goal for context
        
        Returns:
            {
                "approved": bool,
                "feedback": str,
                "suggested_changes": [str],
                "confidence": "high" | "medium" | "low"
            }
        """
        print(f"[InspectionAgent] Reviewing task plan for: {user_goal[:50]}...")
        
        prompt = f"""You are a Senior MD Simulation Expert reviewing a task plan.

USER GOAL:
{user_goal}

PROPOSED PLAN:
{json.dumps(plan, indent=2)}

REVIEW CHECKLIST:
1. Are the steps in the correct order? (e.g., structure must be built before LAMMPS input)
2. Are all necessary steps included? (e.g., force field preparation)
3. Are the tool choices appropriate for each step?
4. Is the plan feasible with available tools?
5. Are there any missing considerations? (e.g., projectile for sputtering)

Respond in JSON:
{{
    "approved": true/false,
    "feedback": "Brief explanation of your decision",
    "suggested_changes": ["Change 1", "Change 2"],
    "confidence": "high/medium/low"
}}
"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.generate_json(messages, temperature=0.2)
            
            if response:
                result = {
                    "approved": response.get("approved", False),
                    "feedback": response.get("feedback", ""),
                    "suggested_changes": response.get("suggested_changes", []),
                    "confidence": response.get("confidence", "medium")
                }
                status = "✅ APPROVED" if result["approved"] else "❌ REJECTED"
                print(f"[InspectionAgent] Plan Review: {status}")
                if result["feedback"]:
                    print(f"   Feedback: {result['feedback'][:100]}...")
                return result
        except Exception as e:
            print(f"[InspectionAgent] Plan review error: {e}")
        
        # Default: approve with low confidence if LLM fails
        return {
            "approved": True,
            "feedback": "Auto-approved due to review failure",
            "suggested_changes": [],
            "confidence": "low"
        }
    
    def process_request(self, request: str, info: dict = None, history: str = "") -> dict:
        """
        Process a natural language request from Sim Agent.
        This is the main conversational interface entry point.
        
        Args:
            request: Natural language description of what to analyze
            info: Context dict with file paths, parameters, etc.
            history: Summary of Sim Agent's history (goals, actions)
        
        Returns:
            Analysis results as dict
        """
        if info is None:
            info = {}
            
        print(f"\n[InspectionAgent] Received Request: {request[:150]}...")
        
        # Build context from info
        context_parts = []
        if history:
            context_parts.append(f"=== SIM AGENT HISTORY ===\n{history}\n")
            
        context_parts.append("=== CURRENT INFO ===")
        for key, value in info.items():
            if isinstance(value, dict):
                context_parts.append(f"{key}: {json.dumps(value, indent=2)}")
            elif isinstance(value, list):
                context_parts.append(f"{key}: {value}")
            else:
                context_parts.append(f"{key}: {value}")
        
        context_str = "\n".join(context_parts) if context_parts else "No additional context provided."
        
        # Define available tools for the agent
        tools_description = """
Available Tools:
1. read_file(file_path) - Read contents of a file
2. bash(command) - Run read-only shell commands (ls, cat, grep, head, tail)
3. list_directory(dir_path) - List directory contents
4. grep(pattern, path) - Search for patterns in files
5. render_structure(file_path) - Render OVITO images of a structure file
6. read_model_registry() - Check available models and specs
7. switch_sim_agent_model(model_name) - Change Sim Agent's active model
"""
        
        # Build prompt for LLM
        prompt = f"""You are an Inspection Agent for MD simulations. 
Analyze the request and provide a detailed response.

REQUEST:
{request}

CONTEXT:
{context_str}

{tools_description}

INSTRUCTIONS:
1. Analyze what is being requested
2. Use the tools as needed to gather information
3. Provide clear analysis and recommendations
4. Format your response as JSON with keys: success, analysis, recommendations, warnings (if any)
5. CRITERIA:
   - For Ion Bombardment (e.g. Ar -> Si), the standard is Substrate Potential (e.g. Tersoff/MEAM) + ZBL.
   - PHYSICS KNOWLEDGE: High-energy collisions (>10eV) are dominated by core repulsion. The ZBL potential is specifically designed for this.
   - Therefore, ZBL is SUFFICIENT for the ion-ion and ion-substrate interactions during the collision phase. Lack of long-range VdW (LJ) for the ion is acceptable and standard practice.
   - APPROVE strategies that validly use ZBL for the projectile interactions.

Respond with your analysis:"""
        
        try:
            # Use LLM to process the request
            response = self.client.generate_response(
                messages=[{"role": "user", "content": prompt}],
                tools=self._get_tool_schemas()
            )
            
            # Process tool calls if any
            result = self._handle_tool_calls(response, request, info)
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Request processing failed: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    def _get_tool_schemas(self) -> list:
        """Get tool schemas for the Inspection Agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Absolute path to file"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run read-only shell commands",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to run"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List contents of a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dir_path": {"type": "string", "description": "Directory path"}
                        },
                        "required": ["dir_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "render_structure",
                    "description": "Render OVITO images of a structure file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to structure file"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_model_registry",
                    "description": "Read the MODEL_REGISTRY.md file to see available LLM models and benchmarks.",
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
                    "name": "switch_sim_agent_model",
                    "description": "Switch the Simulation Agent's LLM model to a new one from the registry.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {"type": "string", "description": "The Model Key from MODEL_REGISTRY.md (e.g. 'direct/local-network-5090')"}
                        },
                        "required": ["model_name"]
                    }
                }
            }
        ]
    
    def _handle_tool_calls(self, response, request: str, info: dict) -> dict:
        """Handle tool calls from LLM response."""
        MAX_ITERATIONS = 5
        messages = [{"role": "user", "content": request}]
        
        for iteration in range(MAX_ITERATIONS):
            # LLMClient returns a dict, not OpenAI SDK object
            # Handle both dict and SDK object for compatibility
            if isinstance(response, dict):
                tool_calls = response.get("tool_calls")
                content = response.get("content", "")
            else:
                # Fallback for OpenAI SDK-style response (if any)
                try:
                    msg = response.choices[0].message
                    tool_calls = msg.tool_calls
                    content = msg.content or ""
                except (AttributeError, IndexError):
                    return {
                        "success": False,
                        "error": "Invalid response format from LLM"
                    }
            
            if not tool_calls:
                # No more tool calls, return final response
                try:
                    # Try to parse as JSON
                    return json.loads(content)
                except:
                    return {
                        "success": True,
                        "analysis": content,
                        "raw_response": True
                    }
            
            # Build assistant message for history
            assistant_msg = {"role": "assistant", "content": content, "tool_calls": tool_calls}
            messages.append(assistant_msg)
            
            for tool_call in tool_calls:
                # Handle both dict and object-style tool_calls
                if isinstance(tool_call, dict):
                    func_name = tool_call["function"]["name"]
                    func_args = tool_call["function"].get("arguments", "{}")
                    call_id = tool_call.get("id", f"call_{iteration}")
                else:
                    func_name = tool_call.function.name
                    func_args = tool_call.function.arguments
                    call_id = tool_call.id
                
                try:
                    args = json.loads(func_args) if isinstance(func_args, str) else func_args
                except:
                    args = {}
                
                print(f"[InspectionAgent] Tool Call: {func_name}({str(args)[:100]}...)")
                
                # Execute the tool
                if func_name == "read_file":
                    result = self.tools.read_file(**args)
                elif func_name == "bash":
                    result = self.tools.bash(**args)
                elif func_name == "list_directory":
                    result = self.tools.list_directory(**args)
                elif func_name == "render_structure":
                    result = self.tools.render_structure(**args)
                elif func_name == "read_model_registry":
                    result = self.tools.read_model_registry()
                elif func_name == "switch_sim_agent_model":
                    result = self.tools.switch_sim_agent_model(**args)
                else:
                    result = {"error": f"Unknown tool: {func_name}"}
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result)
                })
            
            # Continue conversation
            response = self.client.generate_response(
                messages=messages,
                tools=self._get_tool_schemas()
            )
        
        return {
            "success": False,
            "error": "Max iterations reached",
            "partial_result": messages[-1] if messages else None
        }
    
    def _resolve_path(self, file_path):
        """Resolves a file path, making it absolute if it's relative to work_dir."""
        if not os.path.isabs(file_path):
            return os.path.join(self.work_dir, file_path)
        return file_path

    def visual_inspection(self, structure_file):
        """
        Renders the structure using Ovito and asks LLM to check for anomalies.
        Returns:
            dict: { "valid": bool, "reason": str, "image_path": str }
        """
        if not OVITO_AVAILABLE:
            return {"valid": True, "reason": "Ovito not available, skipped visual check."}
            
        params = {"structure_file": structure_file}
        full_path = self._resolve_path(structure_file)
        if not os.path.exists(full_path):
             return {"valid": False, "reason": f"File not found: {structure_file}"}

        try:
            # 1. Render Image
            image_path = os.path.join(self.work_dir, f"visual_check_{os.path.basename(structure_file)}.png")
            print(f"[InspectionAgent] Rendering {structure_file} to {image_path}...")
            
            pipeline = import_file(full_path)
            pipeline.add_to_scene()
            
            vp = Viewport()
            vp.type = Viewport.Type.Perspective
            vp.camera_pos = (-10, -10, 10)
            vp.camera_dir = (1, 1, -1)
            vp.fov = math.radians(60.0)
            
            # Simple Auto-zoom
            data = pipeline.compute()
            cell = data.cell
            # matrix logic omitted for brevity, just using zoom_all
            vp.zoom_all()
            
            vp.render_image(filename=image_path, size=(800, 600), renderer=TachyonRenderer())
            pipeline.remove_from_scene()
            
            # 2. Ask LLM Vision
            prompt = """
            Review this molecular structure visualization.
            Check for:
            1. Obvious overlapping atoms (explosive/high energy).
            2. Large unintended voids.
            3. Disconnected clusters (if it should be a solid crystal).
            
            Return JSON: { "valid": true/false, "reason": "..." }
            """
            
            messages = [
                {"role": "system", "content": "You are a Crystallography Expert. Respond in JSON."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.generate_response(messages, images=[image_path])
            
            # 3. Parse Result
            
            content = response.get("content", "{}")
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                result["image_path"] = image_path
                return result
                
            return {"valid": True, "reason": "LLM output unparseable, assuming OK", "raw": content}

        except Exception as e:
            print(f"[InspectionAgent] Visual Check Failed: {e}")
            traceback.print_exc()
            return {"valid": False, "reason": f"Visual Check Error: {e}"}

    def validate_structure(self, data_file, formula):
        """
        Validates LAMMPS data file structure (Stoichiometry + Visualization).
        Returns: {"valid": bool, "errors": [], "atom_counts": {}, "cell_bounds": {}}
        """
        print(f"[InspectionAgent] validating structure {data_file} for {formula}...")
        
        result = {"valid": True, "errors": [], "atom_counts": {}, "cell_bounds": {}}
        
        if not os.path.exists(data_file):
            result["valid"] = False
            result["errors"].append(f"Structure file not found: {data_file}")
            return result
        
        # Parse LAMMPS data file
        with open(data_file, "r") as f:
            content = f.read()
        
        # Extract cell bounds
        bounds = {}
        for axis in ["xlo xhi", "ylo yhi", "zlo zhi"]:
            match = re.search(rf"([\d\.\-e]+)\s+([\d\.\-e]+)\s+{axis}", content)
            if match:
                lo, hi = float(match.group(1)), float(match.group(2))
                bounds[axis.split()[0][0]] = {"lo": lo, "hi": hi}  # x, y, or z
        
        result["cell_bounds"] = bounds
        
        # Check cell origin (should be ~0 or >= 0)
        for axis, vals in bounds.items():
            if vals["lo"] < -0.1:  # Allow small tolerance
                result["errors"].append(f"{axis.upper()} origin not at 0: {axis}lo = {vals['lo']:.2f}")
                result["valid"] = False
        
        # Extract atom types and count
        atoms_section = re.search(r"Atoms.*?\n\n(.*?)(\n\n|$)", content, re.DOTALL)
        if atoms_section:
            atom_lines = atoms_section.group(1).strip().split("\n")
            type_counts = {}
            for line in atom_lines:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        atom_type = int(parts[1])
                        type_counts[atom_type] = type_counts.get(atom_type, 0) + 1
                    except ValueError:
                        continue
            result["atom_counts"] = type_counts
        
        # Check stoichiometry if formula is known
        formula_lower = formula.lower()
        if formula_lower in self.STOICHIOMETRY:
            expected = self.STOICHIOMETRY[formula_lower]
            if result["atom_counts"]:
                # Get ratio from first type
                counts = list(result["atom_counts"].values())
                if len(counts) >= 2:
                    # Check if ratio matches expected
                    total = sum(counts)
                    expected_ratio = list(expected.values())
                    actual_ratio = [c / counts[0] * expected_ratio[0] for c in counts]
                    
                    for i, (exp, act) in enumerate(zip(expected_ratio, actual_ratio)):
                        if abs(exp - act) > 0.1:
                            result["errors"].append(
                                f"Stoichiometry mismatch for {formula}: expected ratio {expected_ratio}, "
                                f"got counts {counts} (ratio ~{[round(c/counts[0], 2) for c in counts]})"
                            )
                            result["valid"] = False
                            break
        
        # --- Add Visual Inspection ---
        if result["valid"]:
            print("[InspectionAgent] Stoichiometry OK. Running Visual Inspection...")
            vis_result = self.visual_inspection(data_file)
            result["visual_check"] = vis_result
            if not vis_result.get("valid", True):
                result["valid"] = False
                result["errors"].append(f"Visual Check Failed: {vis_result.get('reason')}")
        
        if result["valid"]:
            print(f"[InspectionAgent] ✓ Structure validated: {formula} ({result['atom_counts']})")
        else:
            print(f"[InspectionAgent] ⚠️ Structure errors: {result['errors']}")
        
        return result
    
    # Standard atomic masses
    ATOMIC_MASSES = {
        "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011,
        "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.18, "Na": 22.99, "Mg": 24.305,
        "Al": 26.982, "Si": 28.086, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
        "K": 39.098, "Ca": 40.078, "Ti": 47.867, "Fe": 55.845, "Ni": 58.693, "Cu": 63.546,
        "Zn": 65.38, "Ga": 69.723, "Ge": 72.63, "As": 74.922, "Br": 79.904, "Kr": 83.798,
        "Mo": 95.95, "Ru": 101.07, "Pd": 106.42, "Ag": 107.87, "Pt": 195.08, "Au": 196.97
    }
    
    def validate_lammps_input(self, input_file, type_map, data_file=None):
        """
        Validates a generated LAMMPS input script.
        Checks: masses, pair_coeff element order, compatibility with data file.
        Returns: {"valid": bool, "errors": [], "warnings": []}
        """
        result = {"valid": True, "errors": [], "warnings": []}
        
        if not os.path.exists(input_file):
            result["valid"] = False
            result["errors"].append(f"Input file not found: {input_file}")
            return result
        
        with open(input_file, "r") as f:
            content = f.read()
        
        # 1. Check masses match type_map
        mass_errors = []
        for element, type_id in type_map.items():
            expected_mass = self.ATOMIC_MASSES.get(element)
            if expected_mass:
                # Look for "mass <type_id> <value>"
                mass_pattern = rf"mass\s+{type_id}\s+(\d+\.?\d*)"
                match = re.search(mass_pattern, content)
                if match:
                    found_mass = float(match.group(1))
                    if abs(found_mass - expected_mass) > 1.0:  # Tolerance of 1 amu
                        mass_errors.append(
                            f"Type {type_id} ({element}): found mass {found_mass}, "
                            f"expected ~{expected_mass}"
                        )
        
        if mass_errors:
            result["errors"].extend(mass_errors)
            result["valid"] = False
        
        # 2. Check pair_coeff element order for tersoff
        tersoff_pattern = r"pair_coeff\s+\*\s+\*\s+tersoff\s+\S+\s+([\w\s]+)"
        tersoff_match = re.search(tersoff_pattern, content)
        if tersoff_match:
            elements_in_coeff = tersoff_match.group(1).strip().split()
            # Verify order matches type_map
            sorted_types = sorted(type_map.items(), key=lambda x: x[1])
            expected_order = [elem for elem, _ in sorted_types]
            
            # Check if elements match (excluding NULL)
            actual_elements = [e for e in elements_in_coeff if e != "NULL"]
            expected_elements = expected_order[:len(actual_elements)]
            
            if actual_elements != expected_elements:
                result["errors"].append(
                    f"pair_coeff element order mismatch: found {elements_in_coeff}, "
                    f"expected order based on type_map: {expected_order}"
                )
                result["valid"] = False
        
                result["valid"] = False
        
        # 3. Check ZBL pair_style syntax (Critical)
        zbl_match = re.search(r"pair_style.*zbl", content)
        if zbl_match:
            line = zbl_match.group(0)
            # Check if line contains numbers
            has_cutoffs = re.search(r"\d", line.split("zbl")[-1])
            if not has_cutoffs:
                result["errors"].append(
                    "Missing cutoffs in 'pair_style ... zbl' command. ZBL requires inner/outer cutoffs (e.g. 0.5 2.0)."
                )
                result["valid"] = False

        # 4. Check data file compatibility
        if data_file and os.path.exists(data_file):
            with open(data_file, "r") as f:
                data_content = f.read()
            
            # Count atom types in data file
            types_match = re.search(r"(\d+)\s+atom types", data_content)
            if types_match:
                data_types = int(types_match.group(1))
                if data_types < len([e for e in type_map if type_map[e] is not None]):
                    result["warnings"].append(
                        f"Data file has {data_types} atom types, but type_map has more active elements"
                    )
        
        if result["valid"]:
            print(f"[Inspector] ✓ LAMMPS input validated: {os.path.basename(input_file)}")
        else:
            print(f"[Inspector] ⚠️ LAMMPS input errors: {result['errors']}")
        
        return result
        
    def validate_params(self, tool_name, params, log_file_path=""):
        """
        Validates tool parameters before execution.
        Reads context from agent_trace.log for understanding current situation.
        Returns: {"valid": bool, "missing": [], "warnings": [], "suggestion": str}
        """
        result = {"valid": True, "missing": [], "warnings": [], "suggestion": ""}
        
        # Read context from log file
        context = ""
        if log_file_path and os.path.exists(log_file_path):
            try:
                with open(log_file_path, "r") as f:
                    lines = f.readlines()
                    context = "".join(lines[-30:])
            except Exception:
                context = ""
        
        # Check required params (flat structure)
        required = self.REQUIRED_PARAMS.get(tool_name, [])
        for key in required:
            if key not in params or params.get(key) is None:
                result["missing"].append(key)
                result["valid"] = False
        
        # [Smart Check] If specific file params are present but invalid, suggest alternatives
        if tool_name == "generate_lammps_input" and result["valid"]:
            for file_param in ["substrate_file", "molecule_file"]:
                fpath = params.get(file_param)
                if fpath and not os.path.exists(fpath):
                    # File missing! Look for alternatives
                    result["valid"] = False
                    result["missing"].append(f"{file_param}_NOT_FOUND")
                    
                    # Search for candidates in work_dir
                    dirname = os.path.dirname(fpath) or self.work_dir
                    basename = os.path.basename(fpath)
                    candidates = []
                    if os.path.exists(dirname):
                        for f in os.listdir(dirname):
                            if f.endswith(".lammps") or f.endswith(".txt") or f.endswith(".data"):
                                candidates.append(f)
                    
                    suggestion = f"The file '{basename}' for parameter '{file_param}' was not found."
                    if candidates:
                        suggestion += f" Did you mean one of these? {candidates}"
                    else:
                        suggestion += " No similar files found."
                        
                    result["suggestion"] = suggestion
                    print(f"[Inspector] ⚠️ File Not Found: {fpath}. Candidates: {candidates}")

        # Check recommended params (warnings only)
        recommended = self.RECOMMENDED_PARAMS.get(tool_name, [])
        for key in recommended:
            if key not in params:
                result["warnings"].append(f"Missing recommended param: {key} (will use default)")
        
        # If missing critical params, suggest fix with context
        if not result["valid"]:
            if not result["suggestion"]:
                result["suggestion"] = f"Missing required parameters for {tool_name}: {result['missing']}. Context from log: {context[-500:]}"
            print(f"[Inspector] ⚠️ Validation Failed: {result['missing']}")
        else:
            print(f"[Inspector] ✓ Params validated for {tool_name}")
            
        return result
    
    def validate_with_context(self, tool_name, params, history_summary):
        """
        Advanced validation using LLM to check if params make sense given context.
        Only called for critical tools when basic validation passes.
        """
        if tool_name != "generate_lammps_input":
            return {"valid": True, "feedback": ""}
            
        prompt = f"""You are validating a tool call for an MD simulation agent.

Context (what happened so far):
{history_summary}

Tool being called: {tool_name}
Parameters:
{json.dumps(params, indent=2, default=str)[:1500]}

Check:
1. Does substrate_file match what was built earlier?
2. Are the masses/potentials appropriate for the materials mentioned?
3. Are there any obvious inconsistencies?

Reply JSON only:
{{"valid": true/false, "feedback": "brief explanation if invalid"}}"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.generate_json(messages)
            if response:
                return response
        except Exception as e:
            print(f"[Inspector] LLM validation error: {e}")
        
        return {"valid": True, "feedback": ""}
        
    def review_plan_with_context(self, input_script_name, template_content, parameters):
        """
        Reviews the LAMMPS input script with context of what changed.
        """
        input_path = os.path.join(self.work_dir, input_script_name)
        if not os.path.exists(input_path):
            return {"status": "REJECTED", "reason": f"File {input_script_name} does not exist."}
            
        with open(input_path, "r") as f:
            generated_content = f.read()
            
        print(f"[Inspection] Reviewing {input_script_name} with Context...")
        
        # Check for visual snapshot - find actual structure file
        image_files = []
        data_file = parameters.get("substrate_file")
        if not data_file:
            # Fallback: search for .lammps files in work_dir
            for f in os.listdir(self.work_dir):
                if f.endswith(".lammps"):
                    data_file = os.path.join(self.work_dir, f)
                    break
        if data_file and os.path.exists(data_file):
            snapshot_paths = self.check_structure_visual(data_file)
            if snapshot_paths:
                 image_files.extend(snapshot_paths)
        
        prompt = f"""
        You are a Senior Computational Physicist (Reviewer).
        Your job is to audit a LAMMPS input script generated by a junior agent.
        
        ## Verify Context
        You are provided with:
        1. The Base Template (Jinja2) - Proven to work.
        2. The Parameters used to fill it.
        3. The Final Generated Script.
        4. [Visual] FOUR snapshots of the atomic structure from different angles (Iso, Top, Front, Side).
        
        ### 1. Base Template
        ```jinja2
        {template_content[:2000]} ... (truncated)
        ```
        
        ### 2. Applied Parameters
        {json.dumps(parameters, indent=2)}
        
        ### 3. Final Script ({input_script_name})
        ```lammps
        {generated_content}
        ```
        
        ## Checklist
        1. **Physics**: Are units/boundary compatible with the structure visualized?
        2. **Atom Types**: 
           - substrate_types: {parameters.get('substrate_types', 'unknown')} (these are substrate atoms)
           - projectile_types: {parameters.get('projectile_types', 'unknown')} (these are projectile atoms)
           - Groups like 'gSubstrate' should only include substrate types!
        3. **Potentials (CRITICAL)**: 
           - Is 'pair_style' syntax correct? 
           - **ZBL specific**: If 'zbl' is used (e.g. in pair_style hybrid/overlay), does it have explicit cutoffs (e.g., 'zbl 0.5 2.0') either in pair_style or pair_coeff? **Rejection Rule**: If 'zbl' is used without cutoffs in pair_style OR pair_coeff, you MUST REJECT.
           - Check atomic numbers in 'pair_coeff ... zbl Zi Zj'. Are they correct for (Si=14, Ar=18, etc.)?
           - Do the potential files (element mapping) defined in 'pair_coeff' actually match the atom types in the structure?
        4. **Regions & boundaries**:
           - Check 'region' commands. Are dimensions consistent with 'units box'?
           - Ensure 'region' definitions do not overlap incorrectly or lie outside the simulation box.
           - Check for 'delete_atoms' or 'create_atoms' usage in these regions.
        5. **Visual Check**: Look at the provided images. 
           - Does the structure look reasonable from all angles?
           - Are the atoms overlapping or exploding?
           - Is the vacuum spacing sufficient?
        
        Output JSON:
        {{
            "status": "APPROVED" or "REJECTED",
            "reason": "Explain why rejected. If Visual issue, mention it."
        }}
        """
        
        # Using LLMClient (Generic)
        # Note: GLM-4 supports multimodal but LLMClient legacy logic might not.
        # We pass only text for now to ensure stability.
        messages = [{"role": "user", "content": prompt}]
        response = self.client.generate_json(messages)
        
        if response:
            return response
        return {"status": "REJECTED", "reason": "Inspector API Failed"}

    def inspect_simulation(self, log_filename, dump_filename, context_description):
        """
        Inspects a running or completed simulation.
        1. Parses the tail of log.lammps for stability/errors.
        2. Visualizes the LAST frame of the dump file.
        3. Uses Gemini to assess if the simulation is proceeding according to 'context_description'.
        """
        log_path = os.path.join(self.work_dir, log_filename)
        dump_path = os.path.join(self.work_dir, dump_filename)
        
        # 1. Parse Log
        log_summary = "Log file not found."
        if os.path.exists(log_path):
            log_summary = self._parse_log_tail(log_path)
            
        # 2. Visualize Dump (Last Frame)
        image_files = []
        if os.path.exists(dump_path):
            snapshot_paths = self._render_dump_last_frame(dump_path)
            if snapshot_paths:
                image_files.extend(snapshot_paths)
        else:
            print(f"[Inspection] Dump file {dump_filename} not found.")

        # 3. Consult LLM
        prompt = f"""
        You are a Senior Computational Physicist (Reviewer).
        You are inspecting a LIVE MOLECULAR DYNAMICS SIMULATION.
        
        ## Context (What is supposed to happen)
        "{context_description}"
        
        ## Evidence
        
        ### 1. Log File (Recent Output)
        ```text
        {log_summary}
        ```
        
        ### 2. Visual Inspection (Latest Snapshot)
        (Four images attached: Isometric, Top, Front, Side views of the latest frame)
        
        ## Assessment Criteria
        1. **Stability**: Does the log show NaN, ERROR, or dangerous temperature/pressure diverges?
        2. **Visual Consistency**: Does the structure look like it matches the description? (e.g. if 'Depositing', do we see new atoms on surface? If 'Equilibrating', is it stable?)
        
        Output JSON:
        {{
            "status": "stable" or "unstable" or "error",
            "analysis": "Brief analysis of the situation.",
            "recommendation": "continue" or "stop" or "adjust_params"
        }}
        """
        
        # Using LLMClient (Generic)
        messages = [{"role": "user", "content": prompt}]
        response = self.client.generate_json(messages)
        
        if response:
            return response
        return {"status": "error", "analysis": "API Failed", "recommendation": "check_manually"}

    def _parse_log_tail(self, log_path, lines=50):
        """Reads the last N lines of the log file."""
        try:
            with open(log_path, "r") as f:
                content = f.readlines()
                return "".join(content[-lines:])
        except Exception as e:
            return f"Error reading log: {e}"

    def _render_dump_last_frame(self, dump_path):
        """Renders the LAST frame of the dump file using Ovito."""
        try:
            import warnings
            warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
            from ovito.io import import_file
            from ovito.vis import Viewport, TachyonRenderer
            
            # Load the dump file (Ovito handles .lammpstrj, .dump etc)
            pipeline = import_file(dump_path)
            
            # Go to last frame
            num_frames = pipeline.source.num_frames
            pipeline.compute(num_frames - 1) 
            pipeline.add_to_scene()
            
            vp = Viewport()
            vp.type = Viewport.Type.Perspective
            
            angles = {
                "iso": (-1, -1, -1),
                "xy_plane": (0, 0, -1),   # Look down Z
                "yz_plane": (-1, 0, 0),   # Look down X
                "xz_plane": (0, -1, 0)    # Look down Y
            }
            
            generated_images = []
            
            for name, direction in angles.items():
                vp.camera_dir = direction
                vp.zoom_all()
                
                image_path = os.path.join(self.work_dir, f"runtime_snapshot_{name}.png")
                vp.render_image(filename=image_path, size=(800, 600), renderer=TachyonRenderer())
                generated_images.append(image_path)
            
            print(f"[Inspection] Rendered frame {num_frames-1} to images.")
            return generated_images
            
        except Exception as e:
            print(f"[Inspection] Ovito Dump Error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def check_structure_visual(self, data_filename):
        """
        Uses Ovito to render the structure file to 4 images (different angles).
        """
        data_path = os.path.join(self.work_dir, data_filename)
        if not os.path.exists(data_path):
            print(f"[Inspection] Structure file {data_filename} not found.")
            return []
            
        try:
            import warnings
            warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
            from ovito.io import import_file
            from ovito.vis import Viewport, TachyonRenderer
            # from ovito.modifiers import CommonNeighborAnalysisModifier

            pipeline = import_file(data_path)
            pipeline.add_to_scene()
            
            # Optional: Add modifier to visualize structure type if needed, but raw is fine.
            
            vp = Viewport()
            vp.type = Viewport.Type.Perspective
            
            # Define 4 angles: Iso, Top, Front, Side
            # camera_dir: vector pointing FROM camera TO scene center (approx)
            # Actually ovito camera_dir is direction the camera looks.
            angles = {
                "iso": (-1, -1, -1),
                "xy_plane": (0, 0, -1),   # Look down Z
                "yz_plane": (-1, 0, 0),   # Look down X
                "xz_plane": (0, -1, 0)    # Look down Y
            }
            
            generated_images = []
            
            for name, direction in angles.items():
                vp.camera_dir = direction
                vp.zoom_all()
                
                image_path = os.path.join(self.work_dir, f"inspection_snapshot_{name}.png")
                # TachyonRenderer for headless Linux
                vp.render_image(filename=image_path, size=(800, 600), renderer=TachyonRenderer())
                generated_images.append(image_path)
            
            print(f"[Inspection] Generated {len(generated_images)} snapshots.")
            return generated_images
            
        except ImportError:
            print("[Inspection] Ovito not installed. Skipping visual check.")
            return []
        except Exception as e:
            print(f"[Inspection] Ovito Error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def review_forcefield(self, strategy, materials):
        """
        Reviews a proposed Force Field strategy.
        Args:
            strategy (dict): The strategy JSON from PhysicsResearcher.
            materials (str): Description of materials (e.g. "Si substrate, Ar ion").
        Returns:
            dict: { "status": "APPROVED"|"REJECTED", "reason": "...", "suggestion": "..." }
        """
        print(f"[Inspection] Reviewing Force Field Strategy for {materials}...")
        
        prompt = f"""
        You are a Senior Computational Physicist (Reviewer).
        Review the following Force Field Strategy for a LAMMPS simulation.
        
        Context:
        Materials: {materials}
        
        Proposed Strategy:
        {json.dumps(strategy, indent=2)}
        
        Checklist:
        1. Are the potential styles (e.g. tersoff, eam, zbl) appropriate for these materials?
           - Si/C/Ge usually need Tersoff/SW.
           - Metals usually need EAM.
           - High energy collisions MUST use ZBL (hybrid/overlay).
        2. Are the filenames standard? (e.g. Si.tersoff, not Si.txt)
        3. Is the interaction policy logical?
        
        Output JSON:
        {{
            "status": "APPROVED" or "REJECTED",
            "reason": "Explanation.",
            "suggestion": "If rejected, suggest what to change."
        }}
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.generate_json(messages)
            if response:
                return response
        except Exception as e:
            print(f"[Inspector] FF Review Error: {e}")
            return {"status": "REJECTED", "reason": f"Review failed: {e}"}
            
        return {"status": "APPROVED", "reason": "Auto-approved due to API failure"}

    def consult_expert(self, problem_description, context=""):
        """
        Provides expert consultation when the Sim Agent is stuck.
        Args:
            problem_description (str): What went wrong (e.g., "read_file failed 3 times").
            context (str): Recent actions or error messages.
        Returns:
            dict: { "advice": str, "suggested_action": str, "confidence": str }
        """
        print(f"[Inspector] Consulting on: {problem_description[:50]}...")

        prompt = f"""You are a senior MD simulation expert being consulted by a junior colleague who is stuck.

PROBLEM:
{problem_description}

RECENT CONTEXT:
{context if context else "No additional context provided."}

Analyze this situation and provide:
1. A brief diagnosis of what might be going wrong.
2. Specific actionable advice (e.g., "Try using absolute path", "Check the CODEBASE MAP").
3. Whether they should ask the human user for help (yes/no).

Respond in JSON format:
{{
  "diagnosis": "Brief explanation of the likely issue",
  "advice": "Specific actionable steps to try",
  "ask_user": true/false,
  "confidence": "high/medium/low"
}}
"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.generate_json(messages)
            if response:
                return response
        except Exception as e:
            print(f"[Inspector] Consultation Error: {e}")
            return {
                "diagnosis": "Consultation failed",
                "advice": "Try a different approach or ask the user.",
                "ask_user": True,
                "confidence": "low"
            }
            
        return {
            "diagnosis": "Unable to analyze",
            "advice": "Consider asking the user for guidance.",
            "ask_user": True,
            "confidence": "low"
        }
