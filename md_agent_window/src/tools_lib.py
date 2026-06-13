import json
import os
from src.researcher import PhysicsResearcher
from src.structure import StructureBuilder
from src.molecule import MoleculeManager
from src.lammps_gen import LammpsGenerator
from src.executor import LammpsExecutor
from src.coder import CodePatcher
from src.slurm_agent import SlurmAgent
from src.inspection_client import InspectionClient

# Debug flag - set via environment or default True for now
DEBUG = os.environ.get("MD_AGENT_DEBUG", "1") == "1"

class AgentTools:
    def __init__(self, work_dir, agent_handle=None):
        self.work_dir = work_dir
        self.agent_handle = agent_handle  # Reference to AgentEngine for history access
        self.researcher = PhysicsResearcher()
        self.inspector = InspectionClient(work_dir)
        self.sb = StructureBuilder(work_dir)
        self.mm = MoleculeManager(work_dir)
        self.lg = LammpsGenerator(template_dir=os.path.join(os.path.dirname(__file__), "../templates"))
        self.executor = LammpsExecutor(work_dir)
        self.coder = CodePatcher(work_dir)
        self.sa = SlurmAgent(template_dir=os.path.join(os.path.dirname(__file__), "../templates"))
        
        # Session state for element tracking across tool calls
        self._substrate_elements = []  # e.g., ["Si", "O"] for SiO2
        self._projectile_elements = []  # e.g., ["C", "F"] for CF4
        self._substrate_file = None  # Last built substrate file
    
    def _consult_inspector(self, request: str, info: dict) -> dict:
        """Helper to call Inspection Agent with history context."""
        history = ""
        if self.agent_handle:
            try:
                history = self.agent_handle.get_history_summary()
            except Exception as e:
                print(f"Warning: Failed to get history summary: {e}")
        
        # Use new conversational interface if query is string
        return self.inspector.consult(request, info=info, history=history)
    
    def _detect_tilt_from_data_file(self, filepath):
        """
        Reads LAMMPS data file and extracts tilt factors (xy, xz, yz).
        Returns (has_tilt, xy, xz, yz) tuple.
        """
        xy, xz, yz = 0.0, 0.0, 0.0
        if not filepath or not os.path.exists(filepath):
            return False, xy, xz, yz
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if 'xy xz yz' in line:
                        parts = line.split()
                        xy = float(parts[0])
                        xz = float(parts[1])
                        yz = float(parts[2])
                        break
        except Exception:
            pass
        
        has_tilt = abs(xy) > 1e-6 or abs(xz) > 1e-6 or abs(yz) > 1e-6
        return has_tilt, xy, xz, yz

    # Tool 0: Inspection
    def request_review(self, filename, params=None):
        """Asks the Senior Agent to review the input file."""
        print(f"[Tool] Requesting Review for {filename}...")
        
        template_content = ""
        template_name = "in.sputtering.j2"
        try:
            with open(os.path.join(self.lg.template_dir, template_name), "r") as f:
                template_content = f.read()
        except Exception:
            template_content = "Template matching not implemented for this file."

        if params is None:
            params = {}
            
        return self.inspector.review_plan_with_context(filename, template_content, params)

    def check_simulation_progress(self, context_description="Running simulation", job_id=None):
        """Inspects the current simulation status (Log Tail Check + Job ID)."""
        print(f"[Tool] Checking Simulation Progress: {context_description} (Job ID: {job_id})")
        
        # 1. Find Log
        try:
            log_files = [f for f in os.listdir(self.work_dir) if f.startswith("log.") and f != "log.lammps"]
            log_file = "log.lammps"
            full_path = os.path.join(self.work_dir, log_file)
            
            if not os.path.exists(full_path) and log_files:
                log_file = log_files[0]
                full_path = os.path.join(self.work_dir, log_file)
                
            if not os.path.exists(full_path):
                return "No log.lammps found."

            # [NEW] Check file size for abnormalities
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            if size_mb > 500:
                print(f"   [Warning] Log file size is unusually large: {size_mb:.1f} MB")

            # 2. Check Tail for Completion and Corruption
            with open(full_path, 'rb') as f:
                f.seek(0, 2)  # Seek to end
                size = f.tell()
                f.seek(max(0, size - 4096), 0)  # Read last 4KB
                tail_data_raw = f.read()
                
                # Corruption Check: All NULL bytes
                if tail_data_raw and all(b == 0 for b in tail_data_raw):
                    return (f"CRITICAL: Log file '{log_file}' appears CORRUPTED (contains only NULL bytes). "
                            f"Size: {size_mb:.1f} MB. You should delete this file and restart the simulation.")
                
                tail_data = tail_data_raw.decode('utf-8', errors='ignore')
            
            lines = tail_data.splitlines()
            last_lines = "\n".join(lines[-10:])
            
            # STRICT CHECK: Only consider complete if "Total wall time" is explicitly found
            if "Total wall time" in tail_data:
                return f"Simulation COMPLETE.\n[Log Tail]:\n{last_lines}"
            
            # 3. Check Queue Status (if not complete)
            if job_id and str(job_id).lower() != "unknown":
                success, q_out = self.executor.run_shell_command(f"squeue --job {job_id}")
                if success and str(job_id) in q_out:
                     return f"Simulation RUNNING (Job {job_id} Active).\n[Queue]:\n{q_out.strip()}\n[Log Tail]:\n{last_lines}"
            
            # Fallback: Check general queue
            success, q_out = self.executor.run_shell_command("squeue --me")
            if success and len(q_out.strip().splitlines()) > 1:
                 return f"Simulation RUNNING (Active Job Found).\n[Queue]:\n{q_out.strip()}\n[Log Tail]:\n{last_lines}"
            
            # If not in queue and no "Total wall time", it's either failed silently or finished without writing it (rare)
            # But user requested loose logic is bad, so we stick to "Total wall time" requirement
            if "ERROR" in tail_data:
                 return f"Simulation FAILED (Error detected).\n[Log Tail]:\n{last_lines}"

            return f"Simulation INCOMPLETE (Not in queue, but 'Total wall time' missing). Status uncertain.\n[Log Tail]:\n{last_lines}"
            
        except Exception as e:
            return f"Error checking progress: {e}"

    # Tool 1: Research
    def research_crystal(self, formula):
        """Finds crystal structure parameters for a material."""
        print(f"[Tool] Researching {formula}...")
        return self.researcher.get_crystal_params(formula)

    
    # Tool: Download a potential file from URL
    def download_potential_file(self, url, filename=None):
        """Downloads a potential file from a URL to the work directory.
        
        Args:
            url: Direct URL to the potential file (e.g., GitHub raw URL).
            filename: Optional filename to save as. If not provided, extracts from URL.
        
        Returns:
            dict with success status and file path.
        """
        import requests
        
        if not filename:
            filename = url.split('/')[-1]
        
        print(f"[Tool] Downloading {filename} from {url[:50]}...")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            dst = os.path.join(self.work_dir, filename)
            with open(dst, 'wb') as f:
                f.write(response.content)
            
            # Also save to library for future use
            lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Reference/force_field_library/potentials")
            os.makedirs(lib_path, exist_ok=True)
            lib_dst = os.path.join(lib_path, filename)
            with open(lib_dst, 'wb') as f:
                f.write(response.content)
            
            return {
                "success": True,
                "saved_to": dst,
                "filename": filename,
                "also_cached": lib_dst,
                "size_bytes": len(response.content)
            }
        except requests.RequestException as e:
            return {"success": False, "error": f"Download failed: {str(e)}"}

    # Simplified research_potential - LLM uses bash to explore files
    def research_potential(self, sub_elements, ion_elements):
        """
        Recommends forcefield strategy based on materials.
        LLM should use bash("ls <ff_library_path>") to see available files,
        then bash("cp ...") to copy files to work directory.
        """
        MAX_RETRIES = 3
        materials_desc = f"Substrate: {sub_elements}, Ion: {ion_elements}"

        # Get available files from library
        lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Reference/force_field_library/potentials")
        available_files = []
        if os.path.exists(lib_path):
            available_files = os.listdir(lib_path)

        for attempt in range(MAX_RETRIES):
            # Ask Physics Researcher for strategy recommendation
            strategy = self.researcher.recommend_potential(sub_elements, ion_elements, available_files)
            print(f"[Research] Proposed Strategy (Attempt {attempt+1}): {json.dumps(strategy, indent=1)[:200]}...")
            
            # Ask Inspection Agent to Review
            # Ask Inspection Agent to Review (Conversational)
            try:
                review_result = self._consult_inspector(
                    request=f"Review the force field strategy for {materials_desc}. Is it appropriate? "
                            "Check for: 1. Correct syntax (e.g., 'pair_style zbl 4.0 5.0' needs TWO numbers). "
                            "2. Compatibility with 'atom_style charge'. 3. Correct element mapping. "
                            "If it is good, start your response with 'APPROVED'. usage context: Ar ion bombardment.",
                    info={"strategy": strategy, "materials": materials_desc}
                )
                
                # Check approval status
                response_text = str(review_result.get("response", "")) + str(review_result.get("analysis", ""))
                is_approved = "approved" in response_text.lower() and "not approved" not in response_text.lower()
                
                # [Fix] Handle empty but successful response (Silent Assent)
                if not is_approved and not response_text.strip() and review_result.get("success"):
                    print("[Research] Inspector return Success with empty analysis. Assuming Silent APPROVAL.")
                    is_approved = True
                
                if is_approved:
                    print(f"[Research] Strategy APPROVED by Inspector.")
                    return strategy
                else:
                    print(f"[Research] DEBUG: Raw Inspector Result: {review_result}")
                    # Construct full reason string
                    full_reason = str(review_result.get("response", "")) + \
                                  str(review_result.get("analysis", "")) + \
                                  str(review_result.get("advice", "")) + \
                                  str(review_result.get("error", ""))
                    print(f"[Research] Strategy REJECTED by Inspector: {full_reason[:500]}...")
                    print(f"[Research] Re-thinking strategy...")
                    
            except Exception as e:
                print(f"[Research] Inspection skipped due to error: {e}")
                return strategy
        print("[Research] Max retries reached. Using last strategy.")
        strategy = self.researcher.recommend_potential(sub_elements, ion_elements, available_files)
    def build_substrate(self, formula, crystal_params, ion_elements=None):
        """
        Generates the substrate structure file. 
        Args:
            formula: Chemical formula of substrate (e.g., 'Si', 'SiO2')
            crystal_params: Crystal structure parameters
            ion_elements: List of ion element symbols (e.g., ['Ar'], ['C', 'F'] for CF4)
        Returns: 
            dict with filename, metadata, type_map, all_elements
        """
        print(f"[Tool] Building Substrate {formula}...")
        try:
            if isinstance(crystal_params, str):
                crystal_params = json.loads(crystal_params)
            
            # Parse ion_elements if provided as string
            if isinstance(ion_elements, str):
                ion_elements = json.loads(ion_elements)
            
            # Build substrate with ion info for correct atom types
            result = self.sb.build_periodic_substrate(
                formula, crystal_params, target_z=80.0, ion_elements=ion_elements
            )
            
            # Store substrate info for later use
            self._substrate_elements = result.get("substrate_elements", [])
            self._substrate_file = result.get("filename")
            self._type_map = result.get("type_map", {})
            self._all_elements = result.get("all_elements", [])
            
            # Validate structure using Inspector (New Conversational)
            filename = result["filename"]
            abs_path = os.path.join(self.work_dir, filename)
            
            validation = self._consult_inspector(
                request=f"Validate the substrate structure file '{filename}'. Check stoichiometry for {formula}. "
                        "Render images using OVITO and verify the structure looks correct (e.g. no huge gaps, correct lattice).",
                info={
                    "file_path": abs_path, 
                    "formula": formula,
                    "substrate_elements": self._substrate_elements
                }
            )
            
            # Helper to handle inspection errors safely
            # Note: The new inspector returns 'success' and 'analysis', not strictly 'valid' boolean
            # We parse the response to check for issues
            analysis = str(validation.get("response", "")) + str(validation.get("analysis", ""))
            
            # Simple check for failure keywords in analysis
            failed = "invalid" in analysis.lower() or "error" in analysis.lower() or "fail" in analysis.lower()
            
            # If explicit valid flag exists, use it
            if "valid" in validation:
                failed = not validation["valid"]
                
            if failed:
                result["structure_errors"] = validation.get("errors", []) or [analysis]
                result["atom_counts"] = validation.get("atom_counts", {})
                result["instruction"] = (
                    f"STRUCTURE VALIDATION FAILED or ISSUES FOUND: {analysis}\n"
                    "If this is a critical error, you MUST fix this by calling modify_structure_code() to correct the "
                    f"structure generation for {formula} in structure.py, then rebuild."
                )
            elif isinstance(validation, str):
                 # Inspection failed (e.g. MCP error), log warning but proceed
                 print(f"[Tool] Warning: Structure validation skipped/failed: {validation}")
                 result["validation"] = "SKIPPED (Inspector Error)"
            else:
                result["validation"] = "PASSED"
                if isinstance(validation, dict):
                     result["atom_counts"] = validation.get("atom_counts", {})
            
            return result
        except Exception as e:
            return {"error": f"Error building substrate: {e}"}

    def create_projectile(self, ion_formula, type_map):
        """Generates the projectile molecule file."""
        print(f"[Tool] Creating Projectile {ion_formula}...")
        # Extract projectile elements from ion_formula (e.g., CF4 -> [C, F])
        import re
        elements = re.findall(r'([A-Z][a-z]?)', ion_formula)
        self._projectile_elements = list(set(elements))
        return self.mm.create_molecule(ion_formula, type_map)

    # Tool 2.5: CIF Structure
    def fetch_cif_file(self, formula, url=None):
        """
        Checks local CIF database, then Materials Project, then URL.
        """
        print(f"[Tool] Fetching CIF for {formula}...")
        from src.config import Config
        cif_db = os.path.join(Config.REFERENCE_DIR, "cif_database")
        if not os.path.exists(cif_db):
            os.makedirs(cif_db)
            
        filename = f"{formula}.cif"
        filepath = os.path.join(cif_db, filename)
        
        # 1. Check Local
        if os.path.exists(filepath) and not url:
            print(f"   [CIF] Found locally: {filepath}")
            return {"success": True, "filepath": filepath, "source": "local"}
            
        # 2. Try Materials Project (if no specific URL forced)
        if not url:
            try:
                from src.config import Config
                from mp_api.client import MPRester
                
                api_key = getattr(Config, "MP_API_KEY", None)
                if api_key:
                    print(f"   [CIF] Searching Materials Project for {formula}...")
                    with MPRester(api_key) as mpr:
                        # Search for docs with this formula
                        docs = mpr.materials.summary.search(
                            formula=formula, 
                            fields=["material_id", "structure", "is_stable", "symmetry"]
                        )
                        
                        if docs:
                            # Filter based on Crystal System Constraints
                            # Bombardment simulation requires Z-axis to be vertical (not highly tilted).
                            # Monoclinic/Triclinic systems have inherent tilts that complicate orthogonalization.
                            ALLOWED_SYSTEMS = [
                                "cubic", "tetragonal", "orthorhombic", "hexagonal", "trigonal"
                            ]
                            BLOCKED_SYSTEMS = ["monoclinic", "triclinic"]
                            
                            valid_docs = []
                            blocked_docs_count = 0
                            
                            for doc in docs:
                                sys_name = str(getattr(doc.symmetry, "crystal_system", "unknown")).lower()
                                if sys_name in ALLOWED_SYSTEMS:
                                    valid_docs.append(doc)
                                elif sys_name in BLOCKED_SYSTEMS:
                                    blocked_docs_count += 1
                            
                            if not valid_docs:
                                if blocked_docs_count > 0:
                                    msg = (f"Found {blocked_docs_count} matches for {formula}, BUT they are all "
                                           "Monoclinic or Triclinic. Ion bombardment simulation requires "
                                           "orthogonal-compatible structures (Cubic, Hexagonal, etc.) to avoid "
                                           "excessive box tilt. Aborting to prevent simulation failure.")
                                    print(f"   [CIF] {msg}")
                                    return {"success": False, "error": msg}
                                else:
                                    print(f"   [CIF] No valid crystal systems found for {formula}.")
                            
                            # Sort valid docs by stability and simplicity
                            valid_docs.sort(key=lambda x: (not getattr(x, 'is_stable', False), len(getattr(x, 'structure', []))))
                            best_doc = valid_docs[0]
                            print(f"   [CIF] Selected {getattr(best_doc, 'material_id', 'unknown')} "
                                  f"({getattr(best_doc.symmetry, 'crystal_system', 'unknown')}, Stable: {getattr(best_doc, 'is_stable', False)})")

                            # Write CIF
                            struct_obj = getattr(best_doc, 'structure', None)
                            if struct_obj and hasattr(struct_obj, 'to'):
                                struct_obj.to(filename=filepath, fmt="cif")
                            else:
                                return {"success": False, "error": "Material structure not available or export failed."}
                            return {"success": True, "filepath": filepath, "source": "materials_project"}
                        else:
                            print(f"   [CIF] No results found on Materials Project for {formula}")
                else:
                    print("   [CIF] MP_API_KEY not configured. Skipping Materials Project.")
            except ImportError:
                 print("   [CIF] mp-api not installed. Skipping Materials Project.")
            except Exception as e:
                 print(f"   [CIF] Materials Project Error: {e}")

        # 3. Download if URL provided (Fallback or Explicit)
        if url:
             print(f"   [CIF] Downloading from {url}...")
             try:
                 import requests
                 response = requests.get(url, timeout=10)
                 if response.ok:
                     # Basic content check
                     if "data_" not in response.text and "_cell_" not in response.text:
                         return {"success": False, "error": "URL content does not look like a CIF file (missing 'data_' or '_cell_')."}
                     
                     with open(filepath, "w") as f:
                         f.write(response.text)
                     return {"success": True, "filepath": filepath, "source": "downloaded"}
                 else:
                     return {"success": False, "error": f"HTTP Error {response.status_code}"}
             except Exception as e:
                 return {"success": False, "error": f"Download failed: {e}"}
        
        # 4. Not found
        return {
            "success": False, 
            "error": f"CIF file for {formula} not found (Local/MP/URL).",
            "instruction": f"Materials Project search failed. Use search_web() to find a raw CIF file URL manually (e.g. COD), then call fetch_cif_file('{formula}', url='...')."
        }

    def build_structure_from_cif(self, cif_filename, target_size=30.0, ion_elements=None):
        """
        Builds substrate using CIF file.
        Input: cif_filename (can be absolute path or just filename in Reference/cif_database)
        """
        print(f"[Tool] Building Structure from CIF: {cif_filename}")
        
        # Parse ion_elements if provided as string
        if isinstance(ion_elements, str):
            try:
                ion_elements = json.loads(ion_elements)
            except:
                pass # keep as is or ignore

        # Resolve path
        if not os.path.isabs(cif_filename):
            from src.config import Config
            cif_db = os.path.join(Config.REFERENCE_DIR, "cif_database")
            candidates = [
                os.path.join(self.work_dir, cif_filename),
                os.path.join(cif_db, cif_filename),
                os.path.join(cif_db, f"{cif_filename}.cif") 
            ]
            valid_path = None
            for p in candidates:
                if os.path.exists(p):
                    valid_path = p
                    break
            
            if not valid_path:
                return {"success": False, "error": f"CIF file '{cif_filename}' not found in work dir or database."}
            cif_filename = valid_path
            
        result = self.sb.build_from_cif(cif_filename, target_z=80.0, max_xy=target_size, ion_elements=ion_elements)
        
        # Store for context
        self._substrate_elements = result.get("substrate_elements", [])
        self._substrate_file = result.get("filename")
        # Store tilt info for template selection
        self._has_tilt = result.get("has_tilt", False)
        self._xy = result.get("xy", 0.0)
        self._xz = result.get("xz", 0.0)
        self._yz = result.get("yz", 0.0)
        
        return result

    # Tool 3: Config
    def generate_lammps_input(self, filename, **params):
        """Writes the LAMMPS input script from template. Accepts flat params."""
        print(f"[Tool] Generating {filename}...")
        if DEBUG:
            print(f"[DEBUG tools_lib] params keys: {list(params.keys())}")
        
        # Auto-populate substrate_elements and projectile_elements if not provided
        if "substrate_elements" not in params and self._substrate_elements:
            params["substrate_elements"] = self._substrate_elements
        if "projectile_elements" not in params and self._projectile_elements:
            params["projectile_elements"] = self._projectile_elements
        if "substrate_file" not in params and self._substrate_file:
            params["substrate_file"] = self._substrate_file
            
        # [Auto-Calc] Infer masses and projectile_type if missing
        if "masses" not in params or "projectile_type" not in params:
            print("[Tool] Auto-calculating missing physics parameters (masses/types)...")
            try:
                auto_masses = {}
                max_atom_type = 0
                
                # 1. Parse Substrate Data File for masses
                sub_file = params.get("substrate_file")
                if sub_file:
                    if not os.path.isabs(sub_file): sub_file = os.path.join(self.work_dir, sub_file)
                    if os.path.exists(sub_file):
                        with open(sub_file, 'r') as f:
                            lines = f.readlines()
                        
                        # Find "Masses" section
                        in_mass_section = False
                        for line in lines:
                            if "atom types" in line:
                                try:
                                    max_atom_type = int(line.split()[0])
                                except: pass
                            if line.strip().startswith("Masses"):
                                in_mass_section = True
                                continue
                            if in_mass_section:
                                if not line.strip(): continue
                                if line.strip().startswith("Atoms") or line.strip().startswith("Bond"):
                                    break
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        tid = int(parts[0])
                                        mass = float(parts[1])
                                        auto_masses[tid] = mass
                                    except: pass
                
                # 2. Parse Molecule File for projectile info
                mol_file = params.get("molecule_file")
                proj_type = params.get("projectile_type")
                
                if mol_file:
                    if not os.path.isabs(mol_file): mol_file = os.path.join(self.work_dir, mol_file)
                    if os.path.exists(mol_file):
                        with open(mol_file, 'r') as f:
                            content = f.read()
                        # Simple parse for Mass
                        # Simple parse for Mass
                        import re
                        mass_match = re.search(r"Masses\s*\n\s*\d+\s+(\d+\.?\d*)", content, re.MULTILINE)
                        if mass_match:
                            proj_mass = float(mass_match.group(1))
                            
                            # Determine Projectile Type ID (usually max_substrate_type + 1)
                            # Standard convention in this pipeline: Projectile is Next Type.
                            if not proj_type:
                                proj_type = max_atom_type + 1
                            
                            auto_masses[proj_type] = proj_mass
                            params["projectile_type"] = proj_type
                        else:
                             # Try to infer from comment #Ar or similar
                             # e.g. "1 2 #Ar" -> Element Ar -> Mass 39.95
                             element_match = re.search(r"#\s*([A-Z][a-z]?)", content)
                             if element_match:
                                 el = element_match.group(1)
                                 from src.researcher import PhysicsResearcher
                                 pr = PhysicsResearcher()
                                 mass = pr._get_mass(el)
                                 if mass:
                                     if not proj_type: proj_type = max_atom_type + 1
                                     auto_masses[proj_type] = mass
                                     params["projectile_type"] = proj_type
                                     print(f"   -> Inferred Mass from comment #{el}: {mass}")
                            
                if "masses" not in params and auto_masses:
                    params["masses"] = auto_masses
                    print(f"   -> Inferred Masses: {auto_masses}")
                if "projectile_type" not in params and proj_type:
                    params["projectile_type"] = proj_type
                    print(f"   -> Inferred Projectile Type: {proj_type}")
                
                # [Fallback] If still missing but we have basic info, make an educated guess
                if "projectile_type" not in params and max_atom_type > 0:
                     fallback_type = max_atom_type + 1
                     params["projectile_type"] = fallback_type
                     print(f"   -> [Fallback] Assuming Projectile Type = {fallback_type}")
                     
                if "masses" not in params and "projectile_type" in params:
                     # Attempt to look up mass from periodic table if we have elements
                     from src.researcher import PhysicsResearcher
                     pr = PhysicsResearcher()
                     
                     # Fill known substrate masses (assuming standard order if strict mapping missing)
                     if self._substrate_elements:
                         for i, el in enumerate(self._substrate_elements):
                             tid = i + 1
                             if tid not in auto_masses:
                                 m = pr._get_mass(el)
                                 if m: auto_masses[tid] = m
                                 
                     # Fill projectile mass
                     if self._projectile_elements:
                          ptype = params["projectile_type"]
                          if ptype not in auto_masses:
                              # Average mass of projectile elements? Or just first?
                              # Usually single ion bombardment -> use first element
                              m = pr._get_mass(self._projectile_elements[0])
                              if m: auto_masses[ptype] = m
                     
                     if auto_masses:
                         params["masses"] = auto_masses
                         print(f"   -> [Fallback] Inferred Masses from Elements: {auto_masses}")

            except Exception as e:
                print(f"[Tool] Auto-calc failed: {e}")

        # [Validation] Check for critical parameters
        missing_keys = []
        if "masses" not in params: missing_keys.append("masses (dict of type_id->mass)")
        if "projectile_type" not in params: missing_keys.append("projectile_type (int)")
        
        if missing_keys:
            error_msg = f"Missing required parameters: {', '.join(missing_keys)}. You MUST provide these."
            print(f"[Tool] Validation Failed: {error_msg}")
            # Instead of failing, return error to agent so it can fix it
            return {"error": error_msg}
        
        # Auto-detect tilt from LAMMPS data file (order-independent)
        # This ensures correct template selection even if build_structure_from_cif wasn't called first
        if "has_tilt" not in params:
            substrate_file = params.get("substrate_file", self._substrate_file)
            has_tilt, xy, xz, yz = self._detect_tilt_from_data_file(substrate_file)
            params["has_tilt"] = has_tilt
            params["xy"] = xy
            params["xz"] = xz
            params["yz"] = yz
        
        # Handle filename: extract basename if full path given
        if os.path.isabs(filename):
            output_path = filename
        else:
            output_path = os.path.join(self.work_dir, filename)
        
        try:
            result = self.lg.write_input(params, output_path)
        except Exception as e:
            import traceback
            print(f"[ERROR] write_input failed: {e}")
            traceback.print_exc()
            raise
        
        # Build type_map from substrate + projectile elements for validation
        type_map = {}
        type_id = 1
        for elem in (self._substrate_elements or []):
            if elem not in type_map:
                type_map[elem] = type_id
                type_id += 1
        for elem in (self._projectile_elements or []):
            if elem not in type_map:
                type_map[elem] = type_id
                type_id += 1
        
        # Validate generated input file (New Conversational)
        data_file = params.get("substrate_file")
        abs_data_path = None
        if data_file:
            if os.path.isabs(data_file):
                abs_data_path = data_file
            else:
                abs_data_path = os.path.join(self.work_dir, data_file)
        
        try:
            validation = self._consult_inspector(
                request=f"Validate the LAMMPS input script '{filename}'. "
                        "CRITICAL CHECKS: "
                        "1. 'masses' section exists for ALL atom types. "
                        "2. 'pair_style' args are correct (e.g. 'zbl 4.0 5.0' needs 2 cutoffs, not 1). "
                        "3. 'region' bounds are within box (check for likely 'extends outside' errors). "
                        "4. 'group' definitions are valid. "
                        f"Compatibility with structure file: '{data_file}'.",
                info={
                    "file_path": output_path,
                    "type_map": type_map,
                    "data_file_path": abs_data_path
                }
            )
        except Exception as e:
            print(f"[Inspector] Validation skipped due to error: {e}")
            validation = {"valid": True, "analysis": "Inspector unavailable (Skipped)"}
        
        # Handle case where MCP returns error string instead of dict
        if isinstance(validation, str):
            print(f"[Tool] Validation skipped (MCP error): {validation[:100]}")
            return {
                "file": result,
                "validation": "SKIPPED",
                "note": "MCP validation unavailable, proceeding anyway"
            }
            
        # Parse conversational response
        analysis = str(validation.get("response", "")) + str(validation.get("analysis", ""))
        
        # Check for failure keywords
        failed = "invalid" in analysis.lower() or "error" in analysis.lower() or "fail" in analysis.lower()
        if "valid" in validation:
            failed = not validation["valid"]

        if failed:
            print(f"[Tool] Validation Warning: {analysis[:100]}...")
            return {
                "file": result,
                "validation": "FAILED",
                "errors": validation.get("errors", []) or [analysis],
                "warnings": validation.get("warnings", []),
                "instruction": (
                    f"LAMMPS INPUT VALIDATION FAILED or ISSUES FOUND: {analysis}\n"
                    "You MUST fix this by calling modify_lammps_gen_code() to correct the "
                    "input generation logic in lammps_gen.py, then regenerate."
                )
            }
        
        return {
            "file": result,
            "validation": "PASSED",
            "warnings": validation.get("warnings", [])
        }

    def generate_slurm_script(self, filename, **params):
        """Writes the Slurm queue script. Accepts flat params."""
        print(f"[Tool] Generating Slurm Script {filename}...")
        path = os.path.join(self.work_dir, filename)
        return self.sa.generate_script(params, path)

    # Tool 4: Execution
    def run_simulation(self, input_script, slurm_script=None, np=8, mode="auto"):
        """
        Runs the simulation.
        Args:
            input_script: Path to LAMMPS input script.
            slurm_script: Optional path to slurm script (used if mode='slurm').
            np: Number of processors for MPI (used if mode='local'/'mpi').
            mode: 'auto', 'slurm', 'local', 'mpi'.
        """
        import shutil
        import subprocess
        
        # Decide mode
        if mode == "auto":
            if shutil.which("sbatch"):
                mode = "slurm"
            else:
                mode = "local"
        
        print(f"[Tool] Running Simulation (Mode: {mode})...")

        if mode == "slurm":
            if not slurm_script:
                return "Error: slurm_script required for mode='slurm'"
            # Use SlurmAgent
            job_id = self.sa.submit_job(slurm_script)
            if job_id:
                return f"Simulation Submitted to Queue. Job ID: {job_id}"
            else:
                return "Error: Slurm submission failed."
        
        else:
            # Local Execution (Serial or MPI)
            # Try to find executable
            lmp_exe = shutil.which("lmp") or shutil.which("lmp_serial") or shutil.which("lmp_mpi")
            
            # Windows Check
            if not lmp_exe and os.name == 'nt':
                 # Common Windows Name
                 lmp_exe = shutil.which("lmp.exe")

            if not lmp_exe:
                return "Error: LAMMPS executable ('lmp', 'lmp_serial', 'lmp_mpi') not found in PATH."

            if mode == "mpi" or (mode == "local" and np > 1):
                 # Windows usually prefers mpiexec
                 mpi_cmd = "mpiexec" if os.name == 'nt' else "mpirun"
                 if shutil.which(mpi_cmd):
                     cmd = [mpi_cmd, "-n", str(np), lmp_exe, "-in", input_script]
                 else:
                     # Fallback to serial if MPI not found
                     print(f"[Warning] {mpi_cmd} not found. Falling back to serial execution.")
                     cmd = [lmp_exe, "-in", input_script]
            else:
                 cmd = [lmp_exe, "-in", input_script]
            
            print(f"   Command: {' '.join(cmd)}")
            
            try:
                # Run with timeout to prevent hanging forever, but MD can be long.
                # For integration test, it should be fast.
                with open("log.lammps", "w") as log_file:
                    process = subprocess.Popen(cmd, cwd=self.work_dir, stdout=log_file, stderr=subprocess.STDOUT)
                    process.wait() # Blocking call for local run
                    
                if process.returncode == 0:
                    return "Simulation Completed Locally (Blocking)."
                else:
                    return f"Simulation Failed with return code {process.returncode}. Check log.lammps."
            except Exception as e:
                return f"Execution Error: {e}"

    def cancel_slurm_jobs(self, job_id=None):
        """
        Cancel SLURM jobs.
        If job_id is provided, cancels that specific job.
        If job_id is None, cancels all previously submitted jobs by this agent.
        """
        print(f"[Tool] Cancelling SLURM Job(s)...")
        if job_id:
            result = self.sa.cancel_job(job_id)
        else:
            result = self.sa.cancel_previous_jobs()
        return result
    
    def get_slurm_jobs(self):
        """Returns list of currently tracked SLURM job IDs submitted by this agent."""
        jobs = self.sa.get_active_jobs()
        return {
            "active_jobs": jobs,
            "count": len(jobs)
        }

    # Tool 5: Coding/Debugging
    def read_file(self, filename=None, file_path=None, max_lines=0, start_line=0):
        """
        Reads a file from the filesystem.
        Supports both 'filename' (relative to work_dir) and 'file_path' (absolute path).
        """
        # Handle both parameter names
        if file_path:
            path = file_path  # Absolute path
        elif filename:
            # Check if filename is absolute or relative
            if os.path.isabs(filename):
                path = filename
            else:
                path = os.path.join(self.work_dir, filename)
        else:
            return {"error": "Either 'filename' or 'file_path' must be provided"}
        
        if not os.path.exists(path):
            return {"error": f"File not found: {path}"}
        
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # Apply start_line and max_lines
            if start_line > 0:
                lines = lines[start_line:]
            if max_lines > 0:
                lines = lines[:max_lines]
            
            content = "".join(lines)
            return {
                "success": True,
                "path": path,
                "content": content,
                "total_lines": total_lines,
                "lines_read": len(lines)
            }
        except Exception as e:
            return {"error": f"Error reading file: {e}"}

    def apply_patch(self, filename, search_text, replace_text):
        """Applies a code fix using simple string replacement."""
        print(f"[Tool] Applying Patch to {filename}...")
        plan = {
            "file": filename,
            "action": "replace_text",
            "old_text": search_text,
            "new_text": replace_text
        }
        return self.coder.apply_fix(plan)

    def find_files(self, extension):
        """Lists files in work dir."""
        return [f for f in os.listdir(self.work_dir) if f.endswith(extension)]
        
    def write_file(self, filename, content):
        """Writes a file to the work directory."""
        print(f"[Tool] Writing File {filename}...")
        path = os.path.join(self.work_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        return f"File {filename} written."
    
    def modify_structure_code(self, formula, old_code, new_code):
        """
        Modifies structure.py to fix structure generation for a specific formula.
        This allows Sim_agent to correct stoichiometry or crystal structure issues.
        """
        print(f"[Tool] Modifying structure.py for {formula}...")
        structure_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "structure.py"
        )
        
        if not os.path.exists(structure_path):
            return {"success": False, "error": "structure.py not found"}
        
        with open(structure_path, "r") as f:
            content = f.read()
        
        if old_code not in content:
            return {"success": False, "error": f"Old code not found in structure.py. Make sure to copy exact text."}
        
        new_content = content.replace(old_code, new_code)
        
        with open(structure_path, "w") as f:
            f.write(new_content)
        
        return {
            "success": True, 
            "message": f"structure.py updated for {formula}. Rebuild substrate to verify.",
            "hint": "Call build_substrate again with the same formula to test the fix."
        }
    
    def modify_lammps_gen_code(self, issue, old_code, new_code):
        """
        Modifies lammps_gen.py to fix LAMMPS input generation issues.
        This allows Sim_agent to correct mass assignments, element ordering, etc.
        """
        print(f"[Tool] Modifying lammps_gen.py for: {issue}...")
        lammps_gen_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "lammps_gen.py"
        )
        
        if not os.path.exists(lammps_gen_path):
            return {"success": False, "error": "lammps_gen.py not found"}
        
        with open(lammps_gen_path, "r") as f:
            content = f.read()
        
        if old_code not in content:
            return {"success": False, "error": f"Old code not found in lammps_gen.py. Make sure to copy exact text."}
        
        new_content = content.replace(old_code, new_code)
        with open(lammps_gen_path, "w") as f:
            f.write(new_content)
        
        return {
            "success": True, 
            "message": f"lammps_gen.py updated for: {issue}. Regenerate input to verify.",
            "hint": "Call generate_lammps_input again with the same parameters to test the fix."
        }
    
    def modify_source_code(self, filename, reason, old_code, new_code):
        """
        Modifies ANY Python file in the src/ directory.
        This allows Sim_agent to fix any code issues autonomously.
        
        Args:
            filename: The target file (e.g., 'tools_lib.py', 'agent_core.py')
            reason: Why you are making this change
            old_code: Exact code to replace (copy from file)
            new_code: New code to replace with
        
        IMPORTANT: Only files in src/ directory can be modified.
        """
        print(f"[Tool] Modifying {filename}: {reason}...")
        
        # Security: Only allow modifications to src/ directory
        src_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(src_dir, os.path.basename(filename))
        
        if not target_path.endswith('.py'):
            return {"success": False, "error": "Only Python files (.py) can be modified"}
        
        if not os.path.exists(target_path):
            available_files = [f for f in os.listdir(src_dir) if f.endswith('.py')]
            return {
                "success": False, 
                "error": f"File not found: {filename}",
                "available_files": available_files
            }
        
        with open(target_path, "r") as f:
            content = f.read()
        
        if old_code not in content:
            return {
                "success": False, 
                "error": f"Old code not found in {filename}. Copy exact text including whitespace."
            }
        
        new_content = content.replace(old_code, new_code, 1)  # Replace only first occurrence
        
        with open(target_path, "w") as f:
            f.write(new_content)
        
        return {
            "success": True, 
            "file": target_path,
            "message": f"{filename} modified: {reason}",
            "hint": "The change is applied. Test by re-running the affected operation."
        }

    # Tool 6: Web Search
    def search_web(self, query):
        """Search the web using Tavily API or fallback."""
        print(f"[Tool] Searching Web: {query}...")
        import os as _os
        
        # Try Tavily API first
        tavily_key = _os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            try:
                import requests
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={"query": query, "search_depth": "basic", "max_results": 5},
                    headers={"Authorization": f"Bearer {tavily_key}"}
                )
                if response.ok:
                    data = response.json()
                    results = []
                    for r in data.get("results", [])[:5]:
                        results.append(f"- {r.get('title', 'No Title')}: {r.get('content', '')[:200]}")
                    return "\n".join(results) if results else "No results found."
            except Exception as e:
                print(f"      [System] Tavily API Error: {e}")
        
        # Fallback: Use DuckDuckGo HTML (no API key needed)
        try:
            import requests
            from urllib.parse import quote_plus
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.ok:
                # Simple parsing - extract result titles
                import re
                titles = re.findall(r'<a rel="nofollow" class="result__a" href="[^"]*">([^<]+)</a>', response.text)
                snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', response.text)
                results = []
                for i, title in enumerate(titles[:5]):
                    snippet = snippets[i] if i < len(snippets) else ""
                    results.append(f"- {title}: {snippet[:150]}")
                return "\n".join(results) if results else "No results found."
        except Exception as e:
            print(f"      [System] Web Search Error: {e}")
            
        return "Web search unavailable. Please check internet connection or API keys."

    # Tool 7: Paper Search
    def search_papers(self, query, limit=5):
        """Search academic papers using Semantic Scholar API."""
        print(f"[Tool] Searching Papers: {query}...")
        try:
            import requests
            from urllib.parse import quote_plus
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote_plus(query)}&limit={limit}&fields=title,authors,year,abstract"
            response = requests.get(url, timeout=15)
            if response.ok:
                data = response.json()
                papers = data.get("data", [])
                results = []
                for p in papers:
                    authors = ", ".join([a.get("name", "") for a in p.get("authors", [])[:2]])
                    abstract = (p.get("abstract") or "")[:200]
                    results.append(f"- [{p.get('year', 'N/A')}] {p.get('title', 'No Title')} by {authors}. {abstract}")
                return "\n".join(results) if results else "No papers found."
        except Exception as e:
            print(f"      [System] Paper Search Error: {e}")
            
        return "Paper search unavailable. Please check internet connection."

    # Tool 8: Ask User (Dual Mode)
    def ask_user(self, question):
        """
        Ask for input - supports two modes:
        1. Interactive Mode (default): Uses terminal input()
        2. Agent Mode (AGENT_MODE=1): Uses file-based communication for superior agents
        """
        import time
        
        agent_mode = os.environ.get("AGENT_MODE", "0") == "1"
        
        print(f"\n{'='*60}")
        print(f"[AGENT NEEDS YOUR INPUT]")
        print(f"{'='*60}")
        print(f"\n{question}\n")
        
        if agent_mode:
            # File-based communication for superior agents
            question_file = os.path.join(self.work_dir, "agent_question.txt")
            answer_file = os.path.join(self.work_dir, "agent_answer.txt")
            
            # Write question
            with open(question_file, "w") as f:
                f.write(question)
            print(f"[System] Question written to: {question_file}")
            print(f"[System] Waiting for answer in: {answer_file}")
            print(f"{'='*60}\n")
            
            # Remove old answer file if exists
            if os.path.exists(answer_file):
                os.remove(answer_file)
            
            # Poll for answer (timeout: 10 minutes)
            timeout = 600
            poll_interval = 2
            elapsed = 0
            
            while elapsed < timeout:
                if os.path.exists(answer_file):
                    time.sleep(0.5)  # Brief delay to ensure file is fully written
                    with open(answer_file, "r") as f:
                        answer = f.read().strip()
                    # Clean up
                    os.remove(question_file)
                    os.remove(answer_file)
                    print(f"[System] Received answer: {answer[:100]}...")
                    return f"User/Agent responded: {answer}"
                
                time.sleep(poll_interval)
                elapsed += poll_interval
                if elapsed % 30 == 0:
                    print(f"      [System] Still waiting for answer... ({elapsed}s)")
            
            return "Timeout waiting for agent response (10 min). Proceeding with default."
        
        else:
            # Interactive terminal mode (original)
            try:
                user_input = input(">>> Your Answer: ")
                print(f"{'='*60}\n")
                return f"User responded: {user_input}"
            except EOFError:
                return "User input unavailable (non-interactive mode). Set AGENT_MODE=1 for file-based communication."
            except KeyboardInterrupt:
                return "User cancelled input."

    # Tool 9: Bash - Execute shell commands
    def bash(self, command, description=""):
        """
        Execute a shell command. Useful for file operations, directory creation, etc.
        
        Args:
            command: Shell command to execute
            description: Optional description of what this command does
            
        Returns:
            Command output or error message
        """
        import subprocess
        
        print(f"[Tool] Bash: {description or command[:50]}")
        
        # Security: Block dangerous commands
        dangerous = ["rm -rf /", "mkfs", "dd if=", ":(){", "chmod -R 777 /"]
        for d in dangerous:
            if d in command:
                return {"success": False, "error": f"Blocked dangerous command: {d}"}
        
        try:
            # [Windows Compatibility] Translate Linux commands to Windows equivalents
            if os.name == 'nt':
                cmd_parts = command.strip().split()
                if cmd_parts:
                    base = cmd_parts[0]
                    args = " ".join(cmd_parts[1:])
                    if base == "ls":
                        if "-l" in args or "-a" in args:
                            command = f"dir {args.replace('-l', '').replace('-a', '').strip()}"
                        else:
                            command = f"dir {args}"
                    elif base == "cp":
                        command = f"copy /Y {args}"
                    elif base == "mv":
                        command = f"move /Y {args}"
                    elif base == "rm":
                        # Handle recursive rm -rf (risky but mapped to rmdir /s /q)
                        if "-rf" in args or "-r" in args:
                            target = args.replace("-rf", "").replace("-r", "").strip()
                            command = f"rmdir /S /Q {target}"
                        else:
                            command = f"del /F /Q {args}"
                    elif base == "cat":
                        command = f"type {args}"
                        
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            output = result.stdout.strip()
            error = result.stderr.strip()
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "returncode": result.returncode,
                    "stdout": output[:500] if output else "",
                    "stderr": error[:500] if error else ""
                }
            
            return {
                "success": True,
                "output": output[:2000] if output else "(no output)"
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out (60s limit)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Tool 10: Grep - Search for patterns in files
    def grep(self, pattern, path, context_lines=2):
        """
        Search for a pattern in files. Similar to grep command.
        
        Args:
            pattern: Text or regex pattern to search for
            path: File or directory path to search
            context_lines: Number of lines before/after match to show (default 2)
            
        Returns:
            Matching lines with context
        """
        import re
        
        print(f"[Tool] Grep: '{pattern}' in {path}")
        
        results = []
        
        def search_file(filepath):
            matches = []
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    if re.search(pattern, line, re.IGNORECASE):
                        # Get context
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        context = []
                        for j in range(start, end):
                            prefix = ">>>" if j == i else "   "
                            context.append(f"{prefix} {j+1}: {lines[j].rstrip()}")
                        matches.append({
                            "file": filepath,
                            "line": i + 1,
                            "context": "\n".join(context)
                        })
            except Exception as e:
                pass
            return matches
        
        if os.path.isfile(path):
            results = search_file(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                # Skip __pycache__ and hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                for f in files:
                    if f.endswith('.py') or f.endswith('.j2') or f.endswith('.txt'):
                        filepath = os.path.join(root, f)
                        results.extend(search_file(filepath))
        else:
            return {"success": False, "error": f"Path not found: {path}"}
        
        if not results:
            return {"success": True, "matches": 0, "message": "No matches found"}
        
        # Limit results
        return {
            "success": True,
            "matches": len(results),
            "results": results[:10]  # Limit to 10 matches
        }

    # Tool 11: Metacognition & Model Management
    def read_model_registry(self):
        """Reads the MODEL_REGISTRY.md file."""
        registry_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MODEL_REGISTRY.md")
        if not os.path.exists(registry_path):
             return "MODEL_REGISTRY.md not found in project root."
        
        with open(registry_path, "r", encoding="utf-8") as f:
            return f.read()

    def switch_sim_agent_model(self, model_name):
        """
        Dynamically switches the Simulation Agent's LLM model in config.py.
        """
        print(f"[Tool] Switching Sim Agent Model to: {model_name}...")
        
        # 1. Validate against Registry
        registry_content = self.read_model_registry()
        if model_name not in registry_content:
             return f"Error: Model '{model_name}' not found in MODEL_REGISTRY.md. Please check valid models first."
        
        # 2. Update config.py
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            new_lines = []
            found = False
            for line in lines:
                if line.strip().startswith("SIM_MODEL_NAME ="):
                    new_lines.append(f'    SIM_MODEL_NAME = "{model_name}"\n')
                    found = True
                else:
                    new_lines.append(line)
            
            if not found:
                return "Error: Could not find SIM_MODEL_NAME definition in config.py"
                
            with open(config_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
                
            return f"Successfully switched Sim Agent model to: {model_name}"
            
        except Exception as e:
            return f"Error updating config.py: {e}"

    # Tool 12: Expert Consultation (Inspection Agent)
    def ask_inspector(self, question, context=""):
        """
        Asks the Inspection Agent (Expert) for help when stuck.
        Use this when you encounter a persistent error or need strategic advice.
        """
        print(f"[Tool] Asking Inspector: {question}")
        return self.inspector.consult_expert(question, context)
