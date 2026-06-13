from mcp.server.fastmcp import FastMCP
from src.structure import StructureBuilder
from src.molecule import MoleculeManager
from src.forcefield import ForceFieldSelector
from src.lammps_gen import LammpsGenerator
from src.slurm_agent import SlurmAgent
import os
import shutil

# Initialize MCP Server
mcp = FastMCP("MD-Agent-Tools")

# Initialize Worker Classes
# Assumes running from /app in Docker
WORK_DIR = os.getcwd()
TEMPLATE_DIR = os.path.join(WORK_DIR, "templates")

sb = StructureBuilder(work_dir=WORK_DIR)
mm = MoleculeManager(work_dir=WORK_DIR)
ff = ForceFieldSelector(library_path=os.path.join(WORK_DIR, "Reference/force_field_library"))
lg = LammpsGenerator(template_dir=TEMPLATE_DIR)
sa = SlurmAgent(template_dir=TEMPLATE_DIR)

@mcp.tool()
def generate_substrate(formula: str, phase: str, lx: float, ly: float, lz: float) -> str:
    """
    Generates a substrate structure file (LAMMPS data format).
    Args:
        formula: Chemical formula (e.g., 'SiO2', 'Si')
        phase: 'crystalline' or 'amorphous'
        lx, ly, lz: Box dimensions in Angstroms
    Returns:
        Path to the generated file.
    """
    return sb.create_substrate(formula, phase, lx, ly, lz)

@mcp.tool()
def generate_molecule(formula: str, atom_types: dict) -> str:
    """
    Generates a molecule file (.txt) for the projectile.
    Args:
        formula: Molecule formula (e.g., 'CF2', 'H2')
        atom_types: Dictionary mapping Element to LAMMPS Type ID (e.g., {"C": 3, "F": 4})
    Returns:
        Path to the generated file.
    """
    return mm.create_molecule(formula, atom_types)

@mcp.tool()
def find_forcefield(elements: list) -> str:
    """
    Finds a suitable force field file for the given elements.
    Args:
        elements: List of element symbols (e.g., ["Si", "O", "C", "F"])
    Returns:
        Path to the potential file, or raises Error if not found.
    """
    result = ff.find_potential(elements)
    if result:
        return result
    return "Error: Force field not found locally. (Web Search trigger would happen here)"

@mcp.tool()
def create_input_script(params: dict) -> str:
    """
    Generates the LAMMPS input script (in.sputtering).
    Args:
        params: Dictionary containing:
            - events: Total number of impacts
            - seed: Random seed
            - substrate_file: Path to substrate file
            - masses: Dict of {type_id: mass}
            - projectile_type: Type ID of projectile center
            - molecule_file: (Optional) Path to molecule file
            - max_energy: Max energy in eV
            - max_angle: Max incident angle
            - potential_commands: Raw LAMMPS commands for pair_style/coeff
    Returns:
        Path to the generated input script.
    """
    return lg.write_input(params)

@mcp.tool()
def submit_job(job_name: str, nodes: int, cores: int, input_script: str) -> dict:
    """
    Submits a LAMMPS job. Checks for Slurm; if unavailable, runs locally.
    Args:
        job_name: Name of the job
        nodes: Number of nodes
        cores: Number of MPI tasks
        input_script: Path to the input script
    Returns:
        Dict with status and job_id (or local run result).
    """
    # Check if Slurm is available (simple check)
    has_slurm = shutil.which("sbatch") is not None
    
    if has_slurm:
        # Slurm Path
        script_path = sa.generate_script({
            "job_name": job_name, 
            "nodes": nodes, 
            "cores": cores,
            "input_script": input_script
        })
        return sa.submit_job(script_path)
    else:
        # Local Execution Path (Windows compatible)
        print(f"[Server] Slurm not found. Running locally: {input_script}")
        # Initialize Executor on demand or reuse if global
        from src.executor import LammpsExecutor
        le = LammpsExecutor(work_dir=os.getcwd())
        
        # Use simple local run
        success, output = le.run(input_script=input_script, use_slurm=False, np=cores)
        
        if success:
             return {"status": "success", "job_id": "local_run", "output": output}
        else:
             return {"status": "error", "error": output}

if __name__ == "__main__":
    print("Starting MD-Agent MCP Server...")
    mcp.run()
