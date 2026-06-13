import sys
import os
import shutil

# Ensure we can import modules from src/
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.structure import StructureBuilder
from src.molecule import MoleculeManager
from src.lammps_gen import LammpsGenerator
from src.forcefield import ForceFieldSelector
from src.slurm_agent import SlurmAgent

# =============================================================================
# USER CONFIGURATION
# =============================================================================
CONFIG = {
    "job_name": "CF2_SiN_RealRun",
    
    # Projectile
    "ion_formula": "CF2",
    "ion_elements": {"C": 3, "F": 4},
    "energy_max": 100.0,
    "incident_angle_max": 70.0,
    "events": 100,
    
    # Substrate
    "substrate_formula": "Si3N4", 
    "substrate_phase": "crystalline",
    "box_size": [30.0, 30.0, 80.0],
    
    # System
    "masses": {
        1: 28.085, # Si
        2: 14.007, # N
        3: 12.011, # C
        4: 18.998  # F
    },
    "slurm": {
        "nodes": 1,
        "cores": 48,
        "partition": "96core"
    }
}
# =============================================================================

def run_simulation_pipeline():
    print(f">>> Starting MD Agent Job: {CONFIG['job_name']}")
    
    # Define Work Directory
    work_dir = os.path.join(os.getcwd(), "workdir_sin")
    if os.path.exists(work_dir): shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # 1. Generate Substrate
    print(f"\n[1/5] Building {CONFIG['substrate_formula']}...")
    sb = StructureBuilder(work_dir=work_dir)
    sub_file = sb.create_substrate(CONFIG["substrate_formula"], CONFIG["substrate_phase"], *CONFIG["box_size"])
    print(f"      -> {os.path.basename(sub_file)}")

    # 2. Generate Molecule
    print(f"\n[2/5] Creating Projectile {CONFIG['ion_formula']}...")
    mm = MoleculeManager(work_dir=work_dir)
    mol_file = mm.create_molecule(CONFIG["ion_formula"], CONFIG["ion_elements"])
    print(f"      -> {os.path.basename(mol_file)}")

    # 3. Find Forcefield
    print(f"\n[3/5] Locating Forcefield...")
    # Fix path to Reference
    ref_dir = os.path.abspath(os.path.join(os.getcwd(), "Reference/force_field_library"))
    ff = ForceFieldSelector(library_path=ref_dir)
    ff_path = ff.find_potential(["Si", "N", "C", "F"])
    if not ff_path:
        # Fallback for demo
        ff_path = "/opt/lammps/potentials/SiNCF.tersoff"
    print(f"      -> {ff_path}")

    # 4. Generate Input Script
    print(f"\n[4/5] Writing LAMMPS Input Script...")
    lg = LammpsGenerator(template_dir="md_agent/templates")
    lammps_params = {
        "events": CONFIG["events"],
        "seed": 12345,
        "substrate_file": os.path.basename(sub_file),
        "masses": CONFIG["masses"],
        "projectile_type": 3,
        "projectile_mass": CONFIG["masses"][3],
        "molecule_file": os.path.basename(mol_file),
        "molecule_id": "ion_mol",
        "max_energy": CONFIG["energy_max"],
        "max_angle": CONFIG["incident_angle_max"],
        "potential_commands": f"pair_style tersoff\npair_coeff * * {ff_path} Si N C F"
    }
    input_path = lg.write_input(lammps_params, os.path.join(work_dir, "in.sputtering"))
    print(f"      -> {os.path.basename(input_path)}")

    # 5. Generate Queue Script & Submit
    print(f"\n[5/5] Submitting to Slurm...")
    sa = SlurmAgent(template_dir="md_agent/templates")
    slurm_params = {
        "job_name": CONFIG["job_name"],
        "nodes": CONFIG["slurm"]["nodes"],
        "cores": CONFIG["slurm"]["cores"],
        "partition": CONFIG["slurm"]["partition"],
        "input_script": os.path.basename(input_path) # Use relative path inside workdir
    }
    
    # GENERATING 'queue_script' (Matching your request)
    queue_script_path = sa.generate_script(slurm_params, os.path.join(work_dir, "queue_script"))
    print(f"      -> Generated {os.path.basename(queue_script_path)}")
    
    # Change dir to workdir so sbatch runs from there
    # This ensures log.lammps and dump files appear in workdir
    original_dir = os.getcwd()
    os.chdir(work_dir)
    
    try:
        # EXECUTE: sbatch queue_script
        print(f"      -> Executing: sbatch queue_script")
        result = sa.submit_job("queue_script")
        
        if result['status'] == 'success':
            print(f"\n>>> SUBMISSION SUCCESSFUL")
            print(f"    Job ID: {result['job_id']}")
            print(f"    Output: {result.get('output', '')}")
            print(f"    WorkDir: {work_dir}")
        else:
            print(f"\n>>> SUBMISSION FAILED")
            print(f"    Error: {result.get('error')}")
            # If sbatch fails (e.g. local machine), we still have the files.
            print("    (Note: This is expected if 'sbatch' is not installed on this machine)")
            
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    run_simulation_pipeline()
