import os
import json
import sys

# Add src to path
sys.path.append('src')

from tools_lib import AgentTools

# Initialize tools
work_dir = r'D:\02.Project\02.Agent\01.Sim_Agent\md_agent\results\run_Ar_SiO2_1000evts'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

at = AgentTools(work_dir)

# 1. Build Substrate
print("--- Building Substrate ---")
crystal_params = {"structure": "trigonal", "a": 4.913, "c": 5.405}
sub_result = at.build_substrate("SiO2", crystal_params, ion_elements=["Ar"])
print(f"Substrate Result: {sub_result}")

if "error" in sub_result:
    sys.exit(1)

# 2. Create Projectile
print("--- Creating Projectile ---")
molecule_file = at.create_projectile("Ar", sub_result["type_map"])
print(f"Projectile Result: {molecule_file}")

# 3. Prepare Force Field
# Copy SiO.tersoff to work directory
ff_src = r'D:\02.Project\02.Agent\01.Sim_Agent\md_agent\Reference\force_field_library\potentials\SiO.tersoff'
ff_dst = os.path.join(work_dir, 'SiO.tersoff')
import shutil
shutil.copy(ff_src, ff_dst)
print(f"Copied force field to {ff_dst}")

# 4. Generate LAMMPS Input
print("--- Generating LAMMPS Input ---")
# Define potential commands
# Si=1, O=2, Ar=3 (based on type_map: {'Si': 1, 'O': 2, 'Ar': 3})
# Every pair (i,j) must have at least one style.
# tersoff: (1,1), (1,2), (2,2). 3 is NULL.
# zbl: (1,3), (2,3), (3,3). 
potential_commands = """
pair_style hybrid/overlay tersoff zbl 2.0 2.5
pair_coeff * * tersoff SiO.tersoff Si O NULL
pair_coeff 1 3 zbl 14 18
pair_coeff 2 3 zbl 8 18
pair_coeff 3 3 zbl 18 18
"""

# We need to provide masses and other required params
# masses: {1: 28.085, 2: 15.999, 3: 39.948}
masses = {1: 28.085, 2: 15.999, 3: 39.948}

gen_result = at.generate_lammps_input(
    filename="in.sputtering",
    substrate_file=sub_result["filename"],
    molecule_file=molecule_file,
    events=1000,
    max_energy=100.0,
    potential_commands=potential_commands,
    masses=masses,
    projectile_type=3,
    projectile_mass=39.948
)
print(f"Generation Result: {gen_result}")

# 5. Run Simulation
print("--- Running Simulation ---")
run_result = at.run_simulation("in.sputtering", np=4)
print(f"Run Result: {run_result}")
