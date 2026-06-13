import os, sys
# Add src to path
sys.path.append(os.getcwd())
from src.tools_lib import AgentTools

# Use Windows path for work_dir when running with Windows python
work_dir = r'D:\02.Project\02.Agent\01.Sim_Agent\md_agent\results\run_Ar_SiO2_1000evts'
tools = AgentTools(work_dir)

# Set session state
tools._substrate_elements = ['Si', 'O']
tools._projectile_elements = ['Ar']
tools._substrate_file = 'SiO2_periodic.data'

params = {
    'filename': 'in.sputtering',
    'substrate_file': 'SiO2_periodic.data',
    'molecule_file': 'Ar.txt',
    'events': 1000,
    'max_energy': 100.0,
    'has_tilt': True,
    'xy': -17.202377690357107,
    'xz': 0.0,
    'yz': 0.0,
    'lz1': 21.72520456,
    'masses': {1: 28.086, 2: 15.999, 3: 39.948},
    'projectile_type': 3,
    'potential_commands': """pair_style hybrid/overlay tersoff zbl 0.5 2.0
pair_coeff * * tersoff SiO.tersoff Si O NULL
pair_coeff 1 3 zbl 14 18
pair_coeff 2 3 zbl 8 18
pair_coeff 3 3 zbl 18 18
pair_coeff 1 1 zbl 14 14
pair_coeff 1 2 zbl 14 8
pair_coeff 2 2 zbl 8 8"""
}

# Call the tool
result = tools.generate_lammps_input(**params)
print(result)
