
from ase.io import read, write
import os
import numpy as np

cif_path = "Reference/cif_database/SiO2.cif"

if not os.path.exists(cif_path):
    print("CIF file not found!")
    exit(1)

print(f"Loading {cif_path}...")
atoms = read(cif_path)

from ase.build import make_supercell, bulk
# Use same multiplier logic roughly
# ax ~ 4.9, target ~ 30 -> nx=7
atoms_super = make_supercell(atoms, [[7,0,0], [0,7,0], [0,0,1]])
print("\nSupercell:")
print(atoms_super.cell)

output_filename = "debug_sio2_ase_write.data"
print(f"Writing to {output_filename}...")
write(output_filename, atoms_super, format='lammps-data', atom_style='charge')

print("\n--- Content of generated file ---")
with open(output_filename, 'r') as f:
    for i, line in enumerate(f):
        if i < 20:
            if "xlo" in line or "xy" in line:
                print(line.strip())
