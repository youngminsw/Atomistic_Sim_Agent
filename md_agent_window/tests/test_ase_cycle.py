
import os
from ase.io import read, write
from ase.build import bulk
import numpy as np

# 1. Create a dummy triclinic system
a = 5.0
atoms = bulk('Si', 'diamond', a=a)
# Apply a shear to make it triclinic
cell = atoms.get_cell()
cell[1,0] = 2.0 # xy tilt
atoms.set_cell(cell, scale_atoms=True)

print("Original Cell:")
print(atoms.cell)

# 2. Write to LAMMPS data
filename = "test_triclinic.data"
write(filename, atoms, format='lammps-data', atom_style='charge')

# 3. Read content to see what it looks like
print("\n--- File Content (First Write) ---")
with open(filename, 'r') as f:
    for line in f:
        if 'xlo' in line or 'xy' in line:
            print(line.strip())

# 4. Read back with ASE
atoms2 = read(filename, format='lammps-data', style='charge')
print("\nRead Back Cell:")
print(atoms2.cell)

# 5. Check if it preserved tilt
if not np.allclose(atoms.cell, atoms2.cell):
    print("\n[!] FATAL: Read back cell does not match original!")
else:
    print("\n[OK] Read back cell matches.")

# 6. Write again (Shift cycle simulation)
write("test_triclinic_2.data", atoms2, format='lammps-data', atom_style='charge')

print("\n--- File Content (Second Write) ---")
with open("test_triclinic_2.data", 'r') as f:
    for line in f:
        if 'xlo' in line or 'xy' in line:
            print(line.strip())
