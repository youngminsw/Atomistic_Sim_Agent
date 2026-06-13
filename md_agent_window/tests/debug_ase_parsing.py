
from ase.io import read
import os

# Create a file with the "bad" spacing
content = """
5 atoms
1 atom types

0.0 5.0 xlo xhi
0.0 5.0 ylo yhi
0.0 5.0 zlo zhi
-2.5                       0                       0  xy xz yz

Atoms

1 1 0.0 0.0 0.0 0.0
2 1 0.0 1.0 1.0 1.0
3 1 0.0 2.0 2.0 2.0
4 1 0.0 3.0 3.0 3.0
5 1 0.0 4.0 4.0 4.0
"""

filename = "debug_parsing.data"
with open(filename, 'w') as f:
    f.write(content)

print("Reading file with irregular spacing...")
try:
    atoms = read(filename, format='lammps-data', style='charge')
    print("Read success.")
    print("Cell:")
    print(atoms.cell)
    print("Angles:")
    print(atoms.cell.angles())
except Exception as e:
    print(f"Read failed: {e}")
