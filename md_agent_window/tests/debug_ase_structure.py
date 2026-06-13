
import os
import sys
import numpy as np
from ase.io import read, write
from ase.build import make_supercell

def debug_ase_structure():
    print("[DEBUG] Starting ASE Structure Generation Test...")
    
    # 1. Locate CIF file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cif_path = os.path.join(base_dir, "Reference", "cif_database", "SiO2.cif")
    output_path = os.path.join(base_dir, "tests", "debug_sio2.data")
    
    if not os.path.exists(cif_path):
        print(f"[ERROR] CIF file not found: {cif_path}")
        return

    print(f"[DEBUG] Loading CIF: {cif_path}")
    
    # 2. Read Atoms
    atoms = read(cif_path)
    cell = atoms.get_cell()
    print(f"[DEBUG] Original Cell:")
    print(cell)
    print(f"[DEBUG] Original Cell Angles: {cell.angles()}")
    print(f"[DEBUG] Original Cell Lengths: {cell.lengths()}")

    # 3. Calculate Replication
    target_xy = 30.0
    lengths = cell.lengths()
    ax, ay, az = lengths[0], lengths[1], lengths[2]
    
    import math
    nx = max(1, int(math.ceil(target_xy / ax))) if ax > 0 else 1
    ny = max(1, int(math.ceil(target_xy / ay))) if ay > 0 else 1
    nz = 1 # Simple Z for debugging
    
    print(f"[DEBUG] Multipliers: {nx}x{ny}x{nz}")
    
    # 4. Make Supercell
    # Using specific matrix to preserve symmetry
    final_atoms = make_supercell(atoms, [[nx,0,0], [0,ny,0], [0,0,nz]])
    
    final_cell = final_atoms.get_cell()
    print(f"[DEBUG] Final Supercell (ASE Object):")
    print(final_cell)
    print(f"[DEBUG] Final Angles: {final_cell.angles()}")
    
    # 5. Write to LAMMPS Data
    print(f"[DEBUG] Writing to {output_path}...")
    # atom_style='charge' is critical for correct format
    initial_charges = [0.0] * len(final_atoms)
    final_atoms.set_initial_charges(initial_charges)
    write(output_path, final_atoms, format='lammps-data', atom_style='charge')
    
    # 6. Read back and check for tilt factors
    with open(output_path, 'r') as f:
        lines = f.readlines()
        
    has_tilt = False
    box_lines = []
    for line in lines:
        if "xy xz yz" in line:
            has_tilt = True
            box_lines.append(line.strip())
        if "xlo xhi" in line or "ylo yhi" in line or "zlo zhi" in line:
            box_lines.append(line.strip())
            
    print("\n[RESULT INSPECTION]")
    if has_tilt:
        print("SUCCESS: 'xy xz yz' found in output file.")
    else:
        print("FAILURE: 'xy xz yz' MISSING in output file (Orthogonalized?).")
        
    print("Box Dimensions in File:")
    for l in box_lines:
        print(f"  {l}")

if __name__ == "__main__":
    debug_ase_structure()
