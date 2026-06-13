
import os
import numpy as np
from ase.io import read, write

# Mock function
def _shift_to_origin(filepath):
    print(f"Processing {filepath}...")
    try:
        # Load with ASE
        atoms = read(filepath, format='lammps-data', style='charge')
        
        # 1. Shift Atoms
        pos = atoms.get_positions()
        min_xyz = np.min(pos, axis=0)
        
        print(f"Min XYZ: {min_xyz}")
        
        if np.any(np.abs(min_xyz) > 1e-4):
            print(f"      Shift vector: {-min_xyz}")
            pos -= min_xyz
            atoms.set_positions(pos)
            
            # 2. Shift Box Origin
            print("      Writing file via ASE...")
            write(filepath, atoms, format='lammps-data', atom_style='charge')
            print("      Write complete.")
        else:
            print("      Already at origin.")
            
    except Exception as e:
        print(f"      [Warning] Failed to shift using ASE: {e}")

    # Manual Parser Part
    print("      Running Manual Parser...")
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    xlo = ylo = zlo = 0.0
    shift_x = shift_y = shift_z = 0.0
    
    in_atoms = False
    
    for line in lines:
        if "xlo xhi" in line:
            parts = line.split()
            xlo = float(parts[0])
            xhi = float(parts[1])
            shift_x = xlo
            new_lines.append(f"{0.0} {xhi - xlo} xlo xhi\n")
        elif "ylo yhi" in line:
            parts = line.split()
            ylo = float(parts[0])
            yhi = float(parts[1])
            shift_y = ylo
            new_lines.append(f"{0.0} {yhi - ylo} ylo yhi\n")
        elif "zlo zhi" in line:
            parts = line.split()
            zlo = float(parts[0])
            zhi = float(parts[1])
            shift_z = zlo
            new_lines.append(f"{0.0} {zhi - zlo} zlo zhi\n")
        elif "Atoms" in line:
            in_atoms = True
            new_lines.append(line)
        elif in_atoms:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue
            if stripped[0].isalpha():
                in_atoms = False
                new_lines.append(line)
                continue
                
            parts = line.split()
            if len(parts) >= 6:
                try:
                    # 3, 4, 5 are x, y, z
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    
                    parts[3] = f"{x - shift_x:.6f}"
                    parts[4] = f"{y - shift_y:.6f}"
                    parts[5] = f"{z - shift_z:.6f}"
                    
                    new_lines.append(" ".join(parts) + "\n")
                except ValueError:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            # Check for xy xz yz
            if "xy xz yz" in line:
                print(f"      FOUND TILT LINE: {line.strip()}")
                try:
                    parts = line.split()
                    xy_val = float(parts[0])
                    xz_val = float(parts[1])
                    yz_val = float(parts[2])
                    new_lines.append(f"{xy_val:.6f} {xz_val:.6f} {yz_val:.6f} xy xz yz\n")
                    continue
                except ValueError:
                    pass
            new_lines.append(line)
    
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

# Run test
input_file = "debug_sio2_ase_write.data" # Should exist from previous step
_shift_to_origin(input_file)

print("\n--- Final File Content ---")
with open(input_file, 'r') as f:
    for i, line in enumerate(f):
        if i < 20:
             if "xlo" in line or "xy" in line:
                print(line.strip())
