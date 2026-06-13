import os
import numpy as np

class MoleculeManager:
    def __init__(self, work_dir="."):
        self.work_dir = work_dir
        
    def create_molecule(self, formula, atom_types):
        """
        Creates a molecule file. Supports Ar, H2, CF2, CF4.
        """
        print(f"Generating molecule file for {formula}...")
        
        coords = []
        types = []
        
        # Noble Gases / Single Atoms
        if formula in ["Ar", "He", "Ne", "Kr", "Xe", "F"]:
            coords = [
                (1, 0.0, 0.0, 0.0, formula)
            ]
            types = [
                (1, atom_types.get(formula, 3), formula)
            ]
        elif formula == "CF2":
            coords = [
                (1, 0.0, 0.0, 0.0, "C"),
                (2, 1.3, 0.0, 0.0, "F"),
                (3, -0.65, 1.12, 0.0, "F")
            ]
            types = [
                (1, atom_types.get("C", 1), "C"),
                (2, atom_types.get("F", 2), "F"),
                (3, atom_types.get("F", 2), "F")
            ]
        elif formula == "CF4":
            L = 1.32 / np.sqrt(3)
            coords = [
                (1, 0.0, 0.0, 0.0, "C"),
                (2, L, L, L, "F"),
                (3, L, -L, -L, "F"),
                (4, -L, L, -L, "F"),
                (5, -L, -L, L, "F")
            ]
            types = [
                (1, atom_types.get("C", 1), "C"),
                (2, atom_types.get("F", 2), "F"),
                (3, atom_types.get("F", 2), "F"),
                (4, atom_types.get("F", 2), "F"),
                (5, atom_types.get("F", 2), "F")
            ]
        elif formula == "H2":
            coords = [
                (1, 0.0, 0.0, 0.0, "H"),
                (2, 0.74, 0.0, 0.0, "H")
            ]
            types = [
                (1, atom_types.get("H", 1), "H"),
                (2, atom_types.get("H", 1), "H")
            ]
        else:
            print(f"Warning: Unknown molecule {formula}, creating dummy single atom.")
            coords = [(1, 0.0, 0.0, 0.0, "X")]
            types = [(1, 1, "X")]
            
        filename = os.path.join(self.work_dir, f"{formula}.txt")
        
        with open(filename, 'w') as f:
            f.write(f"#{formula} molecule file\n\n")
            f.write(f"{len(coords)} atoms\n\n")
            f.write("Coords\n\n")
            for pid, x, y, z, comment in coords:
                f.write(f"{pid} {x:.4f} {y:.4f} {z:.4f} #{comment}\n")
            f.write("\nTypes\n\n")
            for pid, tid, comment in types:
                f.write(f"{pid} {tid} #{comment}\n")
                
        return filename
