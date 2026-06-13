import os
import glob

class ForceFieldSelector:
    def __init__(self, library_path="./Reference/force_field_library"):
        self.library_path = library_path

    def find_potential(self, elements):
        """
        Finds a potential file for the given elements.
        
        Args:
            elements (list): e.g. ['Si', 'O', 'C', 'F']
            
        Returns:
            str: Path to potential file or None
        """
        print(f"Searching for force field for elements: {elements}")
        
        # 1. Search Local
        if os.path.exists(self.library_path):
            files = os.listdir(self.library_path)
            # Heuristic: Find a file that contains the primary element (e.g., Si)
            # In production, this needs robust parsing of potential file headers.
            for f in files:
                if "tersoff" in f or "sw" in f or "reax" in f:
                    # Check if relevant elements are in filename (simplistic)
                    if any(el in f for el in elements):
                        found_path = os.path.join(self.library_path, f)
                        print(f"Found local potential: {found_path}")
                        return found_path
        
        # 2. Web Search (Simulation)
        print(f"Potential for {elements} not found locally. Initiating Web Search (Simulated)...")
        # Logic: If I were connected to the internet, I would search NIST Interatomic Potentials Repository.
        # For now, we return a placeholder or fail gracefully.
        
        # Check if we have the standard Si.tersoff available in the parent dir (from Reference)
        fallback_path = "Reference/source_code/Molecular Dynamics/Si.tersoff"
        if os.path.exists(fallback_path) and "Si" in elements:
             print(f"Using fallback Si.tersoff from Reference: {fallback_path}")
             return fallback_path
             
        return None
