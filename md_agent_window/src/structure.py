import os
import numpy as np
from ase.build import make_supercell, bulk
from ase.lattice.cubic import Diamond
from ase.io import write, read

class StructureBuilder:
    def __init__(self, work_dir="."):
        self.work_dir = work_dir

    def get_max_lattice_multiple(self, lattice_const, max_size=35):
        """Returns largest multiple of lattice_const that is < max_size."""
        if lattice_const <= 0:
            return max_size
        n = int(max_size / lattice_const)
        return n * lattice_const if n > 0 else lattice_const

    def build_periodic_substrate(self, formula, crystal_params, target_z=80.0, n_atom_types=None, ion_elements=None):
        """
        Builds a SLAB substrate with Vacuum.
        - lx, ly, lz are the largest multiples of lattice constant < 35.
        - After bulk creation, Z is expanded to target_z (80A) for vacuum.
        - Cell origin is (0, 0, 0).
        - n_atom_types: Total atom types including projectile (for LAMMPS header).
                        Auto-calculated from substrate + ion elements if not provided.
        - ion_elements: List of ion element symbols (e.g., ['Ar'], ['C', 'F'] for CF4).
                        Used to auto-calculate total atom types.
        
        Returns: dict with filename, lz1, c, lx, ly, lz, substrate_elements, all_elements
        """
        print(f"Generating Periodic {formula} Slab...")
        atoms = None
        fmt = formula.lower()
        c = 5.0  # Default lattice constant
        substrate_elements = []  # Will be filled based on structure
        
        try:
            # Simple wrapper or fallback for direct ASE Usage
            if formula.lower() == "sio2":
                from ase.io import read
                # Check for CIF in Reference/cif_database
                cif_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Reference/cif_database/SiO2.cif")
                if os.path.exists(cif_path):
                    atoms = read(cif_path)
                else:
                    atoms = bulk(formula)
            else:
                atoms = bulk(formula)
            
            cell = atoms.get_cell()
            c = cell.lengths()[2]
            # Ensure consistent ordering: Si then O (reverse alpha)
            substrate_elements = sorted(list(set(atoms.get_chemical_symbols())), reverse=True)
            
        except Exception as e:
            print(f"Error building {formula} with ASE, falling back to Si: {e}")
            atoms = bulk("Si")
            c = 5.43
            substrate_elements = ["Si"]

        cell = atoms.get_cell()
        lengths = cell.lengths()
        cx, cy, cz = lengths[0], lengths[1], lengths[2]

        # Calculate lx, ly, lz as largest multiples of lattice const < 35
        lx = self.get_max_lattice_multiple(cx, 35)
        ly = self.get_max_lattice_multiple(cy, 35)
        lz1 = self.get_max_lattice_multiple(cz, 35)  # Bulk height before vacuum
        
        print(f"   Lattice Constants: cx={cx:.3f}, cy={cy:.3f}, cz={cz:.3f}")
        print(f"   Calculated Box: lx={lx:.2f}, ly={ly:.2f}, lz1={lz1:.2f} (bulk)")
        print(f"   Final Z: {target_z} (with vacuum)")

        # Supercell multipliers
        nx = max(1, int(round(lx / cx))) if cx > 0 else 1
        ny = max(1, int(round(ly / cy))) if cy > 0 else 1
        nz = max(1, int(round(lz1 / cz))) if cz > 0 else 1
        
        print(f"   Multipliers: {nx}x{ny}x{nz}")
        
        final_atoms = make_supercell(atoms, [[nx,0,0], [0,ny,0], [0,0,nz]])
        
        # Align to bottom (shift atoms so min Z is 0 - origin at 0,0,0)
        pos = final_atoms.get_positions()
        z_min = np.min(pos[:,2])
        x_min = np.min(pos[:,0])
        y_min = np.min(pos[:,1])
        pos[:,0] -= x_min  # Force X origin to 0
        pos[:,1] -= y_min  # Force Y origin to 0
        pos[:,2] -= z_min  # Force Z origin to 0
        final_atoms.set_positions(pos)
        
        # Get actual dimensions after supercell
        actual_cell = final_atoms.get_cell()
        ax, ay, az = actual_cell.lengths()
        lz1 = az  # Actual bulk height
        
        # Set Final Box with vacuum (Z = target_z) - origin remains at 0,0,0
        final_atoms.set_cell([ax, ay, target_z])
        
        # Verify origin is at 0,0,0
        final_pos = final_atoms.get_positions()
        assert np.min(final_pos[:,2]) >= 0, "Cell origin Z must be >= 0!"
        
        print(f"   Final Box: {ax:.2f} x {ay:.2f} x {target_z} (lz1={lz1:.2f})")

        initial_charges = [0.0] * len(final_atoms)
        final_atoms.set_initial_charges(initial_charges)

        filename = os.path.join(self.work_dir, f"{formula}_periodic.data")
        write(filename, final_atoms, format='lammps-data', atom_style='charge')
        
        # Calculate total atom types from substrate + ion elements
        # ion_elements: ['Ar'] or ['C', 'F'] for CF4
        ion_elems = ion_elements if ion_elements else []
        
        # Build all_elements list: substrate first, then ions (unique)
        all_elements = list(substrate_elements)  # Copy
        for elem in ion_elems:
            if elem not in all_elements:
                all_elements.append(elem)
        
        # Calculate total types
        total_types = n_atom_types if n_atom_types else len(all_elements)
        
        # Build type_map: element -> type_id (1-indexed)
        type_map = {elem: idx+1 for idx, elem in enumerate(all_elements)}

        # Patch Types and Masses
        self._patch_data_file(filename, all_elements, type_map)
        
        # Return filename and metadata for template
        return {
            "filename": filename,
            "lz1": lz1,
            "c": cz,
            "lx": ax,
            "ly": ay,
            "lz": target_z,
            "substrate_elements": substrate_elements,
            "all_elements": all_elements,
            "type_map": type_map
        }
        
    def create_substrate(self, formula, crystal_params=None, lx=None, ly=None, lz=None):
        """
        Alias for build_periodic_substrate to match agent tool signature.
        """
        result = self.build_periodic_substrate(formula, crystal_params, target_z=80.0)
        return result
        
    def build_from_cif(self, cif_path, target_z=80.0, max_xy=30.0, ion_elements=None):
        """
        Builds substrate from CIF file using ASE (Enforced by User).
        OVITO is used ONLY for visualization in a separate step.
        """
        import numpy as np
        import math
        from ase.io import read, write
        from ase.build import make_supercell

        print(f"Building from CIF (ASE Only): {cif_path}")
        
        try:
            # 1. Load Atoms (ASE)
            if not os.path.exists(cif_path):
                return {"error": f"CIF file not found: {cif_path}"}
            
            atoms = read(cif_path)
            cell = atoms.get_cell()
            lengths = cell.lengths()
            ax, ay, az = lengths[0], lengths[1], lengths[2]
            print(f"   Unit Cell (ASE): {ax:.2f} x {ay:.2f} x {az:.2f}")

            # 2. Calculate Replication Factors
            nx = max(1, int(math.ceil(max_xy / ax))) if ax > 0 else 1
            ny = max(1, int(math.ceil(max_xy / ay))) if ay > 0 else 1
            if az > 10.0:
                nz = 1
            else:
                nz = max(1, int(math.ceil(20.0 / az)))

            print(f"   Multipliers: {nx}x{ny}x{nz}")
            
            # 3. Create Supercell
            final_atoms = make_supercell(atoms, [[nx,0,0], [0,ny,0], [0,0,nz]])
            final_atoms.set_initial_charges([0.0] * len(final_atoms))
            
            # 4. Export to LAMMPS Data
            basename = os.path.basename(cif_path).replace('.cif', '')
            filename = os.path.join(self.work_dir, f"{basename}_periodic.data")
            
            print(f"   Exporting to {filename}...")
            write(filename, final_atoms, format='lammps-data', atom_style='charge')
            
            # 5. Post-process: Shift and Vacuum
            self._shift_to_origin(filename)
            self._add_vacuum_to_lammps_data(filename, target_z)
            
            # 6. Extract Elements for Patching
            substrate_elements = sorted(list(set(final_atoms.get_chemical_symbols())), reverse=True)
            ion_elems = ion_elements if ion_elements else []
            all_elements = list(substrate_elements)
            for elem in ion_elems:
                if elem not in all_elements: all_elements.append(elem)
            
            type_map = {elem: i+1 for i, elem in enumerate(all_elements)}
            self._patch_data_file(filename, all_elements, type_map)
            
            # 7. Render Visualization (OVITO)
            image_file = filename.replace('.data', '.png')
            self.render_structure(filename, image_file)
            
            # 8. Extract Dimensions
            final_cell = final_atoms.get_cell()
            lx=ly=lz_bulk=0.0
            xy=xz=yz=0.0
            
            if len(final_atoms) > 0:
               lx = final_cell[0,0]
               xy = final_cell[1,0]
               ly = final_cell[1,1]
               xz = final_cell[2,0]
               yz = final_cell[2,1]
               lz_bulk = final_cell[2,2]
            
            has_tilt = (abs(xy) > 1e-5 or abs(xz) > 1e-5 or abs(yz) > 1e-5)
               
            return {
                "filename": filename,
                "lx": lx, "ly": ly, "lz": target_z, "lz1": lz_bulk,
                "substrate_elements": substrate_elements,
                "xy": xy, "xz": xz, "yz": yz, 
                "has_tilt": has_tilt,
                "all_elements": all_elements, "type_map": type_map,
                "image": image_file
            }

        except Exception as e:
            return {"error": f"ASE Structure Build Failed: {e}"}

    def render_structure(self, data_file, output_image_base):
        """
        Renders the structure using OVITO to 4 distinct PNG files (ISO, Top, Front, Right).
        This fulfils the requirement: "Structure by ASE, Visualization by OVITO".
        output_image_base: e.g. "D:/.../SiO2_periodic.png" -> writes SiO2_periodic_iso.png, etc.
        """
        print(f"   [Visualization] Rendering {data_file} via OVITO (4 Views)...")
        try:
            from ovito.io import import_file
            from ovito.vis import Viewport, TachyonRenderer
            
            pipeline = import_file(data_file)
            pipeline.add_to_scene()
            
            base, ext = os.path.splitext(output_image_base)
            
            # 1. Perspective (ISO)
            vp = Viewport()
            vp.type = Viewport.Type.Perspective
            vp.zoom_all()
            vp.render_image(filename=f"{base}_iso{ext}", size=(800, 600), renderer=TachyonRenderer())
            
            # 2. Top
            vp.type = Viewport.Type.Top
            vp.zoom_all()
            vp.render_image(filename=f"{base}_top{ext}", size=(800, 600), renderer=TachyonRenderer())
            
            # 3. Front
            vp.type = Viewport.Type.Front
            vp.zoom_all()
            vp.render_image(filename=f"{base}_front{ext}", size=(800, 600), renderer=TachyonRenderer())
            
            # 4. Right
            vp.type = Viewport.Type.Right
            vp.zoom_all()
            vp.render_image(filename=f"{base}_right{ext}", size=(800, 600), renderer=TachyonRenderer())
            
            print(f"   [Visualization] Saved 4 images to {base}_*{ext}")
            pipeline.remove_from_scene()
            
        except Exception as e:
            print(f"   [Visualization] Failed to render image: {e}")


    
    def _add_vacuum_to_lammps_data(self, filepath, target_z):
        """Modify zhi in LAMMPS data file to add vacuum and sanitize box."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if 'zlo zhi' in line:
                parts = line.split()
                zlo = float(parts[0])
                new_lines.append(f"{zlo} {target_z} zlo zhi\n")
            elif 'xy xz yz' in line:
                # Keep triclinic tilt factors if present
                new_lines.append(line)
            else:
                new_lines.append(line)
        
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(new_lines)
        
    def create_substrate_from_cif(self, cif_filename):
        """Tool Alias"""
        return self.build_from_cif(cif_filename)

    def _patch_atom_types_count(self, filepath, min_types=5):
        # Legacy alias if needed, but we should use _patch_data_file ideally.
        # For now, let's upgrade this method to also patch masses if we can infer them?
        # No, we need element info.
        pass

    def _patch_data_file(self, filepath, all_elements, type_map):
        """
        Updates 'atom types' count and ensures 'Masses' section contains all types.
        Uses a robust section detection approach.
        """
        from ase.data import atomic_masses, atomic_numbers
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        
        # 1. Update Header (atom types)
        total_types = len(all_elements)
        
        for line in lines:
            if "atom types" in line:
                new_lines.append(f"{total_types} atom types\n")
            else:
                new_lines.append(line)
        
        # 2. Scan for Masses section range
        mass_section_idx = -1
        next_section_idx = len(new_lines)
        
        # Common LAMMPS section headers to detect end of Masses
        lammps_sections = [
            "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers", 
            "Pair Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs"
        ]
        
        for i, line in enumerate(new_lines):
            stripped = line.strip()
            if stripped == "Masses":
                mass_section_idx = i
            elif mass_section_idx != -1 and i > mass_section_idx:
                # Check for next section header
                if stripped and stripped.split('#')[0].strip() in lammps_sections:
                    next_section_idx = i
                    break
        
        # 3. Rebuild Masses Section
        if mass_section_idx != -1:
            # Collect masses for all types in type_map
            final_masses = []
            for elem, tid in type_map.items():
                z = atomic_numbers.get(elem, 0)
                mass = atomic_masses[z] if z < len(atomic_masses) else 0.0
                final_masses.append((tid, mass, elem))
            
            # Sort by Type ID
            final_masses.sort(key=lambda x: x[0])
            
            # Construct the new Masses block
            mass_block = ["\n", "Masses\n", "\n"]
            for tid, mass, elem in final_masses:
                mass_block.append(f"{tid} {mass:.4f} # {elem}\n")
            mass_block.append("\n")

            # Remove the old Masses section lines
            del new_lines[mass_section_idx:next_section_idx]
            
            # Insert the new block
            for line in reversed(mass_block):
                new_lines.insert(mass_section_idx, line)
                
            print(f"[StructureBuilder] Refreshed Masses section with {len(final_masses)} types.")
        else:
            print("[StructureBuilder] Warning: 'Masses' section not found in data file.")
        
        with open(filepath, 'w') as f:
            f.writelines(new_lines)

    def _shift_to_origin(self, filepath):
        """
        Manually shifts the simulation box and atoms to the origin (0,0,0)
        by processing the LAMMPS data file as text. Preserves 'xy xz yz' tilt factors.
        """
        print(f"   [StructureBuilder] Post-processing: Shifting to Origin manually...")
        import re

        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            xlo = ylo = zlo = 0.0
            shift_x = shift_y = shift_z = 0.0
            
            in_atoms = False
            # Standard LAMMPS atom_style charge: id type q x y z
            # We assume columns 3, 4, 5 are x, y, z (0-indexed)
            
            for line in lines:
                # Check for header bounds
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
                
                # Check for Atoms section start
                elif "Atoms" in line:
                    in_atoms = True
                    new_lines.append(line)
                
                # Process Atom lines
                elif in_atoms:
                    # Detect end of Atoms section (empty lines are fine, but new section starts with alpha)
                    stripped = line.strip()
                    if not stripped:
                        new_lines.append(line)
                        continue
                    
                    if stripped[0].isalpha():
                        # New section (e.g. Velocities)
                        in_atoms = False
                        new_lines.append(line)
                        continue
                        
                    # Parse coords
                    parts = line.split()
                    # Expecting at least 6 columns for atom_style charge
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
                            # Might be a comment or malformed line
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                        
                else:
                    # Check for xy xz yz line to clean up formatting
                    if "xy xz yz" in line:
                        try:
                            parts = line.split()
                            # usually: val1 val2 val3 xy xz yz
                            xy_val = float(parts[0])
                            xz_val = float(parts[1])
                            yz_val = float(parts[2])
                            new_lines.append(f"{xy_val:.6f} {xz_val:.6f} {yz_val:.6f} xy xz yz\n")
                            continue
                        except ValueError:
                            # If parsing fails, just keep original
                            pass
                            
                    # Other headers or sections
                    new_lines.append(line)
            
            # Write back
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            
            print(f"      Shifted box by ({shift_x}, {shift_y}, {shift_z})")

        except Exception as e:
            print(f"      [Error] Manual shift failed: {e}")
            raise
