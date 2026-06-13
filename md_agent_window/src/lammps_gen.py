import os
from jinja2 import Environment, FileSystemLoader

class LammpsGenerator:
    # Default atomic masses for common elements
    ATOMIC_MASSES = {
        "H": 1.008, "He": 4.003, "C": 12.011, "N": 14.007, "O": 15.999,
        "F": 18.998, "Ne": 20.180, "Si": 28.086, "Ar": 39.948, "Ru": 101.07,
        "W": 183.84, "Cu": 63.546, "Au": 196.97, "Pt": 195.08, "Ti": 47.867
    }
    
    def __init__(self, template_dir="templates"):
        if not os.path.isabs(template_dir):
            template_dir = os.path.join(os.getcwd(), template_dir)
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def generate_hybrid_potential_commands(self, config, atom_map, total_types=5):
        cmds = []
        sub_style = config.get("substrate_style", "tersoff")
        ion_style = config.get("ion_sub_style", "zbl")
        
        cmds.append(f"pair_style hybrid/overlay {sub_style} {ion_style} 0.5 2.0")
        
        # 1. Substrate (Tersoff)
        ion_elements = ["Ar", "He", "Ne", "Kr", "Xe", "F"]
        tersoff_elements = []
        for t in range(1, total_types + 1):
            if t in atom_map:
                el = atom_map[t]
                if el in ion_elements:
                    tersoff_elements.append("NULL")
                else:
                    tersoff_elements.append(el)
            else:
                tersoff_elements.append("NULL")
        
        elements_str = " ".join(tersoff_elements)
        sub_file = config.get("substrate_file", "Si.tersoff")
        cmds.append(f"pair_coeff * * {sub_style} {sub_file} {elements_str}")

        # 2. ZBL - Must cover ALL remaining pairs
        z_map = {"H":1, "He":2, "C":6, "N":7, "O":8, "F":9, "Si":14, "Ar":18, "Ru":44}
        default_Z = 1 
        
        for i in range(1, total_types + 1):
            if i in atom_map:
                Zi = z_map.get(atom_map[i], default_Z)
            else:
                Zi = default_Z
                
            for j in range(i, total_types + 1):
                if j in atom_map:
                    Zj = z_map.get(atom_map[j], default_Z)
                else:
                    Zj = default_Z
                
                el_i_tersoff = tersoff_elements[i-1]
                el_j_tersoff = tersoff_elements[j-1]
                
                tersoff_covers = (el_i_tersoff != "NULL") and (el_j_tersoff != "NULL")
                
                if not tersoff_covers:
                    cmds.append(f"pair_coeff {i} {j} {ion_style} {Zi} {Zj}")

        return "\n".join(cmds)

    def write_input(self, params, output_filename="in.sputtering"):
        # DEBUG: Print incoming params to trace error
        print(f"[DEBUG write_input] params type: {type(params)}")
        print(f"[DEBUG write_input] output_filename: {output_filename}")
        if isinstance(params, str):
            print(f"[DEBUG ERROR] params is a STRING, not dict! Value: {params[:100]}")
            raise TypeError(f"params must be dict, got string: {params[:50]}")
        
        # Ensure type_map is a valid dict
        type_map = params.get("type_map", {})
        if not isinstance(type_map, dict):
            type_map = {}
        
        # Auto-generate type_map from substrate/projectile elements if missing
        substrate_elements = params.get("substrate_elements", [])
        projectile_elements = params.get("projectile_elements", [])
        if not type_map and (substrate_elements or projectile_elements):
            type_id = 1
            for elem in substrate_elements:
                if elem not in type_map:
                    type_map[elem] = type_id
                    type_id += 1
            for elem in projectile_elements:
                if elem not in type_map:
                    type_map[elem] = type_id
                    type_id += 1
            params["type_map"] = type_map
        
        # Auto-populate defaults for missing required parameters
        if "masses" not in params:
            # Build masses from type_map or use defaults
            masses = {}
            for elem, type_id in type_map.items():
                if elem in self.ATOMIC_MASSES:
                    masses[type_id] = self.ATOMIC_MASSES[elem]
            # If still empty, use common defaults for sputtering
            if not masses:
                masses = {1: 15.999, 2: 28.086, 3: 39.948}  # O, Si, Ar (SiO2 defaults)
            params["masses"] = masses
            
        # Default template variables
        defaults = {
            "events": 10,
            "seed": 12345,
            "c": 4.0,  # lattice constant Z
            "lz1": 34.0,  # bulk height
            "molecule_file": None,
            "molecule_id": "ion",
            "projectile_type": 2,
            "projectile_mass": 12.0,
            "max_energy": 100.0,
            "potential_commands": "pair_style lj/cut 3.0\npair_coeff * * 0.01 3.0"
        }
        
        for key, val in defaults.items():
            if key not in params:
                params[key] = val
        
        # Calculate substrate_types and projectile_types from type_map
        type_map = params.get("type_map", {})
        substrate_elements = params.get("substrate_elements", [])
        projectile_elements = params.get("projectile_elements", [])
        
        # Get substrate type IDs
        substrate_type_ids = []
        for elem in substrate_elements:
            if elem in type_map:
                substrate_type_ids.append(str(type_map[elem]))
        
        # Get projectile type IDs
        projectile_type_ids = []
        for elem in projectile_elements:
            if elem in type_map:
                projectile_type_ids.append(str(type_map[elem]))
        
        # Set as space-separated strings for LAMMPS commands
        if substrate_type_ids:
            params["substrate_types"] = " ".join(substrate_type_ids)
        else:
            params["substrate_types"] = "1"  # Default fallback
            
        if projectile_type_ids:
            params["projectile_types"] = " ".join(projectile_type_ids)
            # Ensure singular variable matches the first projectile type (fixes template mismatch)
            params["projectile_type"] = projectile_type_ids[0]
        else:
            params["projectile_types"] = "2 3"  # Default fallback
        
        # Force absolute path for substrate_file if it exists
        if params.get("substrate_file"):
            # Use basename to allow file relocation (Agent often moves files to results dir)
            params["substrate_file"] = os.path.basename(params["substrate_file"])
        
        # [Sanitization] Fix common potential syntax errors
        pot_cmds = params.get("potential_commands", "")
        if "pair_style" in pot_cmds and "zbl" in pot_cmds:
             lines = pot_cmds.splitlines()
             new_lines = []
             for line in lines:
                 if line.strip().startswith("pair_style") and "zbl" in line:
                     # Check if line ends with numbers (cutoffs)
                     import re
                     if not re.search(r"zbl\s+\d", line) and not re.search(r"\d\.\d", line):
                         # If no numbers found after zbl, append default cutoffs
                         # ZBL defaults for Hybrid/Overlay: 
                         # Inner=0.5 (start ZBL), Outer=2.0 (fully Tersoff)
                         line = line.strip() + " 0.5 2.0"
                         print(f"[LammpsGen] Auto-corrected pair_style with ZBL cutoffs: {line}")
                 new_lines.append(line)
             params["potential_commands"] = "\n".join(new_lines)
        
        # Select template based on cell geometry
        # Use prism template for tilted cells (hexagonal, trigonal, triclinic)
        has_tilt = params.get("has_tilt", False)
        if has_tilt:
            template_name = "in.sputtering_prism.j2"
            print(f"[LammpsGen] Using PRISM template for tilted cell")
        else:
            template_name = "in.sputtering.j2"
            print(f"[LammpsGen] Using standard template for orthogonal cell")
        
        template = self.env.get_template(template_name)
        rendered = template.render(params)
        with open(output_filename, "w") as f:
            f.write(rendered)
        return os.path.abspath(output_filename)

