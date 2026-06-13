
import os
import sys

# Mock class to test render_structure
class StructureBuilder:
    def render_structure(self, data_file, output_image):
        print(f"   [Visualization] Rendering {data_file} via OVITO...")
        try:
            from ovito.io import import_file
            from ovito.vis import Viewport, TachyonRenderer
            
            pipeline = import_file(data_file)
            pipeline.add_to_scene()
            
            vp = Viewport()
            vp.type = Viewport.Type.Perspective
            # vp.camera_from_perspective() # REMOVED
            vp.zoom_all()
            
            vp.render_image(filename=output_image, size=(800, 600), renderer=TachyonRenderer())
            print(f"   [Visualization] Saved image to {output_image}")
            pipeline.remove_from_scene()
            
        except Exception as e:
            print(f"   [Visualization] Failed to render image: {e}")
            import traceback
            traceback.print_exc()

# Path to the generated data file
data_file = r"D:\02.Project\02.Agent\01.Sim_Agent\md_agent\results\run_Ar_SiO2_1000evts\SiO2_periodic.data"
output_image = r"D:\02.Project\02.Agent\01.Sim_Agent\md_agent\results\run_Ar_SiO2_1000evts\SiO2_periodic_debug.png"

if not os.path.exists(data_file):
    print(f"Error: Data file not found at {data_file}")
else:
    sb = StructureBuilder()
    sb.render_structure(data_file, output_image)
