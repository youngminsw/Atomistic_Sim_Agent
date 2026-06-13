
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from executor import LammpsExecutor

work_dir = "/mnt/d/02.Project/02.Agent/01.Sim_Agent/md_agent/results/run_Ar_SiO2_1000evts"
input_script = "in.sputtering"

executor = LammpsExecutor(work_dir)
success, output = executor.run(input_script=input_script, use_slurm=False, np=4)

print(f"Success: {success}")
print("Output:")
print(output)
