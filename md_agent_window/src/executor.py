import subprocess
import os
import sys
import time
import shutil

class LammpsExecutor:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.log_file = os.path.join(work_dir, "log.lammps")
        
    def find_executable(self, cmd):
        return shutil.which(cmd)
    
    def run_shell_command(self, command):
        """Execute arbitrary shell command (not LAMMPS)."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            stdout_str = result.stdout.strip()
            stderr_str = result.stderr.strip()
            output = stdout_str + "\n" + stderr_str if stderr_str else stdout_str
            
            if result.returncode == 0:
                if not output.strip():
                     return True, "Command executed successfully (no output)."
                return True, output
            else:
                 return False, output
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def run(self, input_script="in.sputtering", use_slurm=False, np=8, slurm_script="run.slurm"):
        """
        Executes the simulation.
        Default: Local execution via mpirun.
        """
        
        # 1. Local Execution (Default)
        if not use_slurm:
            print(f"[Executor] Running locally with mpirun -np {np}...")
            return self._run_local_mpi(input_script, np)
        
        # 2. Slurm (Optional)
        if self.find_executable("sbatch"):
            print(f"[Executor] Submitting {input_script} to Slurm via {slurm_script}...")
            return self._run_slurm(input_script, slurm_script)
              
        # 3. Simulate Execution (For Testing without binaries)
        print("[Executor] No binary found. Simulating execution...")
        return self._run_mock(input_script)

    def _run_slurm(self, input_script, slurm_script="run.slurm"):
        """
        Submit job to Slurm and return immediately (non-blocking).
        Optionally launches background monitor for automated tracking.
        """
        result = subprocess.run(["sbatch", slurm_script], cwd=self.work_dir, capture_output=True, text=True)
        if result.returncode != 0:
            return False, f"Sbatch Failed: {result.stderr}"
        
        job_output = result.stdout.strip()
        print(f"   [Slurm] Job Submitted: {job_output}")
        
        # Extract Job ID
        import re
        job_id_match = re.search(r"Submitted batch job (\d+)", job_output)
        job_id = job_id_match.group(1) if job_id_match else "Unknown"
        
        # Optional: Launch background monitor (non-blocking)
        # This runs separately and does NOT block the Agent
        monitor_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "monitor_simulation.py")
        if os.path.exists(monitor_script):
            try:
                # Run monitor in background with nohup (detached process)
                monitor_cmd = [
                    "nohup", "python3", monitor_script,
                    self.work_dir,
                    "--job-id", str(job_id),
                    "--interval", "30",
                    "--max-runtime", "7200"
                ]
                subprocess.Popen(
                    monitor_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True  # Fully detach from parent process
                )
                print(f"   [Slurm] Background monitor started for Job {job_id}")
            except Exception as e:
                print(f"   [Slurm] Warning: Could not start background monitor: {e}")
        
        # Return immediately with job info (non-blocking)
        return True, job_output

    def _kill_lammps_processes(self):
        """Helper to forcefully terminate running LAMMPS/MPI processes."""
        try:
            if os.name == 'nt': # Windows
                # Check first (optional, but saves showing error if none running)
                check = subprocess.run('tasklist /FI "IMAGENAME eq lmp.exe" /FI "IMAGENAME eq mpiexec.exe"', 
                                     shell=True, capture_output=True, text=True)
                if "lmp.exe" in check.stdout or "mpiexec.exe" in check.stdout:
                    print("   [Executor] Found running LAMMPS/MPI processes. Terminating...")
                    subprocess.run("taskkill /F /IM lmp.exe /IM mpiexec.exe /IM smpd.exe", 
                                 shell=True, capture_output=True)
                    time.sleep(2) # Wait for cleanup
        except Exception as e:
            print(f"   [Executor] Warning: Process cleanup failed: {e}")

    def _run_local_mpi(self, input_script, np=8):
        """
        Runs LAMMPS using mpi (mpiexec on Windows, mpirun on Linux).
        """
        # Determine MPI executable
        mpi_exec = "mpiexec" if sys.platform == "win32" else "mpirun"
        if not self.find_executable(mpi_exec):
            # Fallback
            mpi_exec = "mpirun" if sys.platform == "win32" else "mpiexec"
            if not self.find_executable(mpi_exec):
                 return False, "Neither mpiexec nor mpirun found."

        # Determine LAMMPS executable
        lmp_exec = "lmp" # try default alias
        if sys.platform == "win32":
             # Common Windows binary names
             for cand in ["lmp", "lmp.exe", "lmp_mpi", "lmp_serial"]:
                 if self.find_executable(cand):
                     lmp_exec = cand
                     break
        else:
             if self.find_executable("lmp_mpi"): lmp_exec = "lmp_mpi"
        
        # Build Command
        # mpiexec -np 8 lmp -in in.sputtering
        cmd = [mpi_exec, "-np", str(np), lmp_exec, "-in", input_script]
        
        print(f"[Executor] Running: {' '.join(cmd)}")
        
        # [Safety] Check and Kill existing LAMMPS/MPI processes to prevent duplicates
        self._kill_lammps_processes()

        try:
            with open(self.log_file, "w") as log:
                process = subprocess.Popen(
                    cmd, 
                    cwd=self.work_dir,
                    stdout=log,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for completion
                _, stderr = process.communicate()
                
                if process.returncode != 0:
                    return False, f"LAMMPS Output:\n{stderr}"
                
                return True, "Simulation completed successfully."
                
        except KeyboardInterrupt:
            print("\n   [Executor] User Interrupted! Terminating LAMMPS processes...")
            self._kill_lammps_processes()
            raise
        except Exception as e:
            # Try to clean up anyway in case of other weird crashes
            self._kill_lammps_processes()
            return False, f"Execution failed: {e}"

    def _run_local(self, cmd, input_script):
        try:
            result = subprocess.run(
                [cmd, "-in", input_script],
                cwd=self.work_dir,
                capture_output=True,
                text=True
            )
            # Combine stdout/stderr (LAMMPS prints to stdout mostly)
            output = result.stdout + "\n" + result.stderr
            
            # Save log if not saved by lmp
            if not os.path.exists(self.log_file):
                with open(self.log_file, "w") as f:
                    f.write(output)
                    
            return self._check_log_content(output)
            
        except Exception as e:
            return False, str(e)

    def _run_mock(self, input_script):
        # Read input script to see if we should trigger a bug
        input_path = os.path.join(self.work_dir, input_script)
        if os.path.exists(input_path):
            with open(input_path, "r") as f:
                content = f.read()
            if "INVALID_COMMAND" in content:
                 return False, "ERROR: Unknown command 'INVALID_COMMAND' in line 1"
        
        # Mock Success
        time.sleep(1)
        return True, "Total wall time: 0:00:01"

    def _check_log_content(self, content):
        if "ERROR" in content:
            return False, content
        return True, content
