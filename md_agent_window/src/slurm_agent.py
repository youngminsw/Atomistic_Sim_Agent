import os
import subprocess
import random
import atexit
from jinja2 import Environment, FileSystemLoader

class SlurmAgent:
    # Class-level tracking of all submitted jobs (for cleanup on exit)
    _all_submitted_jobs = []
    _cleanup_registered = False
    
    def __init__(self, template_dir="templates"):
        if not os.path.isabs(template_dir):
            template_dir = os.path.join(os.getcwd(), template_dir)
        self.env = Environment(loader=FileSystemLoader(template_dir))
        
        # Instance-level tracking of jobs submitted by this agent
        self.submitted_jobs = []
        
        # Register cleanup handler (only once per process)
        if not SlurmAgent._cleanup_registered:
            atexit.register(SlurmAgent._cleanup_all_jobs)
            SlurmAgent._cleanup_registered = True

    @classmethod
    def _cleanup_all_jobs(cls):
        """Called on process exit. Cancels all jobs submitted by any SlurmAgent instance."""
        if cls._all_submitted_jobs:
            print(f"\n[SlurmAgent] Cleanup: Cancelling {len(cls._all_submitted_jobs)} job(s) on exit...")
            for job_id in cls._all_submitted_jobs:
                try:
                    result = subprocess.run(
                        ["scancel", str(job_id)],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0:
                        print(f"   [SlurmAgent] Cancelled job {job_id}")
                    else:
                        print(f"   [SlurmAgent] Failed to cancel job {job_id}: {result.stderr.strip()}")
                except Exception as e:
                    print(f"   [SlurmAgent] Error cancelling job {job_id}: {e}")
            cls._all_submitted_jobs.clear()

    def cancel_previous_jobs(self):
        """Cancels all previously submitted jobs by this agent instance."""
        if not self.submitted_jobs:
            return {"status": "no_jobs", "message": "No previous jobs to cancel"}
        
        cancelled = []
        failed = []
        
        for job_id in self.submitted_jobs[:]:  # Copy list to avoid modification during iteration
            try:
                result = subprocess.run(
                    ["scancel", str(job_id)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"   [SlurmAgent] Cancelled previous job {job_id}")
                    cancelled.append(job_id)
                    self.submitted_jobs.remove(job_id)
                    if job_id in SlurmAgent._all_submitted_jobs:
                        SlurmAgent._all_submitted_jobs.remove(job_id)
                else:
                    print(f"   [SlurmAgent] Failed to cancel job {job_id}: {result.stderr.strip()}")
                    failed.append(job_id)
            except subprocess.TimeoutExpired:
                failed.append(job_id)
            except FileNotFoundError:
                # scancel not available (no SLURM on this system)
                print("   [SlurmAgent] scancel not found, skipping cancel")
                break
            except Exception as e:
                print(f"   [SlurmAgent] Error cancelling job {job_id}: {e}")
                failed.append(job_id)
        
        return {
            "status": "success" if not failed else "partial",
            "cancelled": cancelled,
            "failed": failed
        }

    def cancel_job(self, job_id):
        """Cancel a specific job by ID."""
        try:
            result = subprocess.run(
                ["scancel", str(job_id)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print(f"   [SlurmAgent] Cancelled job {job_id}")
                if job_id in self.submitted_jobs:
                    self.submitted_jobs.remove(job_id)
                if job_id in SlurmAgent._all_submitted_jobs:
                    SlurmAgent._all_submitted_jobs.remove(job_id)
                return {"status": "success", "job_id": job_id}
            else:
                return {"status": "error", "job_id": job_id, "error": result.stderr.strip()}
        except FileNotFoundError:
            return {"status": "error", "error": "scancel not found (SLURM not available)"}
        except Exception as e:
            return {"status": "error", "job_id": job_id, "error": str(e)}

    def generate_script(self, params, output_filename="submit.sh"):
        if "lammps_binary" not in params:
            params["lammps_binary"] = "/opt/lammps/lammps-stable_29Aug2024/build/lmp"
        template = self.env.get_template("queue_script.j2")
        rendered = template.render(params)
        with open(output_filename, "w") as f:
            f.write(rendered)
        return os.path.abspath(output_filename)

    def submit_job(self, script_path, cancel_previous=True):
        """
        Submits job. If cancel_previous=True, cancels any previously submitted jobs first.
        If sbatch missing, RUNS MOCK SIMULATION to generate output files.
        """
        # Cancel previous jobs if requested
        if cancel_previous and self.submitted_jobs:
            print(f"[SlurmAgent] Cancelling {len(self.submitted_jobs)} previous job(s) before new submission...")
            self.cancel_previous_jobs()
        
        # 1. Try Sbatch
        has_slurm = False
        try:
            subprocess.run(["sbatch", "--version"], capture_output=True, check=True)
            has_slurm = True
        except:
            has_slurm = False

        if has_slurm:
            try:
                result = subprocess.run(["sbatch", script_path], capture_output=True, text=True, check=True)
                job_id = result.stdout.strip().split()[-1]
                
                # Track the job
                self.submitted_jobs.append(job_id)
                SlurmAgent._all_submitted_jobs.append(job_id)
                print(f"   [SlurmAgent] Job {job_id} submitted and tracked for cleanup")
                
                return {"status": "success", "job_id": job_id, "output": result.stdout}
            except subprocess.CalledProcessError as e:
                return {"status": "error", "error": e.stderr}
        
        else:
            # 2. Mock Execution (Local Fallback)
            print("   [SlurmAgent] 'sbatch' not found. Running MOCK Execution to generate outputs...")
            work_dir = os.path.dirname(os.path.abspath(script_path))
            job_id = str(random.randint(1000,9999))
            
            # Track mock job (for API consistency)
            self.submitted_jobs.append(job_id)
            SlurmAgent._all_submitted_jobs.append(job_id)
            
            # Generate Mock Logs
            with open(os.path.join(work_dir, f"job.o{job_id}"), "w") as f:
                f.write("LAMMPS (Mock Execution)\nLoop time of 10.05 on 1 procs\n")
                
            # Generate Mock Incident Dump
            events = 1000 
            with open(os.path.join(work_dir, "incident.dump"), "w") as f:
                for i in range(events):
                    f.write(f"ITEM: TIMESTEP\n{i}\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS p p f\n0 30\n0 30\n0 80\nITEM: ATOMS id type x y z vx vy vz\n")
                    f.write(f"{100+i} 3 15.0 15.0 50.0 0.0 0.0 -5.0\n")
                    
            with open(os.path.join(work_dir, "traj.dump"), "w") as f:
                f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n100\nITEM: ATOMS id type x y z\n")

            return {"status": "success", "job_id": job_id, "output": f"Submitted batch job {job_id} (Mock Local)"}

    def get_active_jobs(self):
        """Returns list of currently tracked job IDs."""
        return list(self.submitted_jobs)

