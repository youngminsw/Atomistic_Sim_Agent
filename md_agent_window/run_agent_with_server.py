import subprocess
import sys
import time
import os
import signal
import atexit
import socket
import shutil

SERVER_SCRIPT = "src/inspection_server.py"
AGENT_SCRIPT = "autonomous_agent.py"
PORT = 8000

def clear_pycache():
    """Remove all __pycache__ directories to ensure fresh code is loaded."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    removed_count = 0
    for root, dirs, files in os.walk(base_dir):
        if "__pycache__" in dirs:
            cache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(cache_path)
                removed_count += 1
            except Exception:
                pass
    if removed_count > 0:
        print(f"[Launcher] Cleared {removed_count} __pycache__ directories.")

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_server():
    print(f"[Launcher] Starting Inspection Server ({SERVER_SCRIPT})...")
    python_exe = sys.executable
    
    # Start server in background
    # We let stderr flow to the console so we can see why it crashes immediately.
    server_process = subprocess.Popen(
        [python_exe, SERVER_SCRIPT],
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL, # Keep stdout quiet (MCP protocol)
        stderr=sys.stderr # Let server errors print directly to console
    )
    return server_process, None

def cleanup(process, _):
    if process:
        print("[Launcher] Shutting down Inspection Server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("[Launcher] Server stopped.")

def print_server_logs():
    print("\n" + "="*40)
    print("       DUMPING SERVER ERROR LOG")
    print("="*40)
    try:
        with open("server_error.log", "r") as f:
            print(f.read())
    except FileNotFoundError:
        print("(Log file not found)")
    print("="*40 + "\n")

def kill_existing_server():
    """Forces kill of any process on the target port using multiple methods."""
    print(f"[Launcher] Checking for existing process on port {PORT}...")
    
    # Method 1: fuser (Standard Linux)
    if shutil.which("fuser"):
        try:
            subprocess.run(["fuser", "-k", f"{PORT}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[Launcher] Cleared port {PORT} using fuser.")
            time.sleep(2)
            return
        except Exception:
            pass
            
    # Method 2: lsof (Alternative)
    if shutil.which("lsof"):
        try:
            # lsof -t -i:8000 returns PIDs
            result = subprocess.run(["lsof", "-t", f"-i:{PORT}"], capture_output=True, text=True)
            pids = result.stdout.strip().split()
            for pid in pids:
                if pid:
                    os.kill(int(pid), signal.SIGKILL)
            if pids:
                print(f"[Launcher] Cleared port {PORT} using lsof (Killed PIDs: {pids}).")
                time.sleep(2)
                return
        except Exception:
            pass
            
    # Method 3: Windows netstat + taskkill
    # Method 3: Windows netstat + taskkill
    try:
        # Find PID using netstat
        result = subprocess.run(f"netstat -ano | findstr :{PORT}", shell=True, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        pids = set()
        for line in lines:
            parts = line.split()
            # Windows netstat -ano output format: Proto Local Address Foreign Address State PID
            # We look for lines containing LISTENING to prevent killing transient connections if preferred,
            # but usually killing anything on that port is what we want for a dev server.
            # However, logic explicitly checked "LISTENING".
            if len(parts) >= 5 and "LISTENING" in parts:
                pid = parts[-1]
                pids.add(pid)
        
        if pids:
            for pid in pids:
                print(f"[Launcher] Killing process {pid} on port {PORT} (Windows)...")
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
            return
        else:
            # Netstat ran but found nothing. This is GOOD.
            # print(f"[Launcher] No existing process found on port {PORT}.") 
            return

    except Exception as e:
        print(f"[Launcher] Windows kill failed: {e}")

    print(f"[Launcher] If you see 'Address already in use' errors, please manually kill the process on port {PORT}.")

def kill_lammps_processes():
    """Forcefully kills lingering LAMMPS/MPI processes to remove file locks."""
    if os.name != 'nt':
        return # Skip on non-Windows for now, or implement pkill as needed
        
    print("[Launcher] Cleaning up potential zombie LAMMPS/MPI processes...")
    try:
        # Check first to avoid error spam
        check = subprocess.run('tasklist /FI "IMAGENAME eq lmp.exe" /FI "IMAGENAME eq mpiexec.exe"', 
                                shell=True, capture_output=True, text=True)
        
        targets = []
        if "lmp.exe" in check.stdout: targets.append("lmp.exe")
        if "mpiexec.exe" in check.stdout: targets.append("mpiexec.exe")
        if "smpd.exe" in check.stdout: targets.append("smpd.exe") # smpd often lingers
        
        if targets:
            print(f"[Launcher] Found {', '.join(targets)}. Killing...")
            subprocess.run("taskkill /F /IM lmp.exe /IM mpiexec.exe /IM smpd.exe", 
                            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1) # Yield for OS cleanup
        else:
            # print("[Launcher] No LAMMPS processes found.")
            pass
            
    except Exception as e:
        print(f"[Launcher] Warning cleaning LAMMPS processes: {e}")

def main():
    agent_args = sys.argv[1:]
    
    # 0. Clear Python cache to ensure fresh code
    clear_pycache()
    
    # 1. ALWAYS Force Kill existing server to ensure freshness
    kill_existing_server()
       
    # 1.5. Kill lingering LAMMPS/MPI processes to unlock files
    kill_lammps_processes()

    # 2. Start New Server
    server_process, err_file = start_server()
    atexit.register(cleanup, server_process, err_file)
    
    # Wait for server to be ready
    print("[Launcher] Waiting for server to initialize (5s)...")
    time.sleep(5)
    
    # Check if server died immediately
    if server_process.poll() is not None:
        print("[Launcher] CRITICAL: Server process died immediately!")
        print_server_logs()
        return

    # 3. Run Agent
    print(f"[Launcher] Running Agent: {AGENT_SCRIPT} {' '.join(agent_args)}")
    python_exe = sys.executable
    
    try:
        subprocess.run([python_exe, AGENT_SCRIPT] + agent_args, check=True)
    except KeyboardInterrupt:
        print("\n[Launcher] Interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n[Launcher] Agent failed with exit code {e.returncode}")
        # Print server logs on failure
        print_server_logs()
    finally:
        if server_process:
             cleanup(server_process, err_file)

if __name__ == "__main__":
    main()
