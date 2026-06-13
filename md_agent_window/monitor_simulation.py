#!/usr/bin/env python3
"""
Background monitor for Slurm simulations.
Polls log.lammps periodically and notifies agent when complete or error occurs.
"""
import os
import sys
import time
import argparse
import subprocess

def check_completion(log_file):
    """
    Check if simulation is complete or errored.
    Returns: ('running', 'complete', 'error', 'not_found'), message
    """
    if not os.path.exists(log_file):
        return 'not_found', "Log file not found"
    
    try:
        # Read last 4KB for efficiency
        with open(log_file, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096), 0)
            tail = f.read().decode('utf-8', errors='ignore')
        
        # Check completion
        if "Total wall time" in tail:
            return 'complete', tail
        
        # Check errors
        if "ERROR" in tail.upper():
            return 'error', tail
        
        return 'running', tail
        
    except Exception as e:
        return 'error', f"Failed to read log: {e}"

def check_slurm_queue(job_id=None):
    """Check if job is still in Slurm queue."""
    try:
        if job_id:
            result = subprocess.run(
                ['squeue', '--job', str(job_id)],
                capture_output=True,
                text=True,
                timeout=5
            )
        else:
            result = subprocess.run(
                ['squeue', '--me'],
                capture_output=True,
                text=True,
                timeout=5
            )
        
        output = result.stdout
        # If job_id exists in output (and not just header), job is running
        lines = output.strip().split('\n')
        if len(lines) > 1:  # More than just header
            if job_id:
                return str(job_id) in output
            else:
                return True
        return False
    except Exception:
        return False

def monitor_loop(work_dir, job_id=None, interval=30, max_runtime=7200):
    """
    Monitor simulation in background.
    
    Args:
        work_dir: Working directory containing log.lammps
        job_id: Slurm job ID (optional)
        interval: Polling interval in seconds
        max_runtime: Maximum runtime before timeout (seconds)
    """
    log_file = os.path.join(work_dir, 'log.lammps')
    start_time = time.time()
    
    print(f"[Monitor] Starting background monitor...")
    print(f"  Work Dir: {work_dir}")
    print(f"  Job ID: {job_id or 'N/A'}")
    print(f"  Poll Interval: {interval}s")
    print(f"  Max Runtime: {max_runtime}s ({max_runtime/3600:.1f}h)")
    
    iteration = 0
    while True:
        iteration += 1
        elapsed = time.time() - start_time
        
        # Check timeout
        if elapsed > max_runtime:
            print(f"\n[Monitor] TIMEOUT after {elapsed:.0f}s")
            print(f"[Monitor] Status: TIMEOUT")
            sys.exit(2)
        
        # Check log file
        status, message = check_completion(log_file)
        
        if status == 'complete':
            print(f"\n[Monitor] COMPLETE after {elapsed:.0f}s ({elapsed/60:.1f}min)")
            print(f"[Monitor] Status: COMPLETE")
            sys.exit(0)
        
        elif status == 'error':
            print(f"\n[Monitor] ERROR detected after {elapsed:.0f}s")
            print(f"[Monitor] Status: ERROR")
            print(f"[Monitor] Message: {message[:200]}")
            sys.exit(1)
        
        elif status == 'not_found':
            # Log not created yet, check queue
            in_queue = check_slurm_queue(job_id)
            if not in_queue and elapsed > 60:
                # Job not in queue and no log after 1 min = likely failed to start
                print(f"\n[Monitor] Job not in queue and no log file after {elapsed:.0f}s")
                print(f"[Monitor] Status: FAILED_TO_START")
                sys.exit(1)
        
        # Still running, print progress
        if iteration % 10 == 0:  # Print every 10th iteration
            print(f"[Monitor] Running... ({elapsed/60:.1f}min elapsed, "
                  f"Queue: {'Yes' if check_slurm_queue(job_id) else 'No'})")
        
        time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor LAMMPS simulation")
    parser.add_argument('work_dir', help='Working directory')
    parser.add_argument('--job-id', help='Slurm job ID', default=None)
    parser.add_argument('--interval', type=int, default=30, 
                        help='Polling interval (seconds)')
    parser.add_argument('--max-runtime', type=int, default=7200,
                        help='Maximum runtime (seconds, default 2h)')
    
    args = parser.parse_args()
    
    try:
        monitor_loop(args.work_dir, args.job_id, args.interval, args.max_runtime)
    except KeyboardInterrupt:
        print("\n[Monitor] Interrupted by user")
        sys.exit(130)
