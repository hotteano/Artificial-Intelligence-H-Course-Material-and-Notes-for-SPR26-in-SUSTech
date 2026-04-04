#!/usr/bin/env python3
"""
Timer wrapper for Heuristic algorithm
Measures execution time without modifying the original algorithm
"""

import subprocess
import sys
import time

def run_with_timer():
    """Run Heuristic algorithm and measure time"""
    args = sys.argv[1:]
    
    print("=" * 60)
    print("Timer Wrapper for Heuristic Algorithm")
    print("=" * 60)
    print(f"Command: python IEMP_Heur.py {' '.join(args)}")
    print()
    
    start_time = time.time()
    
    cmd = [sys.executable, "IEMP_Heur.py"] + args
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print()
    print("=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End time:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Elapsed:    {elapsed:.2f} seconds")
    print(f"           = {elapsed/60:.2f} minutes")
    print(f"           = {elapsed/3600:.2f} hours")
    print("=" * 60)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_with_timer())
