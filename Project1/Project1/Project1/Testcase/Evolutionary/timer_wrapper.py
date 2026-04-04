#!/usr/bin/env python3
"""
Timer wrapper for EA algorithm
Measures execution time without modifying the original algorithm
"""

import subprocess
import sys
import time

def run_with_timer():
    """Run EA algorithm and measure time"""
    # Get command line arguments (pass through to main script)
    args = sys.argv[1:]
    
    print("=" * 60)
    print("Timer Wrapper for EA Algorithm")
    print("=" * 60)
    print(f"Command: python IEMP_Evol.py {' '.join(args)}")
    print()
    
    # Record start time
    start_time = time.time()
    
    # Run the algorithm
    cmd = [sys.executable, "IEMP_Evol.py"] + args
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    # Record end time
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Output timing results
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
