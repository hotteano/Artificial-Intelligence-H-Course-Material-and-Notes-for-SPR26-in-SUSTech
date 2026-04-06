#!/usr/bin/env python3
"""
Evaluator Timing Script
Measures the execution time of Evaluator.py on map1 and map2
"""

import subprocess
import time
import os
import sys

# Configuration
EVALUATOR_PATH = "Evaluator.py"
TEST_CASES = [
    {
        "name": "map1",
        "network": "map1/dataset1",
        "initial": "map1/seed",
        "balanced": "map1/seed_balanced",
        "budget": 10,
        "output": "map1/timed_result.txt"
    },
    {
        "name": "map2", 
        "network": "map2/dataset2",
        "initial": "map2/seed",
        "balanced": "map2/seed_balanced",
        "budget": 15,
        "output": "map2/timed_result.txt"
    }
]

def run_evaluator(test_case):
    """Run evaluator on a test case and return elapsed time"""
    cmd = [
        sys.executable,
        EVALUATOR_PATH,
        "-n", test_case["network"],
        "-i", test_case["initial"],
        "-b", test_case["balanced"],
        "-k", str(test_case["budget"]),
        "-o", test_case["output"]
        # Note: Using default --simulations (50000)
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {test_case['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Record start time
    start_time = time.perf_counter()
    
    # Run the evaluator
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluator: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return None
    
    # Record end time
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    
    return elapsed

def main():
    print("="*60)
    print("Evaluator Timing Test")
    print("="*60)
    print(f"Python: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Note: Using DEFAULT --simulations (50000)")
    
    results = []
    
    for test_case in TEST_CASES:
        elapsed = run_evaluator(test_case)
        
        if elapsed is not None:
            results.append({
                "name": test_case["name"],
                "elapsed": elapsed
            })
            
            print(f"\n{'='*60}")
            print(f"TIMING RESULT for {test_case['name']}")
            print(f"{'='*60}")
            print(f"Elapsed Time: {elapsed:.6f} seconds")
            print(f"             = {elapsed/60:.4f} minutes")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Case':<10} {'Time (s)':<15} {'Time (min)':<12} {'Status'}")
    print("-"*60)
    
    for r in results:
        status = "PASS (<48s)" if r["elapsed"] < 48 else "FAIL (>48s)"
        print(f"{r['name']:<10} {r['elapsed']:<15.6f} {r['elapsed']/60:<12.4f} {status}")
    
    print(f"{'='*60}")
    
    # Check if all passed
    all_pass = all(r["elapsed"] < 48 for r in results)
    if all_pass:
        print("[PASS] ALL TESTS PASSED (All < 48 seconds)")
    else:
        print("[FAIL] SOME TESTS FAILED (Some >= 48 seconds)")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
