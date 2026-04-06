#!/usr/bin/env python3
"""
Evaluate All Heuristic Results using Evaluator
Calls Evaluator.py on all generated seed_balanced files and measures timing
"""

import subprocess
import time
import sys
import os

# Evaluator path (relative to Heuristic directory)
EVALUATOR_PATH = "../Evaluator/Evaluator.py"

# Test cases configuration
# Format: (name, network, initial, balanced, budget, output_result)
TEST_CASES = [
    ("map1", "../Evaluator/map1/dataset1", "../Evaluator/map1/seed", "map1/seed_balanced", 10, "map1/result.txt"),
    ("map2", "../Evaluator/map2/dataset2", "../Evaluator/map2/seed", "map2/seed_balanced", 15, "map2/result.txt"),
    ("map3", "../Evaluator/map2/dataset2", "../Evaluator/map2/seed", "map3/seed_balanced", 15, "map3/result.txt"),
    ("map4", "map4/dataset3", "map4/seed", "map4/seed_balanced", 15, "map4/result.txt"),
    ("map5", "map5/dataset4", "map5/seed", "map5/seed_balanced", 15, "map5/result.txt"),
]

print("="*70)
print("Evaluate All Heuristic Results")
print("="*70)
print(f"Evaluator: {EVALUATOR_PATH}")
print(f"Total test cases: {len(TEST_CASES)}")
print(f"Note: Using default MC=60000 simulations")
print("="*70)

results = []

for name, network, initial, balanced, budget, output in TEST_CASES:
    print(f"\n{'='*70}")
    print(f"[{name}] Evaluating...")
    print(f"  Network: {network}")
    print(f"  Initial: {initial}")
    print(f"  Balanced: {balanced}")
    print(f"  Budget: {budget}")
    print(f"{'='*70}")
    
    # Check if balanced seed file exists
    if not os.path.exists(balanced):
        print(f"[ERROR] Balanced seed file not found: {balanced}")
        print(f"[SKIP] Run heuristic algorithm first to generate {balanced}")
        results.append((name, budget, None, None, "FILE NOT FOUND"))
        continue
    
    cmd = [
        sys.executable, EVALUATOR_PATH,
        "-n", network,
        "-i", initial,
        "-b", balanced,
        "-k", str(budget),
        "-o", output
    ]
    
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        end = time.perf_counter()
        elapsed = end - start
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # Read the score from output file
        score = None
        try:
            with open(output, 'r') as f:
                score = float(f.read().strip())
        except:
            pass
        
        print(f"\n[{name}] EVALUATION COMPLETED")
        print(f"  Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        if score:
            print(f"  Score: {score:.4f}")
        print(f"  Result saved to: {output}")
        
        results.append((name, budget, elapsed, score, "OK"))
        
    except subprocess.CalledProcessError as e:
        end = time.perf_counter()
        elapsed = end - start
        print(f"[ERROR] Evaluation failed for {name}")
        print(f"  STDOUT: {e.stdout}")
        print(f"  STDERR: {e.stderr}")
        results.append((name, budget, elapsed, None, "FAILED"))

# Final summary
print(f"\n{'='*70}")
print("EVALUATION SUMMARY")
print(f"{'='*70}")
print(f"{'Map':<10} {'Budget':<10} {'Time (s)':<12} {'Time (min)':<12} {'Score':<15} {'Status'}")
print("-"*70)

total_time = 0
for name, budget, elapsed, score, status in results:
    if elapsed is not None:
        total_time += elapsed
        time_str = f"{elapsed:<12.2f}"
        min_str = f"{elapsed/60:<12.2f}"
    else:
        time_str = "N/A         "
        min_str = "N/A         "
    
    score_str = f"{score:<15.4f}" if score else "N/A             "
    print(f"{name:<10} {budget:<10} {time_str} {min_str} {score_str} {status}")

print(f"{'='*70}")
print(f"Total evaluation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"{'='*70}")
