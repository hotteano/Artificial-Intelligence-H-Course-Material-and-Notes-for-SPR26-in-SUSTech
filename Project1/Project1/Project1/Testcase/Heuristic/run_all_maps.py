#!/usr/bin/env python3
"""
Run Heuristic Algorithm on All Maps (map1-map5)
Usage: python run_all_maps.py
"""

import subprocess
import time
import sys

TEST_CASES = [
    ("map1", "map1/dataset1", "map1/seed", "map1/seed_balanced", 10),
    ("map2", "map2/dataset2", "map2/seed", "map2/seed_balanced", 15),
    ("map3", "map3/dataset2", "map3/seed2", "map3/seed_balanced", 15),
    ("map4", "map4/dataset3", "map4/seed", "map4/seed_balanced", 15),
    ("map5", "map5/dataset4", "map5/seed", "map5/seed_balanced", 15),
]

MC_SIM = 150

print("="*70)
print("Heuristic Algorithm - All Maps Test")
print("="*70)
print(f"MC Simulations: {MC_SIM}")
print(f"Total test cases: {len(TEST_CASES)}")
print("="*70)

results = []

for name, network, initial, output, budget in TEST_CASES:
    print(f"\n{'='*70}")
    print(f"[{name}] Starting... (budget={budget})")
    print(f"{'='*70}")
    
    cmd = [
        sys.executable, "IEMP_Heur.py",
        "-n", network,
        "-i", initial,
        "-b", output,
        "-k", str(budget),
        "--mc-sim", str(MC_SIM)
    ]
    
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.perf_counter()
    elapsed = end - start
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print(f"\n[{name}] COMPLETED in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    results.append((name, budget, elapsed))

# Final summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Map':<10} {'Budget':<10} {'Time (s)':<15} {'Time (min)':<12}")
print("-"*70)

for name, budget, elapsed in results:
    print(f"{name:<10} {budget:<10} {elapsed:<15.2f} {elapsed/60:<12.2f}")

print(f"{'='*70}")
print(f"Total time: {sum(r[2] for r in results):.2f} seconds ({sum(r[2] for r in results)/60:.2f} minutes)")
print(f"{'='*70}")
