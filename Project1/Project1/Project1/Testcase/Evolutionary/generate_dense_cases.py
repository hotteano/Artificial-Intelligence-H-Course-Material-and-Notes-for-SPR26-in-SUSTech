"""Generate harder test cases for Evolutionary algorithm testing.

This version generates graphs with the same node count and edge count as map4/map5
(3454 nodes, 32140 edges), but with characteristics that make the IEM problem harder:
- Higher propagation probabilities (easier to spread = harder to balance)
- Power-law degree distribution (hubs create unpredictable spread)
- Strategic seed placement
"""
import os
import random
import numpy as np


def generate_hard_graph(n_nodes, n_edges, seed=42):
    """
    Generate a hard IEM problem instance.
    
    Strategy:
    1. Power-law degree distribution (scale-free network)
    2. Higher average propagation probabilities
    3. More variance in probabilities
    """
    rng = np.random.default_rng(seed)
    
    # Generate power-law degree sequence
    # Using exponent 2.5 (typical for scale-free networks)
    alpha = 2.5
    degrees = np.zeros(n_nodes, dtype=int)
    
    # Generate degrees following power law
    for i in range(n_nodes):
        # P(k) ~ k^(-alpha)
        # Use inverse transform sampling
        u = rng.random()
        k = int((1 - u) ** (-1/(alpha - 1)))
        k = min(k, n_nodes - 1)  # Cap at n-1
        k = max(k, 1)  # At least 1
        degrees[i] = k
    
    # Adjust to match exact edge count
    current_edges = degrees.sum()
    scale_factor = n_edges / current_edges
    degrees = (degrees * scale_factor).astype(int)
    degrees = np.clip(degrees, 1, n_nodes - 1)
    
    # Fine-tune to match exact edge count
    diff = n_edges - degrees.sum()
    if diff > 0:
        # Add edges to high-degree nodes
        idx = np.argsort(degrees)[-diff:]
        degrees[idx] += 1
    elif diff < 0:
        # Remove edges from low-degree nodes
        idx = np.argsort(degrees)[:-diff]
        degrees[idx] = np.maximum(degrees[idx] - 1, 1)
    
    # Generate edges based on degree sequence (configuration model)
    edges_set = set()
    
    # Create stubs
    stubs = []
    for node, deg in enumerate(degrees):
        stubs.extend([node] * deg)
    
    # Shuffle and pair
    max_attempts = 100
    for attempt in range(max_attempts):
        rng.shuffle(stubs)
        edges_set.clear()
        valid = True
        
        for i in range(0, len(stubs) - 1, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or (u, v) in edges_set:
                valid = False
                break
            edges_set.add((u, v))
        
        if valid and len(edges_set) == n_edges:
            break
    
    # If configuration model failed, fall back to random edge selection
    if len(edges_set) != n_edges:
        print(f"  Configuration model failed, using random fallback...")
        edges_set.clear()
        attempts = 0
        while len(edges_set) < n_edges and attempts < n_edges * 10:
            u = rng.integers(0, n_nodes)
            v = rng.integers(0, n_nodes)
            if u != v and (u, v) not in edges_set:
                edges_set.add((u, v))
            attempts += 1
    
    edges = list(edges_set)
    
    # Generate HARDER probabilities:
    # 1. Higher average probability (0.05-0.15 instead of 0.01-0.08)
    # 2. Hubs have even higher probabilities
    edges_with_probs = []
    max_degree = degrees.max()
    
    for u, v in edges:
        # Hub nodes get higher probabilities
        u_importance = degrees[u] / max_degree
        
        # Higher base probability 0.05-0.10, hubs up to 0.20
        # This makes information spread more aggressively, harder to balance
        base_p1 = 0.05 + 0.08 * u_importance + rng.uniform(0, 0.07)
        base_p2 = 0.05 + 0.08 * u_importance + rng.uniform(0, 0.07)
        
        p1 = round(min(base_p1, 0.20), 4)
        p2 = round(min(base_p2, 0.20), 4)
        edges_with_probs.append((u, v, p1, p2))
    
    actual_degree = len(edges_with_probs) / n_nodes
    return n_nodes, len(edges_with_probs), edges_with_probs, actual_degree


def generate_seed_file(n_nodes, n1, n2, seed=42):
    """Generate initial seed sets."""
    rng = np.random.default_rng(seed)
    all_nodes = list(range(n_nodes))
    rng.shuffle(all_nodes)
    
    i1 = sorted(all_nodes[:n1])
    i2 = sorted(all_nodes[n1:n1+n2])
    
    return i1, i2


def write_dataset(filepath, n_nodes, n_edges, edges):
    """Write dataset to file."""
    with open(filepath, 'w') as f:
        f.write(f"{n_nodes} {n_edges}\n")
        for u, v, p1, p2 in edges:
            f.write(f"{u} {v} {p1} {p2}\n")


def write_seed_file(filepath, i1, i2):
    """Write seed file."""
    with open(filepath, 'w') as f:
        f.write(f"{len(i1)} {len(i2)}\n")
        for node in i1:
            f.write(f"{node}\n")
        for node in i2:
            f.write(f"{node}\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Same as map4/map5
    N_NODES = 3454
    N_EDGES = 32140  # Same edge count as map4/map5
    
    print("="*70)
    print("Generating Harder Test Cases")
    print(f"Node count: {N_NODES}, Edge count: {N_EDGES}")
    print("(Same size as map4/map5, but harder propagation characteristics)")
    print("="*70)
    
    # Map6: Scale-free network with higher probabilities
    print("\nGenerating Map6 (Scale-free, higher probs)...")
    map6_dir = os.path.join(base_dir, "map6")
    os.makedirs(map6_dir, exist_ok=True)
    
    n1_6, n2_6 = 8, 10
    budget_6 = 25
    
    n_nodes, n_edges, edges, avg_deg = generate_hard_graph(N_NODES, N_EDGES, seed=100)
    i1, i2 = generate_seed_file(n_nodes, n1_6, n2_6, seed=101)
    
    write_dataset(os.path.join(map6_dir, "dataset6_hard"), n_nodes, n_edges, edges)
    write_seed_file(os.path.join(map6_dir, "seed"), i1, i2)
    
    # Calculate average probability
    avg_p1 = sum(p1 for _, _, p1, _ in edges) / len(edges)
    avg_p2 = sum(p2 for _, _, _, p2 in edges) / len(edges)
    
    print(f"  Result: {n_nodes} nodes, {n_edges} edges, avg degree: {avg_deg:.1f}")
    print(f"  Avg probabilities: p1={avg_p1:.4f}, p2={avg_p2:.4f}")
    print(f"  Seeds: I1={len(i1)}, I2={len(i2)}, Budget={budget_6}")
    
    # Map7: Different random seed for variety
    print("\nGenerating Map7 (Scale-free, higher probs, variant)...")
    map7_dir = os.path.join(base_dir, "map7")
    os.makedirs(map7_dir, exist_ok=True)
    
    n1_7, n2_7 = 10, 12
    budget_7 = 25
    
    n_nodes, n_edges, edges, avg_deg = generate_hard_graph(N_NODES, N_EDGES, seed=200)
    i1, i2 = generate_seed_file(n_nodes, n1_7, n2_7, seed=201)
    
    write_dataset(os.path.join(map7_dir, "dataset7_hard"), n_nodes, n_edges, edges)
    write_seed_file(os.path.join(map7_dir, "seed"), i1, i2)
    
    avg_p1 = sum(p1 for _, _, p1, _ in edges) / len(edges)
    avg_p2 = sum(p2 for _, _, _, p2 in edges) / len(edges)
    
    print(f"  Result: {n_nodes} nodes, {n_edges} edges, avg degree: {avg_deg:.1f}")
    print(f"  Avg probabilities: p1={avg_p1:.4f}, p2={avg_p2:.4f}")
    print(f"  Seeds: I1={len(i1)}, I2={len(i2)}, Budget={budget_7}")
    
    print("\n" + "="*70)
    print("Harder test cases generated successfully!")
    print("="*70)
    print("\nComparison with map4/map5:")
    print(f"  map4/5:  {N_NODES} nodes, {N_EDGES} edges, avg prob ~0.04-0.05")
    print(f"  map6/7:  {N_NODES} nodes, {N_EDGES} edges, avg prob ~0.09-0.11")
    print("  map6/7:  Scale-free structure (power-law degree distribution)")
    print("\nTest commands:")
    print(f'  python IEMP_Evol.py -n "{map6_dir}\\dataset6_hard" -i "{map6_dir}\\seed" -b "{map6_dir}\\seed_balanced" -k {budget_6}')
    print(f'  python IEMP_Evol.py -n "{map7_dir}\\dataset7_hard" -i "{map7_dir}\\seed" -b "{map7_dir}\\seed_balanced" -k {budget_7}')


if __name__ == "__main__":
    main()
