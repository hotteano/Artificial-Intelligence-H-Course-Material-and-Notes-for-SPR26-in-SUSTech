"""Generate community-structured graph for Heuristic/map5.

Uses Stochastic Block Model (SBM) with high intra-community edges 
and low inter-community edges to create realistic community structure.

Specs: 3454 nodes, 32140 edges (same as original map5)
"""
import os
import random
import numpy as np


def generate_community_graph(n_nodes, n_edges, n_communities, seed=42):
    """
    Generate a graph with strong community structure.
    
    Strategy:
    - Divide nodes into communities
    - 85% of edges within communities (high density)
    - 15% of edges between communities (sparse connections)
    """
    rng = np.random.default_rng(seed)
    
    # Assign nodes to communities
    community_ids = np.arange(n_nodes) % n_communities
    rng.shuffle(community_ids)
    
    # Community sizes
    community_sizes = np.bincount(community_ids, minlength=n_communities)
    print(f"  Communities: {n_communities}")
    print(f"  Community sizes: min={community_sizes.min()}, max={community_sizes.max()}, avg={community_sizes.mean():.0f}")
    
    # Edge allocation: 85% intra, 15% inter
    intra_edges_target = int(n_edges * 0.85)
    inter_edges_target = n_edges - intra_edges_target
    
    edges_set = set()
    
    # Generate intra-community edges (dense within communities)
    for comm_id in range(n_communities):
        comm_nodes = np.where(community_ids == comm_id)[0]
        n_comm_nodes = len(comm_nodes)
        
        if n_comm_nodes < 2:
            continue
            
        # Target edges for this community (proportional to size)
        comm_edge_share = (n_comm_nodes / n_nodes) * intra_edges_target
        comm_edges = min(int(comm_edge_share), n_comm_nodes * (n_comm_nodes - 1))
        
        # Generate edges within this community
        attempts = 0
        max_attempts = comm_edges * 5
        while len([e for e in edges_set if community_ids[e[0]] == comm_id and community_ids[e[1]] == comm_id]) < comm_edges and attempts < max_attempts:
            u = rng.choice(comm_nodes)
            v = rng.choice(comm_nodes)
            if u != v and (u, v) not in edges_set:
                edges_set.add((u, v))
            attempts += 1
    
    # Generate inter-community edges (sparse between communities)
    inter_edges_generated = len([e for e in edges_set if community_ids[e[0]] != community_ids[e[1]]])
    attempts = 0
    max_attempts = inter_edges_target * 20
    
    while inter_edges_generated < inter_edges_target and attempts < max_attempts:
        u = rng.integers(0, n_nodes)
        v = rng.integers(0, n_nodes)
        if u != v and community_ids[u] != community_ids[v] and (u, v) not in edges_set:
            edges_set.add((u, v))
            inter_edges_generated += 1
        attempts += 1
    
    edges = list(edges_set)
    
    # If we don't have enough edges, fill randomly
    while len(edges) < n_edges:
        u = rng.integers(0, n_nodes)
        v = rng.integers(0, n_nodes)
        if u != v and (u, v) not in edges_set:
            edges_set.add((u, v))
            edges.append((u, v))
    
    # Generate probabilities (higher within communities, lower between)
    edges_with_probs = []
    for u, v in edges:
        if community_ids[u] == community_ids[v]:
            # Intra-community: higher probability (easier to spread within community)
            p1 = round(rng.uniform(0.04, 0.10), 4)
            p2 = round(rng.uniform(0.04, 0.10), 4)
        else:
            # Inter-community: lower probability (harder to bridge communities)
            p1 = round(rng.uniform(0.01, 0.04), 4)
            p2 = round(rng.uniform(0.01, 0.04), 4)
        edges_with_probs.append((u, v, p1, p2))
    
    # Statistics
    intra_count = sum(1 for u, v, _, _ in edges_with_probs if community_ids[u] == community_ids[v])
    inter_count = len(edges_with_probs) - intra_count
    
    return n_nodes, len(edges_with_probs), edges_with_probs, community_ids, intra_count, inter_count


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
    
    # Same specs as original map5
    N_NODES = 3454
    N_EDGES = 32140
    N_COMMUNITIES = 15  # Number of communities
    
    # Same seeds as original
    I1 = [3332, 325, 1161, 1232, 2617, 980]
    I2 = [1222, 2447, 3319, 1511, 3386, 1833, 2781, 3359]
    
    print("="*70)
    print("Generating Community-Structured Map5")
    print(f"Specs: {N_NODES} nodes, {N_EDGES} edges, {N_COMMUNITIES} communities")
    print("="*70)
    
    map5_dir = os.path.join(base_dir, "map5")
    os.makedirs(map5_dir, exist_ok=True)
    
    # Backup original
    original_dataset = os.path.join(map5_dir, "dataset4")
    backup_dataset = os.path.join(map5_dir, "dataset4_original")
    if os.path.exists(original_dataset) and not os.path.exists(backup_dataset):
        print("  Backing up original dataset4 to dataset4_original...")
        import shutil
        shutil.copy2(original_dataset, backup_dataset)
    
    # Generate community graph
    print("\nGenerating graph...")
    n_nodes, n_edges, edges, comm_ids, intra_cnt, inter_cnt = generate_community_graph(
        N_NODES, N_EDGES, N_COMMUNITIES, seed=200
    )
    
    # Calculate modularity-related metrics
    avg_degree = n_edges / n_nodes
    intra_ratio = intra_cnt / n_edges
    
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {n_edges}")
    print(f"  Average degree: {avg_degree:.2f}")
    print(f"  Intra-community edges: {intra_cnt} ({intra_ratio*100:.1f}%)")
    print(f"  Inter-community edges: {inter_cnt} ({(1-intra_ratio)*100:.1f}%)")
    
    # Probability statistics
    intra_probs_p1 = [p1 for u, v, p1, p2 in edges if comm_ids[u] == comm_ids[v]]
    inter_probs_p1 = [p1 for u, v, p1, p2 in edges if comm_ids[u] != comm_ids[v]]
    
    print(f"\nPropagation Probabilities:")
    print(f"  Intra-community p1: avg={sum(intra_probs_p1)/len(intra_probs_p1):.4f}")
    print(f"  Inter-community p1: avg={sum(inter_probs_p1)/len(inter_probs_p1):.4f}")
    
    # Write files
    write_dataset(os.path.join(map5_dir, "dataset4"), n_nodes, n_edges, edges)
    write_seed_file(os.path.join(map5_dir, "seed"), I1, I2)
    
    print(f"\nFiles saved:")
    print(f"  Dataset: {map5_dir}\\dataset4")
    print(f"  Seed: {map5_dir}\\seed")
    if os.path.exists(backup_dataset):
        print(f"  Original backup: {map5_dir}\\dataset4_original")
    
    print("\n" + "="*70)
    print("Community-structured map5 generated successfully!")
    print("="*70)
    print("\nCharacteristics:")
    print("  - Strong community structure (85% intra-community edges)")
    print("  - Higher propagation probability within communities")
    print("  - Lower probability between communities")
    print("  - This creates 'information silos' - harder to balance exposure")


if __name__ == "__main__":
    main()
