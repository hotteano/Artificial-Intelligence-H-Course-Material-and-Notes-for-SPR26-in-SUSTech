import argparse
import importlib.util
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_kgrs_class(path: str):
    """Dynamic import to avoid class name collision between demo and reference_submit."""
    module_name = f"kgrs_compare_{os.path.basename(os.path.dirname(path))}"
    spec = importlib.util.spec_from_file_location(module_name, os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.KGRS


def load_data(data_dir: str, seed: int = 1088):
    """Load data and split per-user into train/val/test (8:1:1)."""
    full_pos = np.load(os.path.join(data_dir, "train_pos.npy"))
    full_neg = np.load(os.path.join(data_dir, "train_neg.npy"))

    rng = np.random.default_rng(seed)

    # Group by user
    user_pos = defaultdict(list)
    user_neg = defaultdict(list)
    for row in full_pos:
        user_pos[int(row[0])].append(row)
    for row in full_neg:
        user_neg[int(row[0])].append(row)

    all_users = sorted(set(user_pos.keys()) | set(user_neg.keys()))
    rng.shuffle(all_users)

    def split_records(records):
        arr = np.array(records, dtype=np.int32)
        rng.shuffle(arr)
        n = len(arr)
        i1 = int(n * 0.8)
        i2 = int(n * 0.9)
        return arr[:i1], arr[i1:i2], arr[i2:]

    train_pos_list, val_pos_list, test_pos_list = [], [], []
    train_neg_list, val_neg_list, test_neg_list = [], [], []

    for u in all_users:
        tp, vp, sp = split_records(user_pos.get(u, []))
        tn, vn, sn = split_records(user_neg.get(u, []))
        train_pos_list.append(tp)
        val_pos_list.append(vp)
        test_pos_list.append(sp)
        train_neg_list.append(tn)
        val_neg_list.append(vn)
        test_neg_list.append(sn)

    def concat(parts):
        parts = [p for p in parts if len(p) > 0]
        return np.vstack(parts) if parts else np.empty((0, 3), dtype=np.int32)

    return (
        concat(train_pos_list), concat(train_neg_list),
        concat(val_pos_list), concat(val_neg_list),
        concat(test_pos_list), concat(test_neg_list),
    )


def get_user_pos_items(pos_array: np.ndarray):
    """Return dict mapping user -> set of positive items."""
    d = defaultdict(set)
    if len(pos_array) == 0:
        return d
    for record in pos_array:
        d[int(record[0])].add(int(record[1]))
    return d


def ndcg_at_k(sorted_items, pos_items, k=5):
    """Compute nDCG@k for a single user."""
    pos_items = set(pos_items)
    if not pos_items:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(sorted_items[:k]):
        if item in pos_items:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(pos_items), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(sorted_items, pos_items, k=5):
    return int(len(set(sorted_items[:k]) & set(pos_items)) > 0)


def precision_at_k(sorted_items, pos_items, k=5):
    if k == 0:
        return 0.0
    return len(set(sorted_items[:k]) & set(pos_items)) / k


def recall_at_k(sorted_items, pos_items, k=5):
    pos_items = set(pos_items)
    if not pos_items:
        return 0.0
    return len(set(sorted_items[:k]) & pos_items) / len(pos_items)


def evaluate_ctr(kgrs, test_data, test_labels):
    """Evaluate CTR and return AUC."""
    if len(test_data) == 0:
        return 0.0
    scores = kgrs.eval_ctr(test_data)
    return roc_auc_score(y_true=test_labels, y_score=scores)


def evaluate_topk(kgrs, users, test_user_pos, train_user_pos, k=5):
    """Evaluate top-k recommendation and return dict of metrics."""
    if not users:
        return {"ndcg": 0.0, "hit": 0.0, "precision": 0.0, "recall": 0.0}

    topk_lists = kgrs.eval_topk(users=users, k=k)

    ndcgs, hits, precisions, recalls = [], [], [], []
    for user, recs in zip(users, topk_lists):
        test_pos = test_user_pos.get(user, set())
        if not test_pos:
            continue
        ndcgs.append(ndcg_at_k(recs, test_pos, k))
        hits.append(hit_at_k(recs, test_pos, k))
        precisions.append(precision_at_k(recs, test_pos, k))
        recalls.append(recall_at_k(recs, test_pos, k))

    def mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    return {
        "ndcg": mean(ndcgs),
        "hit": mean(hits),
        "precision": mean(precisions),
        "recall": mean(recalls),
    }


def run_single(kgrs_path, kg_path, label, data_splits, k=5, seed=1088,
               max_epochs=30, patience=5):
    print(f"\n{'=' * 60}")
    print(f"  Model: {label}")
    print(f"  Source: {kgrs_path}")
    print(f"{'=' * 60}")

    KGRS = load_kgrs_class(kgrs_path)
    train_pos, train_neg, val_pos, val_neg, test_pos, test_neg = data_splits

    seed_everything(seed, workers=True)
    torch.set_num_threads(8)

    t0 = time.perf_counter()

    with open(kg_path, encoding="utf-8") as f:
        kg_lines = f.readlines()

    kgrs = KGRS(
        train_pos=train_pos,
        train_neg=train_neg,
        kg_lines=kg_lines,
    )
    t1 = time.perf_counter()
    init_time = t1 - t0

    # Early stopping based on validation AUC
    best_val_auc = -1.0
    best_state = None
    no_improve = 0

    for epoch in tqdm(range(1, max_epochs + 1), desc=f"{label} Training", leave=False):
        kgrs.model.train_TransE(epoch_num=1, output_log=False)

        # Validation
        if len(val_pos) > 0 and len(val_neg) > 0:
            val_data = np.concatenate((val_neg, val_pos), axis=0)
            rng = np.random.default_rng(seed + epoch)
            rng.shuffle(val_data)
            val_labels = val_data[:, 2]
            val_data = val_data[:, :2]
            val_scores = kgrs.eval_ctr(val_data)
            val_auc = roc_auc_score(y_true=val_labels, y_score=val_scores)
            tqdm.write(f"  Epoch {epoch:02d} | Val AUC: {val_auc:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in kgrs.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    tqdm.write(f"  Early stopping at epoch {epoch} (best val AUC: {best_val_auc:.4f})")
                    break
    t2 = time.perf_counter()
    train_time = t2 - t1

    if best_state is not None:
        kgrs.model.load_state_dict(best_state)

    # ---- Test CTR ----
    test_data = np.concatenate((test_neg, test_pos), axis=0)
    rng = np.random.default_rng(seed + 9999)
    rng.shuffle(test_data)
    test_labels = test_data[:, 2]
    test_data = test_data[:, :2]
    test_auc = evaluate_ctr(kgrs, test_data, test_labels)
    t3 = time.perf_counter()
    ctr_time = t3 - t2

    # ---- Test Top-K ----
    train_user_pos = get_user_pos_items(train_pos)
    test_user_pos = get_user_pos_items(test_pos)
    # Only evaluate users that have test positive items
    eval_users = [u for u in test_user_pos if test_user_pos[u]]
    topk_metrics = evaluate_topk(kgrs, eval_users, test_user_pos, train_user_pos, k=k)
    t4 = time.perf_counter()
    topk_time = t4 - t3
    total_time = t4 - t0

    return {
        "label": label,
        "auc": float(test_auc),
        "ndcg": float(topk_metrics["ndcg"]),
        "hit": float(topk_metrics["hit"]),
        "precision": float(topk_metrics["precision"]),
        "recall": float(topk_metrics["recall"]),
        "init_time": init_time,
        "train_time": train_time,
        "ctr_time": ctr_time,
        "topk_time": topk_time,
        "total_time": total_time,
    }


def print_comparison(results):
    print(f"\n{'=' * 90}")
    print("  RESULT COMPARISON")
    print(f"{'=' * 90}")
    header = f"{'Model':<22} {'AUC':>10} {'nDCG@5':>10} {'Hit@5':>10} {'Prec@5':>10} {'Rec@5':>10} {'Train(s)':>10}"
    print(header)
    print("-" * 90)
    for r in results:
        print(
            f"{r['label']:<22} "
            f"{r['auc']:>10.4f} "
            f"{r['ndcg']:>10.4f} "
            f"{r['hit']:>10.4f} "
            f"{r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} "
            f"{r['train_time']:>10.1f}"
        )
    print(f"{'=' * 90}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare KGRS implementations with per-user 8:1:1 split.")
    parser.add_argument("--demo-path", default=os.path.join(PROJECT_ROOT, "demo", "kgrs.py"))
    parser.add_argument("--ref-path", default=os.path.join(PROJECT_ROOT, "reference_submit", "kgrs.py"))
    parser.add_argument("--kg-path", default=os.path.join(PROJECT_ROOT, "data", "kg.txt"))
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--seed", type=int, default=1088)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    data_splits = load_data(args.data_dir, seed=args.seed)
    train_pos, train_neg, val_pos, val_neg, test_pos, test_neg = data_splits
    print(f"Data loaded: train_pos={len(train_pos)}, train_neg={len(train_neg)}, "
          f"val_pos={len(val_pos)}, val_neg={len(val_neg)}, "
          f"test_pos={len(test_pos)}, test_neg={len(test_neg)}")

    baseline = run_single(
        args.demo_path,
        args.kg_path,
        "TransE (demo)",
        data_splits,
        k=args.k,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    rotate = run_single(
        args.ref_path,
        args.kg_path,
        "RotatE (ours)",
        data_splits,
        k=args.k,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    print_comparison([baseline, rotate])


if __name__ == "__main__":
    main()
