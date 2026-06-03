"""Tune embedding dimension (model capacity)."""
import logging
import sys
import time
from collections import defaultdict
from copy import deepcopy

import math
import numpy as np
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '..')
from reference_submit.kgrs import KGRS


def nDCG(sorted_items, pos_item, train_pos_item, k=5):
    dcg = 0
    train_pos_item = set(train_pos_item)
    filter_item = set(filter(lambda item: item not in train_pos_item, pos_item))
    max_correct = min(len(filter_item), k)
    train_hit_num = 0
    valid_num = 0
    recommended_items = set()
    for index in range(len(sorted_items)):
        if sorted_items[index] in train_pos_item:
            train_hit_num += 1
        else:
            valid_num += 1
            if sorted_items[index] in filter_item and sorted_items[index] not in recommended_items:
                dcg += 1 / math.log2(index - train_hit_num + 2)
                recommended_items.add(sorted_items[index])
            if valid_num >= k:
                break
    idcg = sum([1 / math.log2(i + 2) for i in range(max_correct)])
    return dcg / idcg if idcg > 0 else 0.0


def split_by_user(records, train_ratio=0.8, seed=1088):
    rng = np.random.default_rng(seed)
    user_records = defaultdict(list)
    for row in records:
        user_records[int(row[0])].append(row)

    train_parts, remaining_parts = [], []
    for rows in user_records.values():
        rows = np.asarray(rows, dtype=records.dtype)
        rng.shuffle(rows)
        train_parts.append(rows[:1])
        if len(rows) > 1:
            remaining_parts.append(rows[1:])

    empty = np.empty((0, 3), dtype=records.dtype)
    reserved_train = np.vstack(train_parts) if train_parts else empty
    remaining = np.vstack(remaining_parts) if remaining_parts else empty

    rng.shuffle(remaining)
    target_train_size = int(len(records) * train_ratio)
    extra_train_size = max(0, min(len(remaining), target_train_size - len(reserved_train)))
    train_extra = remaining[:extra_train_size]
    test = remaining[extra_train_size:]
    train = np.vstack([reserved_train, train_extra]) if len(train_extra) else reserved_train

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def load_data():
    full_pos = np.load("../data/train_pos.npy")
    full_neg = np.load("../data/train_neg.npy")
    train_pos, test_pos = split_by_user(full_pos, train_ratio=0.8, seed=1088)
    train_neg, test_neg = split_by_user(full_neg, train_ratio=0.8, seed=1089)
    return train_pos, train_neg, test_pos, test_neg


def get_user_pos_items(train_pos, test_pos):
    user_pos_items, user_train_pos_items = {}, {}
    for record in train_pos:
        user, item = record[0], record[1]
        if user not in user_train_pos_items:
            user_train_pos_items[user] = set()
        user_train_pos_items[user].add(item)
    for record in test_pos:
        user, item = record[0], record[1]
        if user not in user_train_pos_items:
            user_train_pos_items[user] = set()
        if user not in user_pos_items:
            user_pos_items[user] = set()
        user_pos_items[user].add(item)
    return user_pos_items, user_train_pos_items


def evaluate_one(dim, epoch_num=20, lr=2e-3, bpr_weight=0.7):
    train_pos, train_neg, test_pos, test_neg = load_data()
    user_pos_items, user_train_pos_items = get_user_pos_items(train_pos=train_pos, test_pos=test_pos)
    logging.disable(logging.INFO)
    seed_everything(1088, workers=True)
    torch.set_num_threads(8)

    start = time.time()
    kgrs = KGRS(train_pos=deepcopy(train_pos),
                train_neg=deepcopy(train_neg),
                kg_lines=open('../data/kg.txt', encoding='utf-8').readlines(),
                dim=dim, lr=lr, bpr_weight=bpr_weight, epoch_num=epoch_num)
    kgrs.training()
    train_elapsed = time.time() - start

    # CTR AUC
    test_data = np.concatenate((deepcopy(test_neg), deepcopy(test_pos)), axis=0)
    np.random.shuffle(test_data)
    test_label = test_data[:, 2]
    test_data = test_data[:, :2]
    scores = kgrs.eval_ctr(test_data=test_data)
    auc = roc_auc_score(y_true=test_label, y_score=scores)

    # TopK nDCG@5
    users = list(user_pos_items.keys())
    user_item_lists = kgrs.eval_topk(users=users)
    ndcg5 = np.mean([nDCG(user_item_lists[index], user_pos_items[user], user_train_pos_items[user])
                     for index, user in enumerate(users)])

    return auc, ndcg5, train_elapsed


def main():
    dims = [32, 48, 64, 80, 96, 128]
    print("Tuning embedding dimension (original baseline dim=48):")
    print(f"{'dim':>5} | {'AUC':>8} | {'nDCG@5':>8} | {'time(s)':>8}")
    print("-" * 40)
    results = []
    for dim in dims:
        auc, ndcg5, t = evaluate_one(dim=dim)
        results.append((dim, auc, ndcg5, t))
        print(f"{dim:>5} | {auc:>8.5f} | {ndcg5:>8.5f} | {t:>8.1f}")

    best_auc = max(results, key=lambda x: x[1])
    best_ndcg = max(results, key=lambda x: x[2])
    print(f"\nBest AUC   : dim={best_auc[0]}, AUC={best_auc[1]:.5f}")
    print(f"Best nDCG@5: dim={best_ndcg[0]}, nDCG@5={best_ndcg[2]:.5f}")


if __name__ == '__main__':
    main()
