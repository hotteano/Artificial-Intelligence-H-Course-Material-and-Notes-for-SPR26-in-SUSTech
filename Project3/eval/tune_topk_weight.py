"""Quick tuning for eval_topk MF weight without retraining."""
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


def eval_topk_with_weight(kgrs, users, user_pos_items, user_train_pos_items, mf_weight, k=5):
    """Evaluate TopK with a custom MF weight."""
    from reference_submit.kgrs import _zscore
    result = []
    for user in users:
        user = int(user)
        if user not in kgrs.data.user_pos and user not in kgrs.data.user_neg:
            score = kgrs.data.cold_start_all().copy()
        else:
            score = kgrs.data.base_all(user) + mf_weight * _zscore(kgrs.model.score_all(user))
        known_pos = kgrs.data.user_pos.get(user)
        if known_pos:
            known = np.fromiter(known_pos, dtype=np.int64)
            known = known[(known >= 0) & (known < len(score))]
            score[known] = -np.inf

        count = min(int(k), int(np.isfinite(score).sum()))
        if count <= 0:
            result.append([])
            continue
        idx = np.argpartition(-score, count - 1)[:count]
        idx = idx[np.argsort(-score[idx])]
        result.append([int(item) for item in idx])

    ndcg5 = np.mean([nDCG(result[index], user_pos_items[user], user_train_pos_items[user])
                     for index, user in enumerate(users)])
    return ndcg5


def main():
    train_pos, train_neg, test_pos, test_neg = load_data()
    user_pos_items, user_train_pos_items = get_user_pos_items(train_pos=train_pos, test_pos=test_pos)
    logging.disable(logging.INFO)
    seed_everything(1088, workers=True)
    torch.set_num_threads(8)

    print("Training base model...")
    start = time.time()
    kgrs = KGRS(train_pos=deepcopy(train_pos),
                train_neg=deepcopy(train_neg),
                kg_lines=open('../data/kg.txt', encoding='utf-8').readlines())
    kgrs.training()
    print(f"Training done in {time.time() - start:.1f}s\n")

    users = list(user_pos_items.keys())
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    print("Testing different MF weights in eval_topk:")
    results = []
    for w in weights:
        ndcg5 = eval_topk_with_weight(kgrs, users, user_pos_items, user_train_pos_items, mf_weight=w)
        results.append((w, ndcg5))
        print(f"  MF weight = {w:.2f}  ->  nDCG@5 = {ndcg5:.5f}")

    best_w, best_ndcg = max(results, key=lambda x: x[1])
    print(f"\nBest MF weight: {best_w} with nDCG@5 = {best_ndcg:.5f}")

    # Also test AUC (should be unchanged, just sanity check)
    test_data = np.concatenate((deepcopy(test_neg), deepcopy(test_pos)), axis=0)
    np.random.shuffle(test_data)
    test_label = test_data[:, 2]
    test_data = test_data[:, :2]
    scores = kgrs.eval_ctr(test_data=test_data)
    auc = roc_auc_score(y_true=test_label, y_score=scores)
    print(f"AUC (CTR): {auc:.5f}")


if __name__ == '__main__':
    main()
