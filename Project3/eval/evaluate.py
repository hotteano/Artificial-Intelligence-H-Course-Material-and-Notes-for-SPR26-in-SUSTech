import logging
import time
from collections import defaultdict
from copy import deepcopy

import math
import numpy as np
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_auc_score

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
                dcg += 1 / math.log2(index - train_hit_num + 2)  # Rank starts from 0
                recommended_items.add(sorted_items[index])
            if valid_num >= k:
                break
    idcg = sum([1 / math.log2(i + 2) for i in range(max_correct)])
    return dcg / idcg


def split_by_user(records, train_ratio=0.8, seed=1088):
    rng = np.random.default_rng(seed)
    user_records = defaultdict(list)
    for row in records:
        user_records[int(row[0])].append(row)

    train_parts, remaining_parts = [], []
    for rows in user_records.values():
        rows = np.asarray(rows, dtype=records.dtype)
        rng.shuffle(rows)

        # Keep every user visible in training; split the remaining records globally
        # so the final train size stays close to the requested 80%.
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


def evaluate():
    train_pos, train_neg, test_pos, test_neg = load_data()
    user_pos_items, user_train_pos_items = get_user_pos_items(train_pos=train_pos, test_pos=test_pos)
    logging.disable(logging.INFO)
    seed_everything(1088, workers=True)
    torch.set_num_threads(8)
    auc, ndcg5 = 0, 0
    init_timeout, train_timeout, ctr_timeout, topk_timeout = False, False, False, False
    start_time, init_time, train_time, ctr_time, topk_time = time.time(), 0, 0, 0, 0
    kgrs = KGRS(train_pos=deepcopy(train_pos),
                train_neg=deepcopy(train_neg),
                kg_lines=open('../data/kg.txt', encoding='utf-8').readlines())
    init_time = time.time() - start_time

    kgrs.training()
    train_time = time.time() - start_time - init_time

    test_data = np.concatenate((deepcopy(test_neg), deepcopy(test_pos)), axis=0)
    np.random.shuffle(test_data)
    test_label = test_data[:, 2]
    test_data = test_data[:, :2]
    kgrs.eval_ctr(test_data)
    scores = kgrs.eval_ctr(test_data=test_data)
    auc = roc_auc_score(y_true=test_label, y_score=scores)
    ctr_time = time.time() - start_time - init_time - train_time

    users = list(user_pos_items.keys())
    user_item_lists = kgrs.eval_topk(users=users)
    ndcg5 = np.mean([nDCG(user_item_lists[index], user_pos_items[user], user_train_pos_items[user]) for index, user in
                     enumerate(users)])

    topk_time = time.time() - start_time - init_time - train_time - ctr_time

    return auc, ndcg5, init_timeout, train_timeout, ctr_timeout, topk_timeout, init_time, train_time, ctr_time, topk_time


if __name__ == '__main__':
    start = time.time()
    print(evaluate())
    print(time.time() - start)
