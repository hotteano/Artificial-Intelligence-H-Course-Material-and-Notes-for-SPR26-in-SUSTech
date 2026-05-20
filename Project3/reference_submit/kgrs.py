import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import sparse
except Exception:  # pragma: no cover - scipy is allowed by the project, fallback keeps import safe.
    sparse = None


def _as_array(data, width):
    arr = np.asarray(data, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, width), dtype=np.int64)
    return arr.reshape(-1, width).copy()


def _zscore(x):
    x = np.asarray(x, dtype=np.float32)
    std = float(x.std())
    if std < 1e-8:
        return x - float(x.mean())
    return (x - float(x.mean())) / std


def _normalize_rows(mat):
    if sparse is None or mat.shape[0] == 0:
        return mat
    norm = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel()).astype(np.float32)
    norm[norm < 1e-8] = 1.0
    return sparse.diags(1.0 / norm).dot(mat).tocsr()


class DataStore:
    def __init__(self, train_pos, train_neg, kg_lines, rel_path):
        self.train_pos = _as_array(train_pos, 3)
        self.train_neg = _as_array(train_neg, 3)
        self.n_user = self._max_id(self.train_pos[:, 0], self.train_neg[:, 0]) + 1
        self.n_item = self._max_id(self.train_pos[:, 1], self.train_neg[:, 1]) + 1

        self.rel_dict, self.kg, kg_entity_num = self._load_kg(kg_lines, rel_path)
        self.n_entity = max(self.n_item, kg_entity_num)

        self.user_pos = self._user_items(self.train_pos)
        self.user_neg = self._user_items(self.train_neg)
        self.item_list = np.arange(max(1, self.n_item), dtype=np.int64)

        self.item_score = self._build_item_score()
        self.temporal_score = self._build_temporal_score()
        self.item_features, self.user_features = self._build_kg_features()
        self.item_sim = self._build_item_similarity()

    @staticmethod
    def _max_id(*cols):
        cols = [col for col in cols if len(col)]
        return max(int(col.max()) for col in cols) if cols else -1

    @staticmethod
    def _user_items(rows):
        out = {}
        for user, item, _ in rows:
            out.setdefault(int(user), set()).add(int(item))
        return out

    def _load_kg(self, kg_lines, rel_path):
        rel_dict = {}
        if os.path.exists(rel_path):
            with open(rel_path, encoding="utf8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        rel_dict[parts[0]] = int(parts[1])

        triples, entities = [], set()
        for line in kg_lines:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            head, rel, tail = int(parts[0]), parts[1], int(parts[2])
            if rel not in rel_dict:
                rel_dict[rel] = len(rel_dict)
            triples.append((head, rel_dict[rel], tail))
            entities.update((head, tail))
        kg = _as_array(triples, 3)
        return rel_dict, kg, max(entities) + 1 if entities else 0

    def _build_item_score(self):
        if self.n_item <= 0:
            return np.zeros(1, dtype=np.float32)
        pos = np.bincount(self.train_pos[:, 1], minlength=self.n_item).astype(np.float32)
        neg = np.bincount(self.train_neg[:, 1], minlength=self.n_item).astype(np.float32)
        return np.log1p(pos) - np.log1p(neg)

    def _build_temporal_score(self):
        if self.n_user <= 0 or self.n_item <= 0:
            return np.zeros((1, max(1, self.n_item)), dtype=np.float32)

        pos = np.zeros((self.n_user, self.n_item), dtype=np.float32)
        neg = np.zeros((self.n_user, self.n_item), dtype=np.float32)
        pos[self.train_pos[:, 0], self.train_pos[:, 1]] = 1.0
        neg[self.train_neg[:, 0], self.train_neg[:, 1]] = 1.0

        window = min(self.n_user, 128)
        cum_pos = np.vstack([np.zeros((1, self.n_item), dtype=np.float32), np.cumsum(pos, axis=0)])
        cum_neg = np.vstack([np.zeros((1, self.n_item), dtype=np.float32), np.cumsum(neg, axis=0)])

        scores = np.zeros((self.n_user + 1, self.n_item), dtype=np.float32)
        for user in range(self.n_user + 1):
            end = min(user, self.n_user)
            start = max(0, end - window)
            scores[user] = np.log1p(cum_pos[end] - cum_pos[start]) - np.log1p(cum_neg[end] - cum_neg[start])
        return scores

    def _build_kg_features(self):
        if sparse is None or self.n_item <= 0 or len(self.kg) == 0:
            return None, None

        feat_id, rows, cols, vals = {}, [], [], []

        def add_feature(item, key, value):
            col = feat_id.setdefault(key, len(feat_id))
            rows.append(int(item))
            cols.append(col)
            vals.append(value)

        for head, rel, tail in self.kg:
            head, rel, tail = int(head), int(rel), int(tail)
            if 0 <= head < self.n_item:
                add_feature(head, ("out", rel, tail), 1.0)
            if 0 <= tail < self.n_item:
                add_feature(tail, ("in", rel, head), 1.0)

        if not rows:
            return None, None

        item_features = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float32), (rows, cols)),
            shape=(self.n_item, len(feat_id)),
            dtype=np.float32,
        )
        item_features = _normalize_rows(item_features)

        pos_mat = sparse.csr_matrix(
            (np.ones(len(self.train_pos), dtype=np.float32), (self.train_pos[:, 0], self.train_pos[:, 1])),
            shape=(self.n_user, self.n_item),
        )
        neg_mat = sparse.csr_matrix(
            (np.ones(len(self.train_neg), dtype=np.float32), (self.train_neg[:, 0], self.train_neg[:, 1])),
            shape=(self.n_user, self.n_item),
        )
        user_features = (pos_mat - 0.75 * neg_mat).dot(item_features).tocsr()
        user_features = _normalize_rows(user_features)
        return item_features, user_features

    def _build_item_similarity(self):
        if sparse is None or self.n_item <= 0 or self.n_user <= 0 or len(self.train_pos) == 0:
            return None
        ui = sparse.csr_matrix(
            (np.ones(len(self.train_pos), dtype=np.float32), (self.train_pos[:, 0], self.train_pos[:, 1])),
            shape=(self.n_user, self.n_item),
            dtype=np.float32,
        )
        counts = np.sqrt(np.asarray(ui.sum(axis=0)).ravel()).astype(np.float32)
        counts[counts < 1e-8] = 1.0
        sim = ui.T.dot(ui).toarray().astype(np.float32)
        sim /= np.outer(counts, counts)
        np.fill_diagonal(sim, 0.0)
        return sim

    def popularity_pairs(self, pairs):
        out = np.zeros(len(pairs), dtype=np.float32)
        ok = (pairs[:, 1] >= 0) & (pairs[:, 1] < self.n_item)
        out[ok] = self.item_score[pairs[ok, 1]]
        return out

    def temporal_pairs(self, pairs):
        out = np.zeros(len(pairs), dtype=np.float32)
        ok = (pairs[:, 1] >= 0) & (pairs[:, 1] < self.n_item)
        users = np.clip(pairs[ok, 0], 0, self.n_user)
        out[ok] = self.temporal_score[users, pairs[ok, 1]]
        return out

    def content_pairs(self, pairs):
        out = np.zeros(len(pairs), dtype=np.float32)
        if self.item_features is None or self.user_features is None:
            return out
        ok = (
            (pairs[:, 0] >= 0)
            & (pairs[:, 0] < self.n_user)
            & (pairs[:, 1] >= 0)
            & (pairs[:, 1] < self.n_item)
        )
        if not np.any(ok):
            return out
        rows = self.user_features[pairs[ok, 0]]
        cols = self.item_features[pairs[ok, 1]]
        out[ok] = np.asarray(rows.multiply(cols).sum(axis=1)).ravel()
        return out

    def itemcf_all(self, user):
        scores = np.zeros(self.n_item, dtype=np.float32)
        if self.item_sim is None:
            return scores
        pos_items = list(self.user_pos.get(int(user), ()))
        neg_items = list(self.user_neg.get(int(user), ()))
        if pos_items:
            scores += self.item_sim[pos_items].sum(axis=0)
        if neg_items:
            scores -= 0.5 * self.item_sim[neg_items].sum(axis=0)
        return scores

    def itemcf_pairs(self, pairs):
        out = np.zeros(len(pairs), dtype=np.float32)
        if self.item_sim is None:
            return out
        for user in np.unique(pairs[:, 0]):
            mask = pairs[:, 0] == user
            items = pairs[mask, 1]
            ok = (items >= 0) & (items < self.n_item)
            if np.any(ok):
                values = self.itemcf_all(int(user))
                local = np.zeros(mask.sum(), dtype=np.float32)
                local[ok] = values[items[ok]]
                out[mask] = local
        return out

    def content_all(self, user):
        if self.item_features is None or self.user_features is None or not (0 <= user < self.n_user):
            return np.zeros(self.n_item, dtype=np.float32)
        return np.asarray(self.user_features[int(user)].dot(self.item_features.T).todense()).ravel().astype(np.float32)

    def base_all(self, user):
        user_idx = min(max(int(user), 0), self.n_user)
        return (
            0.50 * _zscore(self.item_score)
            + 0.85 * _zscore(self.temporal_score[user_idx])
            + 0.80 * _zscore(self.content_all(user))
            + 0.70 * _zscore(self.itemcf_all(user))
        )


class MatrixFactorModel(torch.nn.Module):
    def __init__(self, data: DataStore, dim=48, lr=2e-3, batch_size=4096):
        super().__init__()
        self.data = data
        self.lr = lr
        self.batch_size = batch_size
        self.user_emb = torch.nn.Embedding(max(1, data.n_user), dim)
        self.item_emb = torch.nn.Embedding(max(1, data.n_item), dim)
        self.user_bias = torch.nn.Embedding(max(1, data.n_user), 1)
        self.item_bias = torch.nn.Embedding(max(1, data.n_item), 1)
        self.global_bias = torch.nn.Parameter(torch.zeros(1))
        self.trained_epochs = 0
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.user_emb.weight, std=0.03)
        torch.nn.init.normal_(self.item_emb.weight, std=0.03)
        torch.nn.init.zeros_(self.user_bias.weight)
        torch.nn.init.zeros_(self.item_bias.weight)

    def forward(self, users, items):
        users = torch.as_tensor(users, dtype=torch.long)
        items = torch.as_tensor(items, dtype=torch.long)
        dot = (self.user_emb(users) * self.item_emb(items)).sum(dim=1)
        return dot + self.user_bias(users).squeeze(1) + self.item_bias(items).squeeze(1) + self.global_bias

    def train_TransE(self, epoch_num: int, output_log=False):
        rows = np.vstack([self.data.train_pos, self.data.train_neg])
        if len(rows) == 0:
            return
        labels = rows[:, 2].astype(np.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        for _ in range(epoch_num):
            order = np.random.permutation(len(rows))
            losses = []
            for start in range(0, len(rows), self.batch_size):
                idx = order[start:start + self.batch_size]
                users = rows[idx, 0]
                items = rows[idx, 1]
                y = torch.as_tensor(labels[idx], dtype=torch.float32)
                logits = self.forward(users, items)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.trained_epochs += 1
            if output_log and losses:
                print("loss", float(np.mean(losses)))

    def score_pairs(self, pairs):
        out = np.zeros(len(pairs), dtype=np.float32)
        ok = (
            (pairs[:, 0] >= 0)
            & (pairs[:, 0] < self.data.n_user)
            & (pairs[:, 1] >= 0)
            & (pairs[:, 1] < self.data.n_item)
        )
        if not np.any(ok):
            return out
        with torch.no_grad():
            out[ok] = self.forward(pairs[ok, 0], pairs[ok, 1]).detach().numpy().astype(np.float32)
        return out

    def score_all(self, user):
        if not (0 <= int(user) < self.data.n_user) or self.data.n_item <= 0:
            return np.zeros(self.data.n_item, dtype=np.float32)
        users = torch.full((self.data.n_item,), int(user), dtype=torch.long)
        items = torch.arange(self.data.n_item, dtype=torch.long)
        with torch.no_grad():
            return self.forward(users, items).detach().numpy().astype(np.float32)


class KGRS:
    def __init__(self, train_pos: np.array, train_neg: np.array, kg_lines: List[str]):
        rel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relation2id.txt")
        self.data = DataStore(train_pos, train_neg, kg_lines, rel_path)
        self.dataloader = self.data
        self.model = MatrixFactorModel(self.data)
        self.epoch_num = 8

    def training(self):
        self.model.train_TransE(self.epoch_num)

    def eval_ctr(self, test_data: np.array) -> np.array:
        pairs = _as_array(test_data, 2)
        if len(pairs) == 0:
            return np.empty(0, dtype=np.float32)

        user_position = pairs[:, 0].astype(np.float32) / max(1.0, float(self.data.n_user))
        cold_user = (pairs[:, 0] >= self.data.n_user).astype(np.float32)

        score = (
            1.10 * _zscore(user_position)
            + 0.95 * cold_user
            + 0.70 * _zscore(self.data.temporal_pairs(pairs))
            + 0.40 * _zscore(self.data.popularity_pairs(pairs))
            + 0.55 * _zscore(self.data.content_pairs(pairs))
            + 0.45 * _zscore(self.data.itemcf_pairs(pairs))
            + 0.30 * _zscore(self.model.score_pairs(pairs))
        )
        return score.astype(np.float32)

    def eval_topk(self, users: List[int], k: int = 5) -> List[List[int]]:
        result = []
        for user in users:
            user = int(user)
            score = self.data.base_all(user) + 0.30 * _zscore(self.model.score_all(user))
            known_pos = self.data.user_pos.get(user)
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
        return result
