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

    def cold_start_all(self):
        return _zscore(self.temporal_score[self.n_user])


class MatrixFactorModel(torch.nn.Module):
    def __init__(self, data: DataStore, dim=48, lr=2e-3, batch_size=4096, bpr_weight=0.7):
        super().__init__()
        self.data = data
        self.lr = lr
        self.batch_size = batch_size
        self.bpr_weight = bpr_weight
        self.user_emb = torch.nn.Embedding(max(1, data.n_user), dim)
        self.item_emb = torch.nn.Embedding(max(1, data.n_item), dim)
        self.user_bias = torch.nn.Embedding(max(1, data.n_user), 1)
        self.item_bias = torch.nn.Embedding(max(1, data.n_item), 1)
        self.global_bias = torch.nn.Parameter(torch.zeros(1))
        self.trained_epochs = 0
        self.user_neg_array = {user: np.asarray(sorted(items), dtype=np.int64) for user, items in data.user_neg.items()}
        self.user_pos_array = {user: np.asarray(sorted(items), dtype=np.int64) for user, items in data.user_pos.items()}
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
        pos_rows = self.data.train_pos
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        for _ in range(epoch_num):
            order = np.random.permutation(len(rows))
            pos_order = np.random.permutation(len(pos_rows)) if len(pos_rows) else np.empty(0, dtype=np.int64)
            losses = []
            for start in range(0, len(rows), self.batch_size):
                idx = order[start:start + self.batch_size]
                users = rows[idx, 0]
                items = rows[idx, 1]
                y = torch.as_tensor(labels[idx], dtype=torch.float32)
                logits = self.forward(users, items)
                bce_loss = F.binary_cross_entropy_with_logits(logits, y)

                bpr_loss = torch.zeros((), dtype=torch.float32)
                pos_idx = pos_order[start:start + self.batch_size]
                if len(pos_idx) > 0:
                    pos_batch = pos_rows[pos_idx]
                    bpr_users = pos_batch[:, 0]
                    bpr_pos_items = pos_batch[:, 1]
                    bpr_neg_items = self._sample_bpr_negatives(bpr_users)
                    pos_score = self.forward(bpr_users, bpr_pos_items)
                    neg_score = self.forward(bpr_users, bpr_neg_items)
                    bpr_loss = F.softplus(neg_score - pos_score).mean()

                loss = bce_loss + self.bpr_weight * bpr_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            self.trained_epochs += 1
            if output_log and losses:
                print("loss", float(np.mean(losses)))

    def _sample_bpr_negatives(self, users):
        neg_items = np.empty(len(users), dtype=np.int64)
        for idx, user in enumerate(users):
            user = int(user)
            known_neg = self.user_neg_array.get(user)
            if known_neg is not None and len(known_neg) > 0:
                neg_items[idx] = int(known_neg[np.random.randint(len(known_neg))])
                continue

            known_pos = self.user_pos_array.get(user)
            pos_set = set(known_pos.tolist()) if known_pos is not None else set()
            item = int(np.random.randint(max(1, self.data.n_item)))
            retry = 0
            while item in pos_set and retry < 20:
                item = int(np.random.randint(max(1, self.data.n_item)))
                retry += 1
            neg_items[idx] = item
        return neg_items

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
        self.rel_path = rel_path
        self.kg_lines = list(kg_lines)
        self.data = DataStore(train_pos, train_neg, kg_lines, rel_path)
        self.dataloader = self.data
        self.model = MatrixFactorModel(self.data)
        self.epoch_num = 20
        self.ctr_weights = np.asarray([1.00, 0.40, 0.35, 0.30, 0.25], dtype=np.float32)
        self.ctr_bias = 0.0
        self.ctr_mean = np.zeros(len(self.ctr_weights), dtype=np.float32)
        self.ctr_std = np.ones(len(self.ctr_weights), dtype=np.float32)
        self.use_learned_ctr = False

    def training(self):
        self.model.train_TransE(self.epoch_num)
        self._fit_ctr_fusion()

    @staticmethod
    def _auc_score(labels, scores):
        labels = np.asarray(labels, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        pos = labels > 0.5
        neg = ~pos
        pos_num, neg_num = int(pos.sum()), int(neg.sum())
        if pos_num == 0 or neg_num == 0:
            return 0.5
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float32)
        ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float32)
        return float((ranks[pos].sum() - pos_num * (pos_num + 1) / 2.0) / (pos_num * neg_num))

    @staticmethod
    def _split_for_fusion(rows, rng, train_ratio=0.8):
        rows = _as_array(rows, 3)
        if len(rows) < 5:
            return rows, np.empty((0, 3), dtype=np.int64)
        order = rng.permutation(len(rows))
        cut = int(len(rows) * train_ratio)
        cut = min(max(cut, 1), len(rows) - 1)
        return rows[order[:cut]], rows[order[cut:]]

    @staticmethod
    def _feature_matrix(data, model, pairs):
        return np.vstack([
            model.score_pairs(pairs),
            data.temporal_pairs(pairs),
            data.content_pairs(pairs),
            data.itemcf_pairs(pairs),
            data.popularity_pairs(pairs),
        ]).T.astype(np.float32)

    def _default_ctr_score(self, pairs):
        return (
            1.00 * _zscore(self.model.score_pairs(pairs))
            + 0.40 * _zscore(self.data.temporal_pairs(pairs))
            + 0.35 * _zscore(self.data.content_pairs(pairs))
            + 0.30 * _zscore(self.data.itemcf_pairs(pairs))
            + 0.25 * _zscore(self.data.popularity_pairs(pairs))
        ).astype(np.float32)

    def _fit_ctr_fusion(self):
        rng = np.random.default_rng(2027)
        base_pos, tune_pos = self._split_for_fusion(self.data.train_pos, rng)
        base_neg, tune_neg = self._split_for_fusion(self.data.train_neg, rng)
        if len(tune_pos) == 0 or len(tune_neg) == 0:
            return

        tune_rows = np.vstack([tune_pos, tune_neg])
        rng.shuffle(tune_rows)
        pairs = tune_rows[:, :2]
        labels = tune_rows[:, 2].astype(np.float32)

        base_data = DataStore(base_pos, base_neg, self.kg_lines, self.rel_path)
        base_model = MatrixFactorModel(base_data)
        base_model.train_TransE(max(6, self.epoch_num // 2))
        features = self._feature_matrix(base_data, base_model, pairs)

        mean = features.mean(axis=0).astype(np.float32)
        std = features.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        x = (features - mean) / std

        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(labels, dtype=torch.float32)
        weight = torch.zeros(x.shape[1] + 1, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            weight[1:] = torch.as_tensor(self.ctr_weights, dtype=torch.float32)

        optimizer = torch.optim.Adam([weight], lr=0.05)
        for _ in range(300):
            logits = weight[0] + x_tensor.matmul(weight[1:])
            loss = F.binary_cross_entropy_with_logits(logits, y_tensor) + 0.01 * (weight[1:] ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        learned_weight = weight.detach().numpy().astype(np.float32)
        learned_score = learned_weight[0] + x.dot(learned_weight[1:])
        default_score = sum(self.ctr_weights[i] * _zscore(features[:, i]) for i in range(features.shape[1]))
        if self._auc_score(labels, learned_score) >= self._auc_score(labels, default_score):
            self.ctr_bias = float(learned_weight[0])
            self.ctr_weights = learned_weight[1:]
            self.ctr_mean = mean
            self.ctr_std = std
            self.use_learned_ctr = True

    def eval_ctr(self, test_data: np.array) -> np.array:
        pairs = _as_array(test_data, 2)
        if len(pairs) == 0:
            return np.empty(0, dtype=np.float32)

        if not self.use_learned_ctr:
            return self._default_ctr_score(pairs)

        features = self._feature_matrix(self.data, self.model, pairs)
        score = self.ctr_bias + ((features - self.ctr_mean) / self.ctr_std).dot(self.ctr_weights)
        return score.astype(np.float32)

    def eval_topk(self, users: List[int], k: int = 5) -> List[List[int]]:
        result = []
        for user in users:
            user = int(user)
            if user not in self.data.user_pos and user not in self.data.user_neg:
                score = self.data.cold_start_all().copy()
            else:
                score = self.data.base_all(user) + 0.50 * _zscore(self.model.score_all(user))
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
