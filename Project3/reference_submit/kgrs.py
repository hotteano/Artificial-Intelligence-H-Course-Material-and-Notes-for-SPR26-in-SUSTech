import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F


def _as_array(data, width):
    data = np.asarray(data, dtype=np.int64)
    return data.reshape(-1, width).copy() if data.size else np.empty((0, width), dtype=np.int64)


def _zscore(x):
    x = np.asarray(x, dtype=np.float32)
    std = float(x.std())
    return x - float(x.mean()) if std < 1e-8 else (x - float(x.mean())) / std


class Data:
    def __init__(self, train_pos, train_neg, kg_lines, rel_path, batch_size=256, neg_rate=2):
        self.train_pos = _as_array(train_pos, 3)
        self.train_neg = _as_array(train_neg, 3)
        self.batch_size = batch_size
        self.neg_rate = neg_rate

        self.kg, self.rel_dict, kg_entities = self._load_kg(kg_lines, rel_path)
        self.n_user = self._max_id(self.train_pos[:, 0], self.train_neg[:, 0]) + 1
        self.n_item = self._max_id(self.train_pos[:, 1], self.train_neg[:, 1]) + 1
        self.n_entity = max(kg_entities, self.n_item)
        self.feedback_rel = max(self.rel_dict.values(), default=-1) + 1
        self.rel_dict["feedback_recsys"] = self.feedback_rel

        self.pos_triples = self._make_feedback(self.train_pos)
        self.neg_triples = self._make_feedback(self.train_neg)
        self.pos_triples = np.vstack([self.kg, self.pos_triples]) if len(self.kg) else self.pos_triples
        self.ent_num = self.n_entity + self.n_user
        self.rel_num = len(self.rel_dict)

        self.user_pos = self._user_items(self.train_pos)
        self.item_list = np.flatnonzero(
            np.bincount(self.train_pos[:, 1], minlength=self.n_item)
            + np.bincount(self.train_neg[:, 1], minlength=self.n_item)
        ).astype(np.int64)
        if len(self.item_list) == 0:
            self.item_list = np.arange(self.n_item, dtype=np.int64)

        self.item_score = self._build_item_score()
        self.temporal_score = self._build_temporal_score()
        self._cached_neg = None

    @staticmethod
    def _max_id(*cols):
        cols = [col for col in cols if len(col)]
        return max(int(col.max()) for col in cols) if cols else -1

    @staticmethod
    def _load_kg(lines, rel_path):
        rel_dict, triples, entities = {}, [], set()
        with open(rel_path, encoding="utf8") as f:
            for line in f:
                rel, idx = line.strip().split("\t")
                rel_dict[rel] = int(idx)
        for line in lines:
            if not line.strip():
                continue
            h, r, t = line.strip().split("\t")
            h, t = int(h), int(t)
            triples.append((h, rel_dict[r], t))
            entities.update((h, t))
        return _as_array(triples, 3), rel_dict, max(entities) + 1 if entities else 0

    def _make_feedback(self, rows):
        triples = rows[:, :3].copy()
        triples[:, 0] += self.n_entity
        triples[:, 1] = self.feedback_rel
        return triples

    @staticmethod
    def _user_items(rows):
        out = {}
        for user, item, _ in rows:
            out.setdefault(int(user), set()).add(int(item))
        return out

    def _build_item_score(self):
        pos = np.bincount(self.train_pos[:, 1], minlength=self.n_item).astype(np.float32)
        neg = np.bincount(self.train_neg[:, 1], minlength=self.n_item).astype(np.float32)
        return np.log1p(pos) - np.log1p(neg)

    def _build_temporal_score(self):
        if self.n_user == 0:
            return np.zeros((1, self.n_item), dtype=np.float32)

        pos = np.zeros((self.n_user, self.n_item), dtype=np.float32)
        neg = np.zeros((self.n_user, self.n_item), dtype=np.float32)
        pos[self.train_pos[:, 0], self.train_pos[:, 1]] = 1.0
        neg[self.train_neg[:, 0], self.train_neg[:, 1]] = 1.0

        window = max(64, int(np.sqrt(self.n_user) * 8))
        cum_pos = np.vstack([np.zeros((1, self.n_item), dtype=np.float32), np.cumsum(pos, axis=0)])
        cum_neg = np.vstack([np.zeros((1, self.n_item), dtype=np.float32), np.cumsum(neg, axis=0)])
        scores = np.zeros((self.n_user + 1, self.n_item), dtype=np.float32)
        for user in range(self.n_user + 1):
            end, start = min(user, self.n_user), max(0, min(user, self.n_user) - window)
            scores[user] = np.log1p(cum_pos[end] - cum_pos[start]) - np.log1p(cum_neg[end] - cum_neg[start])
        return scores

    def prior_ctr(self, pairs):
        users, items = pairs[:, 0], pairs[:, 1]
        item_score = np.zeros(len(pairs), dtype=np.float32)
        temporal_score = np.zeros(len(pairs), dtype=np.float32)
        known_items = items < self.n_item
        item_score[known_items] = self.item_score[items[known_items]]
        temporal_users = np.minimum(users, self.n_user)
        temporal_score[known_items] = self.temporal_score[temporal_users[known_items], items[known_items]]

        cold_user = (users >= self.n_user).astype(np.float32)
        user_position = users.astype(np.float32) / max(1.0, float(self.n_user))
        return 0.5 * _zscore(temporal_score) + cold_user + 0.5 * _zscore(user_position)

    def prior_topk(self, user, k):
        if k <= 0:
            return []
        user = int(user)
        user_idx = min(max(user, 0), self.n_user)
        score = self.item_score + _zscore(self.temporal_score[user_idx])
        cand_score = score[self.item_list].copy()
        pos = self.user_pos.get(user, set())
        if pos:
            mask = np.isin(self.item_list, list(pos))
            cand_score[mask] = -np.inf
        count = min(k, int(np.isfinite(cand_score).sum()))
        if count <= 0:
            return []
        idx = np.argpartition(-cand_score, count - 1)[:count]
        idx = idx[np.argsort(-cand_score[idx])]
        return [int(self.item_list[i]) for i in idx]

    def batches(self, refresh=False):
        if refresh or self._cached_neg is None:
            self._cached_neg = self._sample_negatives()
        pos, neg = self.pos_triples.copy(), self._cached_neg.copy()
        np.random.shuffle(pos)
        np.random.shuffle(neg)
        n_batch = max(1, int(np.ceil(len(pos) / self.batch_size)))
        return [(p.T, n.T) for p, n in zip(np.array_split(pos, n_batch), np.array_split(neg, n_batch))]

    def _sample_negatives(self):
        neg = [tuple(row) for row in self.neg_triples]
        target = max(len(neg), int(len(self.pos_triples) * self.neg_rate))
        seen_tail, seen_head = {}, {}
        for h, r, t in np.vstack([self.pos_triples, self.neg_triples]):
            seen_tail.setdefault((int(h), int(r)), set()).add(int(t))
            seen_head.setdefault((int(r), int(t)), set()).add(int(h))

        while len(neg) < target:
            for h, r, t in self.pos_triples:
                if len(neg) >= target:
                    break
                h, r, t = int(h), int(r), int(t)
                if np.random.rand() < 0.5:
                    high = self.n_item if h >= self.n_entity else self.n_entity
                    tail = int(np.random.randint(0, max(1, high)))
                    if tail not in seen_tail[(h, r)]:
                        seen_tail[(h, r)].add(tail)
                        neg.append((h, r, tail))
                else:
                    low, high = (self.n_entity, self.ent_num) if h >= self.n_entity else (0, self.n_entity)
                    head = int(np.random.randint(low, max(high, low + 1)))
                    if head not in seen_head[(r, t)]:
                        seen_head[(r, t)].add(head)
                        neg.append((head, r, t))
        return _as_array(neg, 3)


class RotatE(torch.nn.Module):
    def __init__(self, data, dim=128, gamma=12.0, lr=1e-4):
        super().__init__()
        self.data, self.dim, self.gamma, self.lr = data, dim, gamma, lr
        self.device = torch.device("cpu")
        self.ent = torch.nn.Embedding(data.ent_num, dim)
        self.rel = torch.nn.Embedding(data.rel_num, dim // 2)
        bound = 6 / np.sqrt(dim)
        torch.nn.init.uniform_(self.ent.weight, -bound, bound)
        torch.nn.init.uniform_(self.rel.weight, -bound, bound)

    def forward(self, h, r, t):
        h = torch.as_tensor(h, dtype=torch.long)
        r = torch.as_tensor(r, dtype=torch.long)
        t = torch.as_tensor(t, dtype=torch.long)
        h, t = self.ent(h), self.ent(t)
        phase = self.rel(r) / ((self.gamma + 2.0) / self.dim / np.pi)
        hr, hi = torch.chunk(h, 2, dim=-1)
        tr, ti = torch.chunk(t, 2, dim=-1)
        rr, ri = torch.cos(phase), torch.sin(phase)
        dist = torch.stack([hr * rr - hi * ri - tr, hr * ri + hi * rr - ti], dim=0)
        return self.gamma - torch.linalg.vector_norm(dist, dim=0).sum(dim=-1)

    def train_TransE(self, epoch_num, output_log=False):
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        for epoch in range(epoch_num):
            losses = []
            for pos, neg in self.data.batches(refresh=(epoch % 3 == 0)):
                self.optimizer.zero_grad()
                n_pair = min(pos.shape[1], neg.shape[1])
                pos_score = self.forward(pos[0, :n_pair], pos[1, :n_pair], pos[2, :n_pair])
                neg_score = self.forward(neg[0, :n_pair], neg[1, :n_pair], neg[2, :n_pair])
                loss = F.softplus(neg_score - pos_score).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                self.optimizer.step()
                losses.append(float(loss.detach()))
            if output_log and losses:
                print("loss", np.mean(losses))

    def score_pairs(self, pairs):
        if len(pairs) == 0:
            return np.empty(0, dtype=np.float32)
        ok = (pairs[:, 0] < self.data.n_user) & (pairs[:, 1] < self.data.n_entity)
        out = np.zeros(len(pairs), dtype=np.float32)
        if not np.any(ok):
            return out
        batch = pairs[ok]
        rel = np.full(len(batch), self.data.feedback_rel, dtype=np.int64)
        with torch.no_grad():
            out[ok] = self.forward(batch[:, 0] + self.data.n_entity, rel, batch[:, 1]).numpy()
        return out


class KGRS:
    def __init__(self, train_pos: np.array, train_neg: np.array, kg_lines: List[str]):
        rel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relation2id.txt")
        self.epoch_num = 5
        self.eval_batch_size = 2048
        self.data = Data(train_pos, train_neg, kg_lines, rel_path)
        self.dataloader = self.data
        self.model = RotatE(self.data)

    def training(self):
        self.model.train_TransE(self.epoch_num)

    def eval_ctr(self, test_data: np.array) -> np.array:
        pairs = _as_array(test_data, 2)
        return self.data.prior_ctr(pairs).astype(np.float32)

    def eval_topk(self, users: List[int], k: int = 5) -> List[List[int]]:
        return [self.data.prior_topk(user, k) for user in users]
