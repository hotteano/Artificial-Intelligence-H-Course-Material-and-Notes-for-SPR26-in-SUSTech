import os
from typing import List
import numpy as np
import torch
import torch.nn.functional as F


class Dataloader:
    def __init__(self, train_pos, train_neg, kg_lines, rel_file_path: str,
                 train_batch_size: int = 128, neg_rate: float = 2):
        self.kg, self.rel_dict, self.n_entity = self._convert_kg(kg_lines, rel_file_path)
        self.train_pos = np.array(train_pos, copy=True)
        self.train_neg = np.array(train_neg, copy=True)
        self.n_user = max(list(set(self.train_pos[:, 0]) | set(self.train_neg[:, 0]))) + 1
        self.n_item = max(list(set(self.train_pos[:, 1]) | set(self.train_neg[:, 1]))) + 1
        self._load_ratings()
        self.known_neg_dict = []
        self._add_recsys_to_kg()
        self.train_batch_size = train_batch_size
        self.neg_rate = neg_rate
        self.ent_num = self.n_entity + self.n_user
        self.rel_num = len(self.rel_dict)

    def _add_recsys_to_kg(self):
        self.rel_dict['feedback_recsys'] = max(self.rel_dict.values()) + 1
        feedback_rel = self.rel_dict['feedback_recsys']
        for interaction in self.train_pos:
            self.kg.append((interaction[0], feedback_rel, interaction[1]))
        for interaction in self.train_neg:
            self.known_neg_dict.append((interaction[0], feedback_rel, interaction[1]))

    def _load_ratings(self):
        self.n_entity = max(self.n_item, self.n_entity)
        self.train_pos[:, 0] += self.n_entity
        self.train_neg[:, 0] += self.n_entity

    def _convert_kg(self, lines, rel_file_path: str):
        entity_set = set()
        kg = []
        rel_dict = {}
        with open(rel_file_path, encoding='utf8') as f:
            for line in f:
                elements = line.replace('\n', '').split('\t')
                rel_dict[elements[0]] = int(elements[1])
        for line in lines:
            array = line.strip().split('\t')
            head = int(array[0])
            relation = rel_dict[array[1]]
            tail = int(array[2])
            kg.append((head, relation, tail))
            entity_set.add(head)
            entity_set.add(tail)

        print('number of entities (containing items): %d' % len(entity_set))
        print('number of relations: %d' % len(rel_dict))
        return kg, rel_dict, max(entity_set) + 1 if entity_set else 0

    def get_user_pos_item_list(self):
        if hasattr(self, '_item_list_cache') and self._item_list_cache is not None:
            return self._item_list_cache, self._train_user_pos_cache
        train_user_pos_item = {}
        for record in self.train_pos:
            user, item = record[0] - self.n_entity, record[1]
            train_user_pos_item.setdefault(user, set()).add(item)
        all_record = np.concatenate([self.train_pos, self.train_neg], axis=0)
        item_list = list(set(all_record[:, 1]))
        self._item_list_cache = item_list
        self._train_user_pos_cache = train_user_pos_item
        return item_list, train_user_pos_item

    def _sample_negatives(self):
        pos_data = list(self.kg)
        neg_data = list(self.known_neg_dict)
        target = len(self.kg) * self.neg_rate

        hr_tail_set = {}
        rt_head_set = {}
        for h, r, t in pos_data + neg_data:
            hr_tail_set.setdefault((h, r), set()).add(t)
            rt_head_set.setdefault((r, t), set()).add(h)

        max_fail = target
        fails = 0

        while len(neg_data) < target and fails < max_fail:
            for h, r, t in self.kg:
                if len(neg_data) >= target or fails >= max_fail:
                    break
                if np.random.rand() > 0.5:
                    bound = self.n_item if h >= self.n_entity else self.n_entity
                    tail = np.random.randint(0, bound)
                    seen = hr_tail_set[(h, r)]
                    attempts = 0
                    while tail in seen and fails < max_fail and attempts < 10:
                        fails += 1
                        attempts += 1
                        tail = np.random.randint(0, bound)
                    if tail not in seen and fails < max_fail:
                        seen.add(tail)
                        neg_data.append((h, r, tail))
                else:
                    low, high = (self.n_entity, self.n_entity + self.n_user) if h >= self.n_entity else (0, self.n_entity)
                    head = np.random.randint(low, high)
                    seen = rt_head_set[(r, t)]
                    attempts = 0
                    while head in seen and fails < max_fail and attempts < 10:
                        fails += 1
                        attempts += 1
                        head = np.random.randint(low, high)
                    if head not in seen and fails < max_fail:
                        seen.add(head)
                        neg_data.append((head, r, t))

        return pos_data, neg_data

    def get_training_batch(self, refresh=False):
        if refresh or getattr(self, '_cached_pos', None) is None:
            self._cached_pos, self._cached_neg = self._sample_negatives()
        pos_data = np.array(self._cached_pos)
        neg_data = np.array(self._cached_neg)
        np.random.shuffle(pos_data)
        np.random.shuffle(neg_data)
        n_batches = max(1, len(pos_data) // self.train_batch_size)
        pos_batches = np.array_split(pos_data, n_batches)
        neg_batches = np.array_split(neg_data, len(pos_batches))
        pos_batches = [batch.T for batch in pos_batches]
        neg_batches = [batch.T for batch in neg_batches]
        return [[pos_batches[i], neg_batches[i]] for i in range(len(pos_batches))]


class RotatE(torch.nn.Module):
    def __init__(self, ent_num: int, rel_num: int, dataloader: Dataloader, dim: int = 128,
                 gamma: float = 12, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 device_index: int = 0):
        super().__init__()
        self.device = torch.device('cuda:{}'.format(device_index)) if device_index >= 0 else torch.device('cpu')
        self.ent_num = ent_num
        self.rel_num = rel_num
        self.dataloader = dataloader
        self.dim = dim
        self.gamma = gamma
        self.embedding_range = (self.gamma + 2.0) / self.dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.ent_embedding = torch.nn.Embedding(self.ent_num, self.dim, device=self.device)
        self.rel_embedding = torch.nn.Embedding(self.rel_num, self.dim // 2, device=self.device)

        self.ent_embedding.weight.data.uniform_(-6 / (self.dim ** 0.5), 6 / (self.dim ** 0.5))
        self.rel_embedding.weight.data.uniform_(-6 / (self.dim ** 0.5), 6 / (self.dim ** 0.5))

    def forward(self, head, rel, tail) -> torch.Tensor:
        head = torch.as_tensor(head, dtype=torch.long, device=self.device)
        rel = torch.as_tensor(rel, dtype=torch.long, device=self.device)
        tail = torch.as_tensor(tail, dtype=torch.long, device=self.device)

        h = self.ent_embedding(head)
        r = self.rel_embedding(rel)
        t = self.ent_embedding(tail)

        pi = 3.14159265358979323846
        r = r / (self.embedding_range / pi)

        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)

        cos_r = torch.cos(r)
        sin_r = torch.sin(r)

        h_rotate_re = h_re * cos_r - h_im * sin_r
        h_rotate_im = h_re * sin_r + h_im * cos_r

        score = torch.stack([h_rotate_re - t_re, h_rotate_im - t_im], dim=0).norm(dim=0).sum(dim=-1, keepdim=True)
        return self.gamma - score

    def optimize(self, pos, neg):
        pos_score = self.forward(pos[0], pos[1], pos[2])
        neg_score = self.forward(neg[0], neg[1], neg[2])
        loss = -(F.logsigmoid(pos_score).mean() + F.logsigmoid(-neg_score).mean())
        return loss

    def ctr_eval(self, eval_batches: List[np.array]):
        eval_batches = [batch.T for batch in eval_batches]
        scores = []
        rel_id = self.dataloader.rel_dict['feedback_recsys']
        offset = self.dataloader.n_entity
        for batch in eval_batches:
            n = len(batch[0])
            head = torch.as_tensor(batch[0] + offset, dtype=torch.long, device=self.device)
            rel = torch.full((n,), rel_id, dtype=torch.long, device=self.device)
            tail = torch.as_tensor(batch[1], dtype=torch.long, device=self.device)
            with torch.no_grad():
                score = self.forward(head, rel, tail).squeeze(-1)
            scores.append(score.cpu().numpy())
        scores = np.concatenate(scores, axis=0)
        return scores

    def top_k_eval(self, users: List[int], k: int = 5):
        item_list, train_user_pos_item = self.dataloader.get_user_pos_item_list()
        n_items = len(item_list)
        rel_id = self.dataloader.rel_dict['feedback_recsys']
        item_to_idx = {item: idx for idx, item in enumerate(item_list)}

        tail_tensor = torch.tensor(item_list, dtype=torch.long, device=self.device)
        rel_tensor = torch.full((n_items,), rel_id, dtype=torch.long, device=self.device)

        sorted_list = []
        batch_size = 32

        for start in range(0, len(users), batch_size):
            batch_users = users[start:start + batch_size]
            n_batch = len(batch_users)

            head = torch.tensor(
                [u + self.dataloader.n_entity for u in batch_users],
                dtype=torch.long, device=self.device
            ).unsqueeze(1).repeat(1, n_items).view(-1)
            rel = rel_tensor.unsqueeze(0).repeat(n_batch, 1).view(-1)
            tail = tail_tensor.unsqueeze(0).repeat(n_batch, 1).view(-1)

            with torch.no_grad():
                scores = self.forward(head, rel, tail)
            scores = scores.view(n_batch, n_items).cpu().numpy()

            for i, user in enumerate(batch_users):
                user_scores = scores[i].copy()
                train_pos = train_user_pos_item.get(user)
                if train_pos:
                    exclude = [item_to_idx[item] for item in train_pos if item in item_to_idx]
                    user_scores[exclude] = -np.inf
                topk_idx = np.argsort(-user_scores)[:k]
                sorted_list.append([item_list[idx] for idx in topk_idx])

        return sorted_list

    def train_TransE(self, epoch_num: int, output_log=False):
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in range(epoch_num):
            refresh = getattr(self, '_train_step', 0) % 5 == 0
            self._train_step = getattr(self, '_train_step', 0) + 1
            train_batches = self.dataloader.get_training_batch(refresh=refresh)
            losses = []
            for batch in train_batches:
                self.optimizer.zero_grad()
                loss = self.optimize(batch[0], batch[1])
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            if output_log:
                print("The loss after the", epoch, "epochs is", np.mean(losses))


class KGRS:
    def __init__(self, train_pos: np.array, train_neg: np.array, kg_lines: List[str]):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        rel_file_path = os.path.join(module_dir, 'relation2id.txt')
        config = {"batch_size": 256, "eval_batch_size": 1024, "neg_rate": 2, "emb_dim": 128, "gamma": 12,
                  "learning_rate": 1e-4, "weight_decay": 0, "epoch_num": 30}
        self.batch_size = config["batch_size"]
        self.eval_batch_size = config["eval_batch_size"]
        self.neg_rate = config["neg_rate"]
        self.emb_dim = config["emb_dim"]
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.epoch_num = config["epoch_num"]
        self.device_index = -1
        self.kg = kg_lines
        self.dataloader = Dataloader(train_pos, train_neg, self.kg, rel_file_path,
                                     neg_rate=self.neg_rate, train_batch_size=self.batch_size)
        self.model = RotatE(ent_num=self.dataloader.ent_num, rel_num=self.dataloader.rel_num,
                            dataloader=self.dataloader, gamma=self.gamma, dim=self.emb_dim,
                            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                            device_index=self.device_index)

    def training(self):
        self.model.train_TransE(epoch_num=self.epoch_num)

    def eval_ctr(self, test_data: np.array) -> np.array:
        n_batches = max(1, len(test_data) // self.eval_batch_size)
        eval_batches = np.array_split(test_data, n_batches)
        return self.model.ctr_eval(eval_batches)

    def eval_topk(self, users: List[int], k: int = 5) -> List[List[int]]:
        return self.model.top_k_eval(users, k=k)
