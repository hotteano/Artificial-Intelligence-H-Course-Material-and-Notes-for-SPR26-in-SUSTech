import os
from typing import List
import numpy as np
import torch
import random
from tqdm import tqdm


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
        self.rel_dict['feedback_recsys'] = max([self.rel_dict[key] for key in self.rel_dict]) + 1
        for interaction in self.train_pos:
            self.kg.append((interaction[0], self.rel_dict['feedback_recsys'], interaction[1]))
        for interaction in self.train_neg:
            self.known_neg_dict.append((interaction[0], self.rel_dict['feedback_recsys'], interaction[1]))

    def _load_ratings(self):
        self.n_entity = max(self.n_item, self.n_entity)
        for i in range(len(self.train_pos)):
            self.train_pos[i][0] += self.n_entity
        for i in range(len(self.train_neg)):
            self.train_neg[i][0] += self.n_entity

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
        return kg, rel_dict, max(list(entity_set)) + 1 if len(entity_set) > 0 else 0

    def get_user_pos_item_list(self):
        train_user_pos_item = {}
        all_record = np.concatenate([self.train_pos, self.train_neg], axis=0)
        for record in self.train_pos:
            user, item = record[0] - self.n_entity, record[1]
            if user not in train_user_pos_item:
                train_user_pos_item[user] = set()
            train_user_pos_item[user].add(item)
        item_list = list(set(all_record[:, 1]))
        return item_list, train_user_pos_item

    def get_training_batch(self):
        pos_data = [fact for fact in self.kg]
        neg_data = [fact for fact in self.known_neg_dict]
        hr_tail_set = {}
        rt_head_set = {}
        for fact in pos_data + neg_data:
            if (fact[0], fact[1]) not in hr_tail_set:
                hr_tail_set[(fact[0], fact[1])] = set()
            if (fact[1], fact[2]) not in rt_head_set:
                rt_head_set[(fact[1], fact[2])] = set()
            hr_tail_set[(fact[0], fact[1])].add(fact[2])
            rt_head_set[(fact[1], fact[2])].add(fact[0])
        sample_failed_time = 0
        sample_failed_max = len(self.kg) * self.neg_rate
        while len(neg_data) < len(self.kg) * self.neg_rate and sample_failed_time < sample_failed_max:
            if sample_failed_time < sample_failed_max:
                for fact in self.kg:
                    if len(neg_data) >= len(self.kg) * self.neg_rate:
                        break
                    if random.random() > 0.5:
                        if fact[0] >= self.n_entity:
                            tail = random.randint(0, self.n_item - 1)
                            while tail in hr_tail_set[(fact[0], fact[1])] and sample_failed_time < sample_failed_max:
                                sample_failed_time += 1
                                tail = random.randint(0, self.n_item - 1)
                        else:
                            tail = random.randint(0, self.n_entity - 1)
                            while tail in hr_tail_set[(fact[0], fact[1])] and sample_failed_time < sample_failed_max:
                                sample_failed_time += 1
                                tail = random.randint(0, self.n_entity - 1)
                        if sample_failed_time < sample_failed_max:
                            hr_tail_set[(fact[0], fact[1])].add(tail)
                            neg_data.append((fact[0], fact[1], tail))
                    else:
                        if fact[0] >= self.n_entity:
                            head = random.randint(self.n_entity, self.n_entity + self.n_user - 1)
                            while head in rt_head_set[(fact[1], fact[2])] and sample_failed_time < sample_failed_max:
                                sample_failed_time += 1
                                head = random.randint(self.n_entity, self.n_entity + self.n_user - 1)
                        else:
                            head = random.randint(0, self.n_entity - 1)
                            while head in rt_head_set[(fact[1], fact[2])] and sample_failed_time < sample_failed_max:
                                sample_failed_time += 1
                                head = random.randint(0, self.n_entity - 1)
                        if sample_failed_time < sample_failed_max:
                            rt_head_set[(fact[1], fact[2])].add(head)
                            neg_data.append((head, fact[1], fact[2]))
        random.shuffle(pos_data)
        random.shuffle(neg_data)
        pos_batches = np.array_split(pos_data, max(1, len(pos_data) // self.train_batch_size))
        neg_batches = np.array_split(neg_data, len(pos_batches))
        pos_batches = [batch.transpose() for batch in pos_batches]
        neg_batches = [batch.transpose() for batch in neg_batches]
        return [[pos_batches[index], neg_batches[index]] for index in range(len(pos_batches))]


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
        head = torch.IntTensor(head).to(self.device)
        rel = torch.IntTensor(rel).to(self.device)
        tail = torch.IntTensor(tail).to(self.device)

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
        loss = -torch.mean(torch.log(torch.sigmoid(pos_score) + 1e-8)) \
               - torch.mean(torch.log(torch.sigmoid(-neg_score) + 1e-8))
        return loss

    def ctr_eval(self, eval_batches: List[np.array]):
        eval_batches = [batch.transpose() for batch in eval_batches]
        scores = []
        for batch in eval_batches:
            rel = [self.dataloader.rel_dict['feedback_recsys'] for _ in range(len(batch[0]))]
            score = torch.squeeze(self.forward(batch[0] + self.dataloader.n_entity, rel, batch[1]), dim=-1)
            scores.append(score.cpu().detach().numpy())
        scores = np.concatenate(scores, axis=0)
        return scores

    def top_k_eval(self, users: List[int], k: int = 5):
        item_list, train_user_pos_item = self.dataloader.get_user_pos_item_list()
        sorted_list = []
        for user in users:
            head = [user + self.dataloader.n_entity for _ in range(len(item_list))]
            rel = [self.dataloader.rel_dict['feedback_recsys'] for _ in range(len(item_list))]
            tail = item_list
            scores = torch.squeeze(self.forward(head, rel, tail), dim=-1)
            score_ast = np.argsort(scores.cpu().detach().numpy(), axis=-1)[::-1]
            sorted_items = []
            for index in score_ast:
                if len(sorted_items) >= k:
                    break
                if user not in train_user_pos_item or item_list[index] not in train_user_pos_item[user]:
                    sorted_items.append(item_list[index])
            sorted_list.append(sorted_items)
        return sorted_list

    def train_TransE(self, epoch_num: int, output_log=False):
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in tqdm(range(epoch_num)):
            train_batches = self.dataloader.get_training_batch()
            losses = []
            for batch in train_batches:
                loss = self.optimize(batch[0], batch[1])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.cpu().detach().numpy())
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
