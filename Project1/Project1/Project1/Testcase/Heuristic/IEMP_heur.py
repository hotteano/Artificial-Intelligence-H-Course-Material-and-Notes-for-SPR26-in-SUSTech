class IEMData:
    
    def __init__(self):
        self.n_nodes = 0
        self.n_edges = 0
        self.graph = {} 
        self.reverse_graph = {}
        self.I1 = set()  
        self.I2 = set() 
        self.S1 = set() 
        self.S2 = set() 
    
    def load_graph(self, filepath):
        self.graph = {}
        self.reverse_graph = {}
        with open(filepath, 'r') as f:
            line = f.readline().strip().split()
            self.n_nodes = int(line[0])
            self.n_edges = int(line[1])
            
            for i in range(self.n_nodes):
                self.graph[i] = []
                self.reverse_graph[i] = []
            
            for _ in range(self.n_edges):
                u, v, p1, p2 = f.readline().strip().split()
                u, v = int(u), int(v)
                p1, p2 = float(p1), float(p2)
                
                self.graph[u].append((v, p1, p2))
                self.reverse_graph[v].append((u, p1, p2))

    
    def load_initial_seeds(self, filepath):
        with open(filepath, 'r') as f:
            n1, n2 = map(int, f.readline().strip().split())
            
            self.I1 = set(int(f.readline().strip()) for _ in range(n1))
            self.I2 = set(int(f.readline().strip()) for _ in range(n2))
    
    def save_solution(self, filepath):
        with open(filepath, 'w') as f:
            f.write(f"{len(self.S1)} {len(self.S2)}\n")
            for node in sorted(self.S1):
                f.write(f"{node}\n")
            for node in sorted(self.S2):
                f.write(f"{node}\n")

class IEMPHeuristic:

    def __init__(self, data, budget = 10, max_iter = 20):
        self.data = data
        self.budget = budget
        self.max_iter = max_iter
    
    def compute_degree_ranking(self):

        degrees = []
        for node in range(self.data.n_nodes):
            deg = len(self.data.graph[node])
            degrees.append((node, deg))
        
        degrees.sort(key=lambda x: x[1], reverse=True)

        return [node for node, deg in degrees]
    
    def lfa_strategy(self, ranking, campaign_idx):

        pos = {node: i for i, node in enumerate(ranking)}

        M = {node: 1.0 for node in ranking}

        for i in range(len(ranking) - 1, 0, -1): 
            v = ranking[i] 

            for u, p1, p2 in self.data.graph.get(v, []):

                if u in pos and pos[u] < i:

                    p = p1 if campaign_idx == 0 else p2

                    M[u] += M[v] * p 
        return M
    
    def run_imrank(self): 

        ranking1 = self.compute_degree_ranking()
        ranking2 = self.compute_degree_ranking()

        available = set(range(self.data.n_nodes)) - self.data.I1 - self.data.I2
        ranking1 = [n for n in ranking1 if n in available]
        ranking2 = [n for n in ranking2 if n in available]

        for iteration in range(self.max_iter): 

            M1 = self.lfa_strategy(ranking1, 0)
            M2 = self.lfa_strategy(ranking2, 1)

            new_ranking1 = sorted(available, key=lambda n: M1[n], reverse=True)
            new_ranking2 = sorted(available, key=lambda n: M2[n], reverse=True)

            k1 = self.budget // 2
            k2 = self.budget - k1

            topk1_old = set(ranking1[:k1])
            topk2_old = set(ranking2[:k2])
            topk1_new = set(new_ranking1[:k1])
            topk2_new = set(new_ranking2[:k2])

            ranking1, ranking2 = new_ranking1, new_ranking2

            if topk1_new == topk1_old and topk2_new == topk2_old:
                break
        
        k1 = self.budget // 2
        k2 = self.budget - k1

        self.data.S1 = set(ranking1[:k1])
        self.data.S2 = set(ranking2[:k2])

        return self.data.S1, self.data.S2

if __name__ == "__main__":
    # 加载数据
    data = IEMData()
    data.load_graph("map1/dataset1")
    data.load_initial_seeds("map1/seed")
    
    # 运行IMRank
    heuristic = IEMPHeuristic(data, budget=10, max_iter=10)
    S1, S2 = heuristic.run_imrank()
    
    print(f"S1: {S1}")
    print(f"S2: {S2}")
    
    # 保存结果
    data.save_solution("map1/seed_balanced")