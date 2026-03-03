"""
Cross-Region Attention Communication (CRAC) Module
===================================================

為 RegionLight 提供基於圖注意力的跨區域訊息傳遞機制。

核心思想：
    每個 Region Agent 從自身 local state 生成一個 message embedding，
    再透過 Scaled Dot-Product Attention 對鄰居 Region 的 messages 做加權聚合，
    得到的 comm_vector 與 local representation 串接後供 BDQ 做決策。

創新點：
    1. 基於路網拓撲自動建構 Region 鄰接圖（Region Adjacency Graph）
    2. Attention-based Message Passing —— 選擇性地聚合鄰居資訊
    3. Learnable Gating Mechanism —— 自適應控制對鄰居信息的信任程度
    4. 端到端可訓練，message encoder 與 Q-network 共享 shared representation
"""

import numpy as np


def parse_itsx_coords(itsx_id):
    """
    Parse intersection id -> (x, y) coordinates.
    e.g. 'intersection_2_3' -> (2, 3)
    Returns None for 'dummy' intersections.
    """
    if itsx_id == 'dummy':
        return None
    parts = itsx_id.split('_')
    return (int(parts[1]), int(parts[2]))


def build_region_adjacency(itsx_assignment):
    """
    根據各 Region 包含的 intersection 座標，自動建構 Region 鄰接圖。
    兩個 Region 若有 intersection 的 Manhattan Distance = 1，即為鄰居。

    Args:
        itsx_assignment: list of lists，每個內層 list 為一個 Region 的 intersection IDs

    Returns:
        adjacency: list of lists，adjacency[i] = Region i 的鄰居 Region indices
    """
    # 解析各 Region 的 intersection 座標
    region_coords = []
    for region in itsx_assignment:
        coords = set()
        for itsx in region:
            c = parse_itsx_coords(itsx)
            if c is not None:
                coords.add(c)
        region_coords.append(coords)

    num_regions = len(itsx_assignment)
    adjacency = [[] for _ in range(num_regions)]

    for i in range(num_regions):
        for j in range(i + 1, num_regions):
            is_adjacent = False
            for (x1, y1) in region_coords[i]:
                for (x2, y2) in region_coords[j]:
                    if abs(x1 - x2) + abs(y1 - y2) == 1:
                        is_adjacent = True
                        break
                if is_adjacent:
                    break
            if is_adjacent:
                adjacency[i].append(j)
                adjacency[j].append(i)

    return adjacency


class RegionCommCoordinator:
    """
    跨區域訊息交換協調器。

    在每個 decision step 中：
        1. 收集所有 Agent 的 local state → 生成 message embedding
        2. 根據 Region 鄰接圖，將鄰居的 messages 打包成 (msgs, mask) 傳給各 Agent
        3. 各 Agent 利用 attention 聚合鄰居資訊，做出更好的決策

    Attributes:
        adjacency: Region 鄰接表
        max_neighbors: 所有 Region 中最大鄰居數（用於 padding）
        comm_dim: message embedding 維度
    """

    def __init__(self, adjacency, comm_dim):
        self.adjacency = adjacency
        self.max_neighbors = max(len(n) for n in adjacency) if adjacency else 1
        self.comm_dim = comm_dim

        print(f"\n{'='*50}")
        print(f"  Cross-Region Attention Communication (CRAC)")
        print(f"{'='*50}")
        print(f"  Region adjacency graph:")
        for i, neighbors in enumerate(adjacency):
            print(f"    Region {i} <-> Regions {neighbors}")
        print(f"  Max neighbors:  {self.max_neighbors}")
        print(f"  Comm dim:       {comm_dim}")
        print(f"{'='*50}\n")

    def exchange_messages(self, agents, observations):
        """
        執行一輪跨區域訊息交換。

        Args:
            agents: list of CommBDQ_agent
            observations: list of np.array，各 Agent 的 local observation

        Returns:
            neighbor_data: list of dict，每個 dict 包含：
                'msgs': (max_neighbors, comm_dim) 鄰居的 messages（padding 部分為 0）
                'mask': (max_neighbors,) 有效鄰居標記（1=有效, 0=padding）
        """
        # Step 1: 每個 Agent 從自身 local state 生成 message
        messages = []
        for aid, agent in enumerate(agents):
            msg = agent.generate_message(observations[aid])
            messages.append(msg)

        # Step 2: 根據鄰接圖，為每個 Agent 打包鄰居 messages
        neighbor_data = []
        for aid in range(len(agents)):
            msgs = np.zeros((self.max_neighbors, self.comm_dim), dtype=np.float32)
            mask = np.zeros(self.max_neighbors, dtype=np.float32)
            for nidx, neighbor_id in enumerate(self.adjacency[aid]):
                if nidx < self.max_neighbors:
                    msgs[nidx] = messages[neighbor_id]
                    mask[nidx] = 1.0
            neighbor_data.append({'msgs': msgs, 'mask': mask})

        return neighbor_data
