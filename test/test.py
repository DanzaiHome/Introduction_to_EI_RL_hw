import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from data_define import data_used as data
import numpy as np

def max_score_path(grid):
    """
    求从 (0,0) 出发到 (H-1, W-1) 的最高得分。
    若存在可让得分无限增大的可达终点的正增益回路，返回 float('inf')。
    """
    import collections

    H = len(grid)
    W = len(grid[0])
    
    # 特殊情况：只有一个格子时，直接返回该格子的分数 + 终点奖励
    if H == 1 and W == 1:
        return grid[0][0] + data.reward_at_destination
    
    # 方向向量：0=上, 1=右, 2=下, 3=左。
    # 注：d=0 表示“本次移动是 (r+1,c)->(r,c)”；也就是从下往上走。
    #     所以下一步若想保持连贯，需要到 (r-1,c) ...
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # opposite[d] = d的反方向：上<->下, 左<->右
    opposite = {0:2, 1:3, 2:0, 3:1}
    
    # 为每一个 (r, c, d) 分配一个 index
    # 其中 d=-1 仅在起点 (0,0) 使用，其它 (r,c) 均可有 d=0,1,2,3
    # 这里我们将 d=-1 存成一个单独的状态索引。
    # 整体状态数不超过 1(特殊的d=-1) + H*W*4(正常方向)。
    
    # 建立状态列表和从 (r,c,d) 到 state_id 的映射
    state_list = []
    state_id = {}
    idx = 0
    
    # 给 (0,0,-1) 一个专属 ID
    start_state = (0, 0, -1)
    state_list.append(start_state)
    state_id[start_state] = idx
    idx += 1

    # 给其它格子 (r,c) 的 d=0,1,2,3 建立 ID
    for r in range(H):
        for c in range(W):
            for d in range(4):
                st = (r, c, d)
                state_list.append(st)
                state_id[st] = idx
                idx += 1
    
    N = len(state_list)  # 总状态数
    
    # 判断一个 (r,c) 是否在网格内
    def in_bounds(rr, cc):
        return 0 <= rr < H and 0 <= cc < W
    
    # 预先把所有可能的边收集出来：edges = [(u,v,weight), ...]
    edges = []
    
    # 起点 (0,0,-1) 可以向 (0,0,d) 产生一条“虚拟”边，表示“选定入站方向 d 后真正处于 (0,0,d)”
    # 不过，这相当于站在 (0,0) 之后“发第一步”到自己，这里可以简化为：
    #   我们从 start_state 直接“转移”到下一个实际网格格子 (r', c')?
    #   也可以干脆只把 dist[start_state] = grid[0][0] 看做抵达 (0,0) 时的加分，
    #   然后从 (0,0,-1) 出发可以走任何方向 d(0..3) 去下一个格子。
    # 下面做法：从 (0,0,-1) 出发，可以尝试向 4 个方向移动到相邻格子(若在范围内)。
    
    start_u = state_id[(0, 0, -1)]
    for d in range(4):
        dr, dc = directions[d]
        nr, nc = 0 + dr, 0 + dc
        if in_bounds(nr, nc):
            v = state_id[(nr, nc, d)]
            # 边权：到达 (nr,nc) 时加 grid[nr][nc]，走一步扣6
            # 若 (nr,nc) 即是终点，还要额外加 20
            w = grid[nr][nc] + data.cost_each_step
            if (nr, nc) == (H-1, W-1):
                w += data.reward_at_destination
            # 注意：dist[start_u] 初始化后自带 grid[0][0]，这里不需重复加
            # 因为这条边的含义是“从start_state走到(nr,nc,d)”
            edges.append((start_u, v, w))
    
    # 对于普通状态 (r,c,d)，可以向下一个状态 (nr,nc,d') 转移
    # 条件：d' != opposite[d]（不能直接回头），并且 (nr,nc) 在界内
    # 其中 nr = r + directions[d'][0], nc = c + directions[d'][1]
    # 边权 = grid[nr][nc] - 6 (+20 if 是终点)
    for r in range(H):
        for c in range(W):
            for d in range(4):
                # 如果当前已经是终点，就不再发射任何边
                if (r, c) == (H-1, W-1):
                    continue

                u = state_id[(r, c, d)]
                for d_next in range(4):
                    if d_next == opposite[d]:
                        continue
                    nr, nc = r + directions[d_next][0], c + directions[d_next][1]
                    if in_bounds(nr, nc):
                        v = state_id[(nr, nc, d_next)]
                        w = grid[nr][nc] + data.cost_each_step
                        # 如果下一个是终点则加终点奖励
                        if (nr, nc) == (H-1, W-1):
                            w += data.reward_at_destination
                            w -= grid[nr][nc]
                        edges.append((u, v, w))
    
    # 我们要用 Bellman-Ford 来处理“最长路”+“检测正增益环”问题
    # dist[u] 表示从 start_state 到达状态u的最大分数，初始化为 -inf
    dist = [-float('inf')] * N
    dist[start_u] = 0
    
    # 终点状态集合： (H-1, W-1, d) for d in [0..3]
    end_states = [state_id[(H-1, W-1, d)] for d in range(4)]
    
    # Bellman-Ford 主循环：做 N-1 次松弛
    for _ in range(N - 1):
        updated = False
        for (u, v, w) in edges:
            if dist[u] > -float('inf'):
                new_val = dist[u] + w
                if new_val > dist[v]:
                    dist[v] = new_val
                    updated = True
        if not updated:
            break
    
    # 检查第 N 次是否还能更新 —— 若能更新说明存在正增益回路
    # 需要确认该回路“可从起点到达”且“能到达终点”。
    # 如果第 N 次还能更新 dist[v]，将 v 标记为 in_cycle_candidates。
    in_cycle_candidates = set()
    for (u, v, w) in edges:
        if dist[u] > -float('inf'):
            new_val = dist[u] + w
            if new_val > dist[v]:
                in_cycle_candidates.add(v)

    if not in_cycle_candidates:
        # 没有可进一步更新的点 => 无正增益回路
        # 直接取所有终点状态的 dist 最大值
        answer = max(dist[s] for s in end_states)
        if answer == -float('inf'):
            # 如果所有终点不可达，题目未说明是否需要返回特定值，这里返回 -inf
            return -float('inf')
        else:
            return answer
    else:
        # 存在可继续变大的点，说明有正增益环，但还需确定这些环能否到达终点
        # 做一个图的 BFS / DFS 看从 in_cycle_candidates 出发是否能到达某个 end_state
        graph_forward = collections.defaultdict(list)
        for (u, v, _) in edges:
            graph_forward[u].append(v)
        
        # 从这些候选点出发，看看是否能到达终点
        visited = set()
        queue = collections.deque(in_cycle_candidates)
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for nxt in graph_forward[cur]:
                if nxt not in visited:
                    queue.append(nxt)
        
        # 如果 visited 和 end_states 有交集，说明有环可到达终点 => 分数无穷大
        if any(s in visited for s in end_states):
            return float('inf')
        else:
            # 否则，这些正增益环到不了终点，仍要返回原先的最大值
            answer = max(dist[s] for s in end_states)
            if answer == -float('inf'):
                return -float('inf')
            else:
                return answer


if __name__ == "__main__":
    grid = data.score_map
    H, W = np.array(grid).shape
    grid[0][0] = 0
    grid[H-1][W-1] = 0
    print(grid)
    ans = max_score_path(grid)
    print("最大得分 =", ans)
