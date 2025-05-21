import numpy as np
import networkx as nx
import random

def run_bfs(map, start, goal):
    n_rows = len(map)
    n_cols = len(map[0])
    queue = [(goal, [])]
    visited = {goal}
    d = {goal: 0}

    while queue:
        current, path = queue.pop(0)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if (0 <= next_pos[0] < n_rows and 0 <= next_pos[1] < n_cols and
                next_pos not in visited and map[next_pos[0]][next_pos[1]] == 0):
                visited.add(next_pos)
                d[next_pos] = d[current] + 1
                queue.append((next_pos, path + [next_pos]))
    
    if start not in d:
        return 'S', 100000
    t = 0
    actions = ['U', 'D', 'L', 'R']
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        next_pos = (start[0] + dx, start[1] + dy)
        if next_pos in d and d[next_pos] == d[start] - 1:
            return actions[t], d[next_pos]
        t += 1
    return 'S', d[start]

class OptimizedFlowAgent:
    def __init__(self):
        self.n_robots = 0
        self.state = None
        self.robots = []
        self.packages = {}
        self.board_path = {}
        self.map = []
        self.waiting_packages = []
        self.in_transit_packages = []
        self.transited_packages = []
        self.transit_success = 0
        self.packages_owned = {}
        self.count_repeat = {}
        self.last_move = {}
        self.NUM_REPEAT = 5

    def init_agents(self, state):
        self.state = state
        self.n_robots = len(state['robots'])
        self.map = state['map']
        self.robots = [(r[0]-1, r[1]-1, r[2]) for r in state['robots']]
        for robot_id in range(self.n_robots):
            self.count_repeat[robot_id] = 0
            self.last_move[robot_id] = None
        self.update_packages(state)

    def update_packages(self, state):
        for p in state['packages']:
            self.packages[p[0]] = (p[1]-1, p[2]-1, p[3]-1, p[4]-1, p[5], p[6])
            if p[0] not in self.packages_owned:
                self.waiting_packages.append(p[0])

    def optimal_assign(self, state, robots_free):
        G = nx.DiGraph()
        G.add_node('s')
        G.add_node('t')
        for robot_id in robots_free:
            G.add_node(f'r{robot_id}')
            G.add_edge('s', f'r{robot_id}', capacity=1, weight=0)
        
        valid_pos_package = {}
        for package_id in self.waiting_packages:
            pkg = self.packages[package_id]
            pos = (pkg[0], pkg[1])
            if pos not in valid_pos_package:
                valid_pos_package[pos] = []
            valid_pos_package[pos].append(package_id)
        
        # Chuẩn hóa tên node gói hàng
        for pos in valid_pos_package:
            node_name = f'p_{pos[0]}_{pos[1]}'  # Định dạng: p_x_y
            G.add_node(node_name)
            G.add_edge(node_name, 't', capacity=1, weight=0)
            for robot_id in robots_free:
                robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
                pkg = self.packages[valid_pos_package[pos][0]]
                start_path = run_bfs(self.map, robot_pos, pos)[1]
                target_path = run_bfs(self.map, pos, (pkg[2], pkg[3]))[1]
                len_path = start_path + target_path
                if state['time_step'] + len_path <= pkg[5]:
                    G.add_edge(f'r{robot_id}', node_name, capacity=1, weight=len_path)
        
        flow_dict = nx.max_flow_min_cost(G, 's', 't')
        assign = {}
        for robot_id in robots_free:
            for pos in valid_pos_package:
                node_name = f'p_{pos[0]}_{pos[1]}'
                if flow_dict[f'r{robot_id}'].get(node_name, 0) > 0:
                    assign[robot_id] = valid_pos_package[pos][0]
                    break
        return assign

    def get_actions(self, state):
        self.state = state
        self.update_packages(state)
        actions = []
        robots_free = [i for i in range(self.n_robots) if self.robots[i][2] == 0]
        assign = self.optimal_assign(state, robots_free)

        for robot_id in range(self.n_robots):
            robot_pos = (self.robots[robot_id][0], self.robots[robot_id][1])
            if robot_id in self.count_repeat:
                if self.last_move[robot_id] == robot_pos:
                    self.count_repeat[robot_id] += 1
                else:
                    self.count_repeat[robot_id] = 0
                self.last_move[robot_id] = robot_pos

            if self.count_repeat.get(robot_id, 0) >= self.NUM_REPEAT:
                moves = ['U', 'D', 'L', 'R']
                random.shuffle(moves)
                actions.append((moves[0], '0'))
                continue

            if robot_id in assign:
                package_id = assign[robot_id]
                self.packages_owned[package_id] = robot_id
                pkg = self.packages[package_id]
                target_pos = (pkg[0], pkg[1])
                move, dist = run_bfs(self.map, robot_pos, target_pos)
                pkg_act = '1' if dist == 0 else '0'
                actions.append((move, pkg_act))
            elif self.robots[robot_id][2] != 0:
                package_id = self.robots[robot_id][2]
                pkg = self.packages[package_id]
                target_pos = (pkg[2], pkg[3])
                move, dist = run_bfs(self.map, robot_pos, target_pos)
                pkg_act = '2' if dist == 0 else '0'
                actions.append((move, pkg_act))
            else:
                actions.append(('S', '0'))

        self.robots = [(r[0]-1, r[1]-1, r[2]) for r in state['robots']]
        return actions
