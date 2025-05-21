import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import LabelEncoder
from env import Environment
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

class DeepQEnv(gym.Env):
    def __init__(self, map_file, max_time_steps, n_robots, n_packages,
                 move_cost, delivery_reward, delay_reward, seed):
        super(DeepQEnv, self).__init__()
        self.env = Environment(map_file, max_time_steps, n_robots, n_packages,
                               move_cost, delivery_reward, delay_reward, seed)
        self.action_space = spaces.MultiDiscrete([5, 3] * self.env.n_robots)
        self.prev_raw_state = self.env.reset()
        self.n_robots = n_robots
        self.n_packages = n_packages
        self.max_time_steps = max_time_steps
        self.prev_positions = [None] * self.n_robots

        # Tính toán kích thước cố định cho observation_space
        obs_shape = 1 + (self.n_robots * 3) + (self.n_packages * 5) + (self.n_robots * 2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_shape,), dtype=np.float32)

        self.le1 = LabelEncoder()
        self.le2 = LabelEncoder()
        self.le1.fit(['S', 'L', 'R', 'U', 'D'])
        self.le2.fit(['0', '1', '2'])

    def convert_state(self, raw_state):
        time_step = raw_state['time_step']
        robots = raw_state['robots']
        packages = raw_state['packages']

        robot_features = []
        for rx, ry, carrying in robots:
            robot_features.extend([rx / self.env.n_rows, ry / self.env.n_cols, carrying / self.env.n_packages])

        package_features = []
        status_map = {pkg.package_id: pkg.status for pkg in self.env.packages}
        for package_id, sx, sy, tx, ty, start_time, deadline in packages:
            status = status_map.get(package_id, 'waiting')
            status_numeric = 0.0 if status == 'waiting' else (0.5 if status == 'in_transit' else 1.0)
            package_features.extend([
                sx / self.env.n_rows, sy / self.env.n_cols,
                tx / self.env.n_rows, ty / self.env.n_cols,
                status_numeric
            ])

        # Pad hoặc cắt package_features để có kích thước cố định
        features_per_package = 5
        expected_package_features = self.n_packages * features_per_package
        if len(package_features) < expected_package_features:
            package_features.extend([0.0] * (expected_package_features - len(package_features)))
        elif len(package_features) > expected_package_features:
            package_features = package_features[:expected_package_features]

        # Cải thiện không gian quan sát: Thêm khoảng cách cụ thể đến gói gần nhất hoặc mục tiêu
        distance_features = []
        for i, (rx, ry, carrying) in enumerate(robots):
            min_dist_x, min_dist_y = float('inf'), float('inf')
            for pkg in self.env.packages:
                if carrying == 0 and pkg.status == 'waiting' and self.env.t >= pkg.start_time:
                    dist_x = (rx - pkg.start[0]) / self.env.n_rows
                    dist_y = (ry - pkg.start[1]) / self.env.n_cols
                    if abs(dist_x) + abs(dist_y) < abs(min_dist_x) + abs(min_dist_y):
                        min_dist_x, min_dist_y = dist_x, dist_y
                elif carrying == pkg.package_id:
                    dist_x = (rx - pkg.target[0]) / self.env.n_rows
                    dist_y = (ry - pkg.target[1]) / self.env.n_cols
                    if abs(dist_x) + abs(dist_y) < abs(min_dist_x) + abs(min_dist_y):
                        min_dist_x, min_dist_y = dist_x, dist_y
            distance_features.extend([min_dist_x, min_dist_y])

        concatenated_state = np.concatenate([
            [time_step / self.env.max_time_steps],
            robot_features,
            package_features,
            distance_features
        ])
        return concatenated_state

    def reward_shaping(self, base_reward, actions, next_state):
        new_r = base_reward if base_reward is not None else 0.0
        for i in range(self.n_robots):
            robot = self.env.robots[i]
            move_act, pkg_act = actions[i]
            robot_pos = robot.position
            next_robots = next_state['robots']
            next_robot = next_robots[i]

            # Phạt nhẹ nếu đứng im
            if self.prev_positions[i] == robot_pos and move_act != 'S':
                new_r -= 0.1
            self.prev_positions[i] = robot_pos

            # Chỉ thưởng khi giao thành công (giữ nguyên logic)
            if pkg_act == '2' and robot.carrying != 0 and next_robot[2] == 0:
                new_r += 10.0
                pkg = self.env.packages[robot.carrying - 1]
                if self.env.t > pkg.deadline:
                    new_r -= 9.0

            # Phạt nhẹ nếu mang gói lâu
            if robot.carrying != 0:
                new_r -= 0.05

        return new_r

    def step(self, action):
        actions = []
        for i in range(self.n_robots):
            move_action = self.le1.inverse_transform([action[i * 2]])[0]
            package_action = self.le2.inverse_transform([action[i * 2 + 1]])[0]
            actions.append((move_action, package_action))

        # Đảm bảo hành động nhặt (1) và giao (2) được thực thi đúng
        processed_actions = []
        for i, (move, pkg_act) in enumerate(actions):
            robot = self.env.robots[i]
            robot_pos = robot.position

            # Kiểm tra điều kiện nhặt
            if pkg_act == '1':
                if robot.carrying != 0:
                    pkg_act = '0'
                else:
                    # Kiểm tra xem robot có ở đúng vị trí để nhặt không
                    for pkg in self.env.packages:
                        if pkg.status == 'waiting' and self.env.t >= pkg.start_time:
                            if robot_pos[0] == pkg.start[0] and robot_pos[1] == pkg.start[1]:
                                print(f"Robot {i} at position {robot_pos} attempts to pick up package {pkg.package_id}")
                                break
                    else:
                        pkg_act = '0'  # Không ở đúng vị trí để nhặt

            # Kiểm tra điều kiện giao
            elif pkg_act == '2':
                if robot.carrying == 0:
                    pkg_act = '0'
                else:
                    pkg = self.env.packages[robot.carrying - 1]
                    if robot_pos[0] == pkg.target[0] and robot_pos[1] == pkg.target[1]:
                        print(f"Robot {i} at position {robot_pos} attempts to deliver package {pkg.package_id}")
                    else:
                        pkg_act = '0'  # Không ở đúng vị trí để giao

            processed_actions.append((move, pkg_act))

        step_result = self.env.step(processed_actions)
        if len(step_result) == 4:
            raw_state, reward, terminated, truncated = step_result
            info = {}
        elif len(step_result) == 5:
            raw_state, reward, terminated, truncated, info = step_result
        else:
            raise ValueError(f"Expected 4 or 5 values from env.step, got {len(step_result)}")

        shaped_reward = self.reward_shaping(reward, processed_actions, raw_state)
        converted_state = self.convert_state(raw_state)
        self.prev_raw_state = raw_state

        return converted_state, shaped_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        raw_state = self.env.reset()
        converted_state = self.convert_state(raw_state)
        self.prev_raw_state = raw_state
        self.prev_positions = [None] * self.n_robots
        return converted_state, {}

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        pass

def train_deepq(map_file, num_agents, n_packages, max_time_steps, seed):
    env = DeepQEnv(map_file, max_time_steps, num_agents, n_packages,
                   move_cost=-0.01, delivery_reward=10.0, delay_reward=1.0, seed=seed)
    env = Monitor(env)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        n_steps=10,
        batch_size=10,
        gamma=0.99,
        ent_coef=0.1,
        verbose=1
    )
    eval_env = DeepQEnv(map_file, max_time_steps, num_agents, n_packages,
                        move_cost=-0.01, delivery_reward=10.0, delay_reward=1.0, seed=seed)
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./deepq_outputs/",
                                 log_path="./deepq_outputs/", eval_freq=5000,
                                 deterministic=True, render=False)

    # Tăng thời gian học
    model.learn(total_timesteps=10000, callback=eval_callback)
    model_save_path = f"deepq_outputs/ppo_delivery_{os.path.basename(map_file).split('.')[0]}"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.zip")
    return model

def test_deepq(model, env: DeepQEnv):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward_accumulated = 0.0
    step_count = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward_step, terminated, truncated, info = env.step(action)
        total_reward_accumulated += reward_step
        step_count += 1

        delivered_packages_count = len([p for p in env.env.packages if p.status in ['delivered', 'delivered_late']])
        print(f"Step {step_count}: Reward = {reward_step:.2f}, Total Reward = {total_reward_accumulated:.2f}, "
              f"Delivered Packages = {delivered_packages_count}")

    delivered_packages_count = len([p for p in env.env.packages if p.status in ['delivered', 'delivered_late']])
    total_time_steps_in_episode = env.env.t
    print(f"Test Results: Accumulated Shaped Reward = {total_reward_accumulated:.2f}, "
          f"Delivered Packages = {delivered_packages_count}, Total Time Steps = {total_time_steps_in_episode}")
    env.close()
    return total_reward_accumulated, delivered_packages_count, total_time_steps_in_episode

if __name__ == "__main__":
    model = train_deepq("maps/map1.txt", num_agents=2, n_packages=5, max_time_steps=100, seed=2025)
    test_env = DeepQEnv("maps/map1.txt", max_time_steps=100, n_robots=2, n_packages=5,
                        move_cost=-0.01, delivery_reward=10.0, delay_reward=1.0, seed=2025)
    test_deepq(model, test_env)