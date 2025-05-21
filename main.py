from env import Environment
from OptimizedFlowAgent import OptimizedFlowAgent
from greedyagent import GreedyAgents
from deepq_agent import DeepQEnv, train_deepq, test_deepq
import numpy as np
import csv
import os
from stable_baselines3 import PPO

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning for Delivery")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=10, help="Number of packages")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--max_time_steps", type=int, default=1000, help="Maximum time steps for the environment")
    parser.add_argument("--map", type=str, default="maps/map1.txt", help="Map name")
    parser.add_argument("--agent", type=str, default="all", choices=["greedy", "maxflow", "deepq", "all"], help="Agent to run")

    args = parser.parse_args()
    np.random.seed(args.seed)

    configs = [
        {"map": "maps/map1.txt", "num_agents": 5, "n_packages": 100},
        {"map": "maps/map2.txt", "num_agents": 5, "n_packages": 100},
        {"map": "maps/map3.txt", "num_agents": 5, "n_packages": 500},
        {"map": "maps/map4.txt", "num_agents": 10, "n_packages": 500},
        {"map": "maps/map5.txt", "num_agents": 10, "n_packages": 1000},
    ]

    # Tạo file results.csv nếu chưa có
    results_file = "results.csv"
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Agent", "Map", "Num_Agents", "Num_Packages", "Total_Reward", "Delivered_Packages", "Total_Steps"])

    for config in configs:
        if args.agent in ["greedy", "all"]:
            env = Environment(
                map_file=config["map"],
                max_time_steps=args.max_time_steps,
                n_robots=config["num_agents"],
                n_packages=config["n_packages"],
                seed=args.seed
            )
            state = env.reset()
            agents = GreedyAgents()
            agents.init_agents(state)
            
            done = False
            t = 0
            delivered_packages = 0
            
            while not done:
                actions = agents.get_actions(state)
                next_state, reward, done, infos = env.step(actions)
                state = next_state
                if 'total_reward' in infos:
                    delivered_packages = len([p for p in env.packages if p.status == 'delivered'])
                t += 1

            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "GreedyAgent",
                    config["map"],
                    config["num_agents"],
                    config["n_packages"],
                    infos.get('total_reward', 0),
                    delivered_packages,
                    infos.get('total_time_steps', t)
                ])

            print(f"Agent: GreedyAgent, Map: {config['map']}")
            print(f"Total reward: {infos.get('total_reward', 0)}")
            print(f"Delivered packages: {delivered_packages}")
            print(f"Total time steps: {infos.get('total_time_steps', t)}")

        if args.agent in ["maxflow", "all"]:
            env = Environment(
                map_file=config["map"],
                max_time_steps=args.max_time_steps,
                n_robots=config["num_agents"],
                n_packages=config["n_packages"],
                seed=args.seed
            )
            state = env.reset()
            agents = OptimizedFlowAgent()
            agents.init_agents(state)
            
            done = False
            t = 0
            delivered_packages = 0
            
            while not done:
                actions = agents.get_actions(state)
                next_state, reward, done, infos = env.step(actions)
                state = next_state
                if 'total_reward' in infos:
                    delivered_packages = len([p for p in env.packages if p.status == 'delivered'])
                t += 1

            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "OptimizedFlowAgent",
                    config["map"],
                    config["num_agents"],
                    config["n_packages"],
                    infos.get('total_reward', 0),
                    delivered_packages,
                    infos.get('total_time_steps', t)
                ])

            print(f"Agent: OptimizedFlowAgent, Map: {config['map']}")
            print(f"Total reward: {infos.get('total_reward', 0)}")
            print(f"Delivered packages: {delivered_packages}")
            print(f"Total time steps: {infos.get('total_time_steps', t)}")

        if args.agent in ["deepq", "all"]:
            deepq_env = DeepQEnv(
                map_file=config["map"],
                max_time_steps=args.max_time_steps,
                n_robots=config["num_agents"],
                n_packages=config["n_packages"],
                move_cost=-0.01,
                delivery_reward=10.0,
                delay_reward=1.0,
                seed=args.seed
            )
            model_file = f"deepq_outputs/ppo_delivery_{config['map'].replace('/', '_')}"  # Lưu vào deepq_outputs
            if not os.path.exists(model_file + ".zip"):
                model = train_deepq(
                    map_file=config["map"],
                    num_agents=config["num_agents"],
                    n_packages=config["n_packages"],
                    max_time_steps=args.max_time_steps,
                    seed=args.seed
                )
            else:
                model = PPO.load(model_file)
            
            total_reward, delivered_packages, total_steps = test_deepq(model, deepq_env)
            
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "DeepQAgent",
                    config["map"],
                    config["num_agents"],
                    config["n_packages"],
                    total_reward,
                    delivered_packages,
                    total_steps
                ])

            print(f"Agent: DeepQAgent, Map: {config['map']}")
            print(f"Total reward: {total_reward}")
            print(f"Delivered packages: {delivered_packages}")
            print(f"Total time steps: {total_steps}")