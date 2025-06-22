import time
import torch
import numpy as np
import copy
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from envs.stable_env import StableAgentEnv
from train.train import QNetwork
from data_define import data_used as d

def preprocess_observation(obs: int) -> torch.LongTensor:
    return torch.LongTensor([obs])

def visualize_map(grid: np.ndarray, agent_pos: tuple):
    H, W = np.array(grid).shape
    visual = ""
    for i in range(H):
        for j in range(W):
            if [i, j] == agent_pos:
                visual += " A "
            else:
                visual += " . "
        visual += "\n"
    print(visual)

def _judge_valid(pos, pre_pos, action, grid_size, allow_backtrack=0):
    temp_pos = copy.copy(pos)
    if action == 0:
        temp_pos[1] += 1
    elif action == 1:
        temp_pos[1] -= 1
    elif action == 2:
        temp_pos[0] -= 1
    else:
        temp_pos[0] += 1

    if (temp_pos[0] < 0 or temp_pos[1] < 0 or
        temp_pos[0] >= grid_size[0] or temp_pos[1] >= grid_size[1]):
        return False

    if not allow_backtrack and temp_pos == pre_pos:
        return False

    return True

def run_demo(grid_size, score_map, allow_backtrack=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== DEMO BEGIN =====")

    env = StableAgentEnv(grid_size=grid_size,
                         score_map=score_map)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy_net = QNetwork(n_states, n_actions).to(device)
    policy_net.load_state_dict(torch.load(f"{d.save_dir_name}/q_learning_agent.pth",
                                          map_location=device))
    policy_net.eval()

    obs = env.reset()
    done = False
    step = 0
    reward_count = 0

    pos, pre_pos = env.get_position()
    trajectory = [[pos[0], pos[1]]]
    grid = env.get_current_map()
    visualize_map(grid, pos)

    while not done and step < 100:
        obs_t = preprocess_observation(obs).to(device)
        old_pos = copy.copy(pos)
        with torch.no_grad():
            q_values = policy_net(obs_t) 
            action = q_values.argmax(dim=1).item()

            while not _judge_valid(old_pos, pre_pos, action, grid_size, allow_backtrack):
                print(f"Action {action} invalid, re-choose...")
                q_values[0][action] = float("-inf")
                action = q_values.argmax(dim=1).item()

        next_obs, reward, done, info = env.step(action)
        reward_count += reward
        pre_pos = copy.copy(pos)
        pos, _ = env.get_position() 
        obs = next_obs 
        pre_pos = old_pos

        trajectory.append([pos[0], pos[1]])
        print(f"Step {step:02d}: Agent moved to {pos}, Reward: {reward:.2f}")
        grid = env.get_current_map() 
        visualize_map(grid, pos)

        time.sleep(0.3) 
        step += 1

    print("===== TRAJECTORY =====")
    print(" → ".join(map(str, trajectory)))
    print(f"REWARD: {reward_count}")
    print("===== DEMO END =====")

if __name__ == "__main__":
    from data_define import data_used
    
    score_map = copy.copy(data_used.score_map)
    grid_size = np.array(score_map).shape

    run_demo(grid_size=grid_size,
             score_map=score_map,
             allow_backtrack=0)
