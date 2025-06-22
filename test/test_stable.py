import os
import sys
import time
import torch
import copy

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from train.train import QNetwork
from envs.stable_env import StableAgentEnv
from data_define import data_used as d

def preprocess_observation(obs: int) -> torch.LongTensor:
    return torch.LongTensor([obs])

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
    
    env = StableAgentEnv(grid_size=grid_size,
                         score_map=score_map)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy_net = QNetwork(n_states, n_actions).to(device)
    project_root = os.path.dirname(os.path.dirname(__file__))
    policy_net.load_state_dict(torch.load(f"{project_root}/{d.save_dir_name}/stable_agent.pth",
                                          map_location=device))
    policy_net.eval()

    obs = env.reset()
    done = False
    reward_count = 0

    pos, pre_pos = env.get_position()

    success_count = 0
    
    episode = 0
    
    while episode < 100:
        obs_t = preprocess_observation(obs).to(device)
        old_pos = copy.copy(pos)
        
        with torch.no_grad():
            q_values = policy_net(obs_t)
            action = q_values.argmax(dim=1).item()

            while not _judge_valid(old_pos, pre_pos, action, grid_size, allow_backtrack):
                q_values[0][action] = float("-inf")
                action = q_values.argmax(dim=1).item()

        next_obs, reward, done, info = env.step(action)
        reward_count += reward

        pre_pos = old_pos
        pos, _ = env.get_position()

        obs = next_obs

        if done:
            if reward_count == 4:
                success_count += 1
            
            obs = env.reset()
            pos, _ = env.get_position()
            pre_pos = copy.copy(pos)
            episode += 1
            reward_count = 0
    
    return success_count / 100
    
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    success_rate = run_demo([4,4], d.score_map, allow_backtrack=d.allow_backtrack)
    print(f"Success Rate: {success_rate}")