import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict
from tqdm import tqdm
import copy
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from data_define import data_used as d
from CustomNetwork.QNetwork import StableQNetwork as QNetwork

# Hyperparameters
EPISODES = d.EPISODES
BATCH_SIZE = d.BATCH_SIZE
GAMMA = d.GAMMA
EPS_START = d.EPS_START
EPS_END = d.EPS_END
EPS_DECAY = d.EPS_DECAY
TARGET_UPDATE = d.TARGET_UPDATE
MEMORY_CAPACITY = d.MEMORY_CAPACITY
LR = d.LR

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def preprocess_observation(obs: int) -> torch.LongTensor:
    return torch.LongTensor([obs])  # shape: [1]

def select_action(state: torch.LongTensor, 
                  policy_net: QNetwork, 
                  eps_threshold: float, 
                  n_actions: int,
                  pos=None,
                  pre_pos=None) -> int:
    if random.random() > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)  # shape [1, n_actions]
            action = q_values.argmax(dim=1).item()

            if pos is not None and pre_pos is not None:
                while not _judge_valid(pos=pos, pre_pos=pre_pos, action=action):
                    q_values[0][action] = -np.inf
                    action = q_values.argmax(dim=1).item()
    else:
        action = random.randrange(n_actions)
    return action

def optimize_model(memory: ReplayMemory,
                   policy_net: QNetwork,
                   target_net: QNetwork,
                   optimizer: optim.Optimizer,
                   device="cpu"):
    
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch_s, batch_a, batch_r, batch_s_next, batch_done = zip(*transitions)
    
    state_batch = torch.LongTensor(batch_s).to(device)      
    action_batch = torch.LongTensor(batch_a).to(device)     
    reward_batch = torch.FloatTensor(batch_r).to(device)   
    done_batch = torch.FloatTensor(batch_done).to(device) 
    
    non_final_mask = torch.tensor([ns is not None for ns in batch_s_next],
                                  dtype=torch.bool, device=device)
    non_final_next_states = [ns for ns in batch_s_next if ns is not None]
    if len(non_final_next_states) > 0:
        non_final_next_states_tensor = torch.LongTensor(non_final_next_states).to(device)
    else:
        non_final_next_states_tensor = None

    q_values_all = policy_net(state_batch)     
    current_q = q_values_all.gather(1, action_batch.unsqueeze(1)).squeeze(1)

    next_q = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if non_final_next_states_tensor is not None:
            next_q_values_all = target_net(non_final_next_states_tensor) 
            max_next_q = next_q_values_all.max(dim=1)[0] 
            next_q[non_final_mask] = max_next_q
        
        target_q = reward_batch + (GAMMA * next_q * (1 - done_batch))

    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def _judge_valid(pos, pre_pos, action):

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
        temp_pos[0] >= grid_size[0] or temp_pos[1] >= grid_size[1] or temp_pos == pre_pos):
        return False
    return True

def main(grid_size,
         score_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    from envs.stable_env import StableAgentEnv
    env = StableAgentEnv(grid_size=grid_size,
                         score_map=score_map)

    n_actions = env.action_space.n
    n_states = env.observation_space.n
    
    policy_net = QNetwork(n_states, n_actions).to(device)
    target_net = QNetwork(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_CAPACITY)
    
    eps_threshold = EPS_START
    train_bar = tqdm(range(EPISODES), desc="Training", unit="episode")
    
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    for episode in train_bar:
        obs = env.reset()
        state = obs
        done = False
        total_reward = 0.0
        
        while not done:
            state_t = preprocess_observation(state).to(device)

            pos, pre_pos = env.get_position()
            
            action = select_action(state_t, policy_net, eps_threshold, n_actions,
                                   pos, pre_pos)
            
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if not done:
                next_state = next_obs
            else:
                next_state = None
            
            memory.push(state, action, reward, next_state, done)
            state = next_obs if not done else None
            
            optimize_model(memory, policy_net, target_net, optimizer, device)
        
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        eps_threshold = max(EPS_END, eps_threshold * EPS_DECAY)
        
        train_bar.set_postfix({
            "Episode": episode,
            "Reward": f"{total_reward:.2f}",
            "Eps": f"{eps_threshold:.2f}"
        })
        
        if episode > 1000 and episode % 10 == 0:
            torch.save(policy_net.state_dict(),f"{parent_dir}/{d.save_dir_name}/stable_agent.pth")
            print()
            print(f"Model saved as '{parent_dir}/{d.save_dir_name}/stable_agent.pth'")

    torch.save(policy_net.state_dict(),f"{parent_dir}/{d.save_dir_name}/stable_agent.pth")
    print()
    print(f"Training finished. Model saved as '{parent_dir}/{d.save_dir_name}/stable_agent.pth'")



if __name__ == "__main__":
    score_map = copy.copy(d.score_map)
    grid_size = np.array(score_map).shape

    main(grid_size=grid_size,
         score_map=score_map)
