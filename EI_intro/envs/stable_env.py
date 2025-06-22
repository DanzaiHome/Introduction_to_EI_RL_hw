import gym
from gym import spaces
import numpy as np
import copy
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from data_define import data_used

class StableAgentEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, 
                 score_map: np.array,
                 grid_size):
        super(StableAgentEnv, self).__init__()

        assert len(grid_size) ==  2, 'grid_size must belike [x,y]'
        self.grid_size = grid_size
        self.n_rows, self.n_cols = grid_size
        self.score_map = score_map
        
        self.observation_space = spaces.Discrete(self.n_rows * self.n_cols)

        # Action: an integer: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)

        self.agent_pos = [0,0]
        self.goal_pos = [self.n_rows - 1, self.n_cols - 1]
        
        self.passed_map = np.zeros(shape=grid_size)
        self.passed_map[0][0] = 1 
        self.pre_pos = np.zeros(shape=(2,))
        
        self.score_record = np.ones((self.n_rows, self.n_cols)) * np.inf
        self.reward_record = 0
        
        self.reset()
        
    #def reset(self):
    #    self.agent_pos = [0,0]
    #    self.pre_pos = [0,0]
    #    return self._get_state()

    def step(self, action):
        done = False
        reward = data_used.cost_each_step
        
        temp_pos = copy.copy(self.agent_pos)
        self._deploy_action(action)
        if(self.agent_pos[0] < 0 or 
           self.agent_pos[1] < 0 or 
           self.agent_pos[0] >= self.n_rows or 
           self.agent_pos[1] >= self.n_cols or 
           (data_used.allow_backtrack == 0 and self.agent_pos == self.pre_pos)):
            self.agent_pos = temp_pos
        else:
            if(self.agent_pos == self.goal_pos):
                reward = reward + data_used.reward_at_destination
                done = True
            else:
                reward = reward + self.score_map[self.agent_pos[0]][self.agent_pos[1]]
                self.pre_pos = copy.copy(self.agent_pos)
                self.passed_map[self.agent_pos[0]][self.agent_pos[1]] = 1
        if done:
           self.current_step = 0
                
        self.reward_record += reward
        
        if self.score_record[self.agent_pos[0]][self.agent_pos[1]] < self.reward_record:
            print(f"\nThe result is infinity.")
            sys.exit(1)
        
        self.score_record[self.agent_pos[0]][self.agent_pos[1]] = self.reward_record
            
        return self._get_state(), reward, done, {}

    def render(self, mode="human"):
        # Unnecessary for this project
        pass
    
    def reset(self):
        self.agent_pos = [0,0]
        self.pre_pos = [0,0]
        self.score_record = self.score_record = np.ones((self.n_rows, self.n_cols)) * np.inf

        return self._get_state()
    
    def _deploy_action(self, action):
        assert isinstance(action, int), 'Invalid action type, which must be an integer.'
        assert action in [0,1,2,3], 'Invalid action value, which must be in [0,1,2,3]'
        if(action == 0):
            self.agent_pos[1] += 1
        elif(action == 1):
            self.agent_pos[1] -= 1
        elif(action == 2):
            self.agent_pos[0] -= 1
        else:
            self.agent_pos[0] += 1
            
    def get_position(self):
        return self.agent_pos, self.pre_pos
    
    def get_current_map(self):
        return self.score_map
    
    def _get_state(self):
        return self.n_cols * self.agent_pos[0] + self.agent_pos[1]