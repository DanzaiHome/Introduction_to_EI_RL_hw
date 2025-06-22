import gym
from gym import spaces
import numpy as np
import random
import copy
import sys
import os
import numbers

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from data_define import data_used as d

class FlexibleAgentEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(FlexibleAgentEnv, self).__init__()

        self.grid_size = d.flexible_grid_size
        assert len(self.grid_size) ==  2, 'grid_size must belike [x,y]'
        self.n_rows, self.n_cols = self.grid_size
        self.current_step = 0
        
        # Observation space
        map_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.grid_size, dtype=np.int32)
        agent_position_space = spaces.Discrete(self.n_rows * self.n_cols)
        self.observation_space = spaces.Dict({
            "score_map": map_observation_space,
            "position": agent_position_space
        })
        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)

        self.agent_pos = [0,0]
        self.goal_pos = [self.n_rows - 1, self.n_cols - 1]
        
        self.passed_map = np.zeros(shape=self.grid_size)
        self.passed_map[0][0] = 1 
        
        reward_lower_bound = copy.copy(d.reward_lower_bound)
        reward_upper_bound = copy.copy(d.reward_upper_bound)
        assert reward_lower_bound <= reward_upper_bound, "Error: reward_lower_bound must <= reward_upper_bound"
        assert reward_upper_bound < -d.cost_each_step, "Error: inf answer may occur: reward_upper_bound >= -cost_each_step"
        self.reward_lower_bound = reward_lower_bound
        self.reward_upper_bound = reward_upper_bound
        self.score_map = np.random.randint(low=reward_lower_bound, high=reward_upper_bound, size=self.grid_size, dtype=np.int32)
        self.pre_pos = np.zeros(shape=(2,))
        
        self.score_record = np.full(shape=self.grid_size, fill_value=np.inf)
        self.reward_record = 0
        
        self.reset()
        
    def reset(self):
        self.agent_pos = [0,0]
        self.pre_pos = [0,0]
        self.score_map = np.random.randint(low=self.reward_lower_bound, high=self.reward_upper_bound, size=self.grid_size, dtype=np.int32)
        self.score_record = np.full(shape=self.grid_size, fill_value=np.inf)
        self.reward_record = 0
        return self._get_state()

    def step(self, action):
        done = False
        reward = d.cost_each_step
        
        temp_pos = copy.copy(self.agent_pos)
        self._deploy_action(action)
        if(self.agent_pos[0] < 0 or 
           self.agent_pos[1] < 0 or 
           self.agent_pos[0] >= self.n_rows or 
           self.agent_pos[1] >= self.n_cols or 
           (d.allow_backtrack == 0 and self.agent_pos == self.pre_pos)):
            self.agent_pos = temp_pos
        else:
            if(self.agent_pos == self.goal_pos):
                reward = reward + d.reward_at_destination
                done = True
            else:
                reward = reward + self.score_map[self.agent_pos[0]][self.agent_pos[1]]
                self.pre_pos = copy.copy(self.agent_pos)
                self.passed_map[self.agent_pos[0]][self.agent_pos[1]] = 1
                
        if done:
           self.current_step = 0
                
        self.reward_record += reward
        
        if self.score_record[self.agent_pos[0]][self.agent_pos[1]] < self.reward_record:
            print(f"\nThe answer is infinity.")
            sys.exit(1)
        
        self.score_record[self.agent_pos[0]][self.agent_pos[1]] = self.reward_record
            
        return self._get_state(), reward, done, {}

    def render(self, mode="human"):
        # Unnecessary for this project
        pass
    
    def get_grid(self):
        return self.score_map
    
    def _get_state(self):
        return {"score_map": self.score_map, 
                "position": self.agent_pos[0] * self.n_cols + self.agent_pos[1]}

    def _deploy_action(self, action):
        assert isinstance(action, numbers.Integral), 'Invalid action type, which must be an integer.'
        assert action in [0,1,2,3], 'Invalid action value, which must be in [0,1,2,3]'
        if(action == 0):
            self.agent_pos[1] += 1
        elif(action == 1):
            self.agent_pos[1] -= 1
        elif(action == 2):
            self.agent_pos[0] -= 1
        else:
            self.agent_pos[0] += 1
            
    def _get_position(self):
        # Used to transfer position information to the demo
        return self.agent_pos, self.pre_pos