class data_used():
    ##############
    # Basic env: #
    ##############
    allow_backtrack = 1
    cost_each_step = -4
    reward_at_destination = 20
    
    # Training hyperparameters
    EPISODES = 100
    BATCH_SIZE = 64
    GAMMA = 1
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.995
    TARGET_UPDATE = 10
    MEMORY_CAPACITY = 10000
    LR = 2e-5
    
    ###############
    # Stable env: #
    ###############
    score_map = [[0,5,3,0],
                 [4,-2,1,-3],
                 [0,-4,6,5],
                 [3,-2,-1,0]]

    #####################
    # For flexible env: #
    #####################
    reward_lower_bound = -4
    reward_upper_bound = 4
    flexible_grid_size = [4,4]
    
    # Special Training Hyperparameters
    FLEXIBLE_TIMESTEPS = 80_000
    
    # For Test
    # If you want to make the map random, set it None
    test_grid = [[0,5,3,0],
                 [4,-2,1,-3],
                 [0,-4,6,5],
                 [3,-2,-1,0]]
    
    
    ################
    # Other Config #
    ################
    
    # relative to /RL
    save_dir_name = "checkpoint"
    