class data_used():
    ##############
    # Basic env: #
    ##############
    allow_backtrack = 0
    cost_each_step = -6
    reward_at_destination = 20
    
    # Training hyperparameters
    EPISODES = 40
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.995
    TARGET_UPDATE = 10
    MEMORY_CAPACITY = 10000
    LR = 1e-3
    
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
    reward_lower_bound = -3
    reward_upper_bound = 3
    flexible_grid_size = [5,5]
    
    # Special Training Hyperparameters
    FLEXIBLE_EPISODES = 500
    test_grid = [[0,5,3,0],
                 [4,-2,1,-3],
                 [0,-4,6,5],
                 [3,-2,-1,0]]
    
    
    ################
    # Other Config #
    ################
    
    save_dir_name = ""
    