# Piece numbers
NK = 1
NC = 2
NP = 3
NM = 4
NX = 5
NS = 6
NB = 7

# b_params
N_HAN_PCS = 0
N_CHO_PCS = 1
N_MOVE_CNT = 2
N_CUR_PLAYER = 3
N_HAN_SCORE = 4
N_CHO_SCORE = 5
N_CAPTURED = 6
N_IS_BIC = 7

# Players
PLAYER_CHO = 0
PLAYER_HAN = 1

# Masks
ATTACK_MASK = 63998 # 11111/00/11/11/11/11/0

# Board/State related constants
CONFIG_X = 9     # board width
CONFIG_Y = 10    # board height
CONFIG_T = 4     # number of timesteps recorded
CONFIG_L = 2     # number of aux. info (current player & move count)
CONFIG_A = 58    # number of action planes (excluding turn skip)