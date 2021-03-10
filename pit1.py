import Arena
from MCTS import MCTS
from checkers.CheckersGame import CheckersGame
from checkers.CheckersPlayers import *
from checkers.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = CheckersGame(6)

rp = RandomPlayer(g).play
gp = GreedyCheckersPlayer(g).play
hp = HumanCheckersPlayer(g).play

arena = Arena.Arena(gp, hp, g, display=CheckersGame.display)

print(arena.playGames(2, verbose=True))