import JanggiArena
from JanggiMCTS import JanggiMCTS
from janggi.JanggiGame import JanggiGame
from janggi.JanggiPlayers import *
from janggi.pytorch.NNet import NNetWrapper as NNet
import random


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

# g = JanggiGame(random.randint(0, 4), random.randint(0, 4))
g = JanggiGame(0, 0)

# all players
rp = RandomPlayer(g).play
gp = GreedyJanggiPlayer(g).play
hp = HumanJanggiPlayer(g).play

arena = JanggiArena.JanggiArena(hp, hp, g, display=JanggiGame.display)
# arena = JanggiArena.JanggiArena(HumanJanggiPlayer(g).play, HumanJanggiPlayer(g).play, g, display=JanggiGame.display)

print(arena.playGames(2, verbose=True))