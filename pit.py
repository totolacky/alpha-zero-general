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

# mini = True  # Play in 6x6 instead of the normal 8x8.
# human_vs_cpu = True

# if mini:
#     g = CheckersGame(6)
# else:
#     g = CheckersGame(8)

# # all players
# rp = RandomPlayer(g).play
# gp = GreedyCheckersPlayer(g).play
# hp = HumanCheckersPlayer(g).play



# # nnet players
# n1 = NNet(g)
# if mini:
# #     n1.load_checkpoint('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
#         n1.load_checkpoint('./temp/','latest.pth.tar')
# else:
#     n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
# args1 = dotdict({'numMCTSSims': 80, 'cpuct':1.0})
# mcts1 = MCTS(g, n1, args1)
# n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# if human_vs_cpu:
#     player2 = gp
# else:
#     n2 = NNet(g)
#     n2.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
#     args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
#     mcts2 = MCTS(g, n2, args2)
#     n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

#     player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

g = CheckersGame(6)

rp = []
gp = []

for i in range(15):
    n1 = NNet(g)
    n1.load_checkpoint('./checkers/pytorch/checkpoints/temp/','checkpoint_'+str(2*(i+1))+'.pth.tar')
    args1 = dotdict({'numMCTSSims': 80, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    p1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    p2 = RandomPlayer(g).play

    arena = Arena.Arena(p1, p2, g, display=CheckersGame.display)
    res = arena.playGames(30, verbose=False)
    print('CP'+str(2*i)+' vs RP: (Win/Lose/Draw) = '+str(res))
    rp.append(res[0])

    p2 = GreedyCheckersPlayer(g).play

    arena = Arena.Arena(p1, p2, g, display=CheckersGame.display)
    res = arena.playGames(30, verbose=False)
    print('CP'+str(2*i)+' vs GP: (Win/Lose/Draw) = '+str(res))
    gp.append(res[0])

print(rp)
print(gp)

# arena = Arena.Arena(n1p, player2, g, display=CheckersGame.display)

# print(arena.playGames(10, verbose=True))