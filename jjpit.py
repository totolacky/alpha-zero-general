import JanggiArena
from JanggiMCTS import JanggiMCTS
from janggi.JanggiGame import JanggiGame
from janggi.JanggiPlayers import *
from janggi.pytorch.NNet import NNetWrapper as NNet
import random
import pickle, requests


import numpy as np
from utils import *
from JanggiMainConstants import request_base_url

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

ibs = pickle.loads(requests.get(url = request_base_url+"/getIBS").content)

g = JanggiGame(0, 0, ibs)

rp = []
gp = []

rp_rate = []
gp_rate = []

play_num = 20
gpu_num = 1

# checkpoints = [40, 100, 150, 202, 259, 310, 370, 430, 490, 560, 630, 708, 781, 860, 943, 1029, 1118]
# checkpoints = [17, 52, 55, 70, 78, 96, 107, 138, 162, 173, 180, 188, 211, 220, 234]
ref = 1987
checkpoints = [1218]

n2 = NNet(g)
n2.load_checkpoint("./mnt/sds/", "checkpoint_"+str(ref)+".pickle")
args2 = dotdict({'numMCTSSims': 120, 'cpuct':1.0})
mcts2 = JanggiMCTS(g, n2, args2)
p2 = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

for i in checkpoints:
    print("Testing Checkpoint "+str(i))
    n1 = NNet(g)
    n1.load_checkpoint("./mnt/sds/", "checkpoint_"+str(i)+".pickle")
    # cp_name = "./mnt/sds/sd_"+str(i)+".pickle"
    # with open(cp_name, 'rb') as handle:
    #     state_dict = pickle.load(handle)
    # n1.nnet.load_state_dict(state_dict)

    args1 = dotdict({'numMCTSSims': 120, 'cpuct':1.0})
    mcts1 = JanggiMCTS(g, n1, args1)
    p1 = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    #p2 = RandomPlayer(g).play

    arena = JanggiArena.JanggiArena(p1, p2, g, display=JanggiGame.display)
    rpres = arena.playGames(play_num, verbose=False)
    print('CP self-play x'+str(i)+' vs CP self-play '+str(ref)+': (Win/Lose/Draw) = '+str(rpres))
    rp.append(rpres)
    rp_rate.append((rpres[0]+rpres[2]/2)/play_num*100)

    requests.post(url = request_base_url+"/postPerf", data = pickle.dumps((i, (rpres[0]+rpres[2]/2)/play_num*100)))

print("RP:"+str(rp))
print('RP Rate:'+str( rp_rate))

# human_vs_cpu = True

# # g = JanggiGame(random.randint(0, 4), random.randint(0, 4))
# g = JanggiGame(0, 0)

# # all players
# rp = RandomPlayer(g).play
# gp = GreedyJanggiPlayer(g).play
# hp = HumanJanggiPlayer(g).play

# # # nnet players
# # n1 = NNet(g)
# # n1.load_checkpoint('./temp/','checkpoint_5.pth.tar')

# # args1 = dotdict({'numMCTSSims': 800, 'cpuct':1.0})
# # mcts1 = JanggiMCTS(g, n1, args1)
# # n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# # if human_vs_cpu:
# #     player2 = hp
# # else:
# #     n2 = NNet(g)
# #     n2.load_checkpoint('./pretrained_models/janggi/pytorch/', '8x8_100checkpoints_best.pth.tar')
# #     args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
# #     mcts2 = MCTS(g, n2, args2)
# #     n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

# #     player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

# n1p = hp
# player2 = gp

# arena = JanggiArena.JanggiArena(n1p, player2, g, display=JanggiGame.display)
# # arena = JanggiArena.JanggiArena(HumanJanggiPlayer(g).play, HumanJanggiPlayer(g).play, g, display=JanggiGame.display)

# print(arena.playGames(2, verbose=True))
