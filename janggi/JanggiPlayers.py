import numpy as np
from .JanggiConstants import *
from .JanggiLogic import Board

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanJanggiPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board)
        cnt = 0
        for i in range(len(valid)):
            if valid[i]:
                cnt = cnt+1
                a,x,y = (int(i/(CONFIG_X*CONFIG_Y)), int((i%(CONFIG_X*CONFIG_Y))/CONFIG_Y), i%CONFIG_Y)
                dx,dy = Board._action_to_dxdy(a)
                print("[("+str(x)+","+str(y)+"): ("+str(dx)+","+str(dy)+")]", end="\t")
                if cnt%6 == 0:
                    print("")
        print("")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 4:
                try:
                    x,y,dx,dy = [int(i) for i in input_a]
                    a = Board._dxdy_to_action(dx, dy)
                    action = a*(CONFIG_X*CONFIG_Y) + x * CONFIG_Y + y
                    if valid[action]:
                        break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return action


class GreedyJanggiPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard = self.game.getNextState(board, a)
            score = self.game.getScore(nextBoard)
            candidates += [(-score, a)]
        candidates.sort()

        optimals = []
        bestScore = candidates[0][0]
        for a in candidates:
            if a[0] == bestScore:
                optimals.append(a[1])
        return optimals[np.random.randint(len(optimals))]
