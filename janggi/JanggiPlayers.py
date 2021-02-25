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
        for i in range(len(valid)):
            if valid[i]:
                a,x,y = (int(i/(CONFIG_X*CONFIG_Y)), int((i%(CONFIG_X*CONFIG_Y))/CONFIG_Y), i%CONFIG_Y)
                dx,dy = Board._action_to_dxdy(a)
                print("[(",x,y,")(",x+dx,y+dy, end=")] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 4:
                try:
                    x,y,nx,ny = [int(i) for i in input_a]
                    a = Board._dxdy_to_action(nx-x, ny-y)
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
            nextBoard, _ = self.game.getNextState(board, a)
            score = self.game.getScore(nextBoard)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
