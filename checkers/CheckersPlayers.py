import numpy as np

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanCheckersPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                ((x,y),(z,w)) = action2move(i)
                print("[("+str(x)+","+str(y)+"),("+str(z)+","+str(w)+")] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 4:
                try:
                    x,y,z,w = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = move2action(x,y,z,w)
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a


class GreedyCheckersPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]

def action2move(action):
    multiplier = (action//4)%2+1
    xval = (action//8)//(n//2)
    yval = (action//8)%(n//2)*2 + xval%2
    return ((xval, yval), ((-1+2*(action%2))*multiplier, (-1+2*((action%4)//2))*multiplier))

def move2action(x, y, z, w):
    direction = (z+2*w+3)//2 if abs(z) == 1 else (z//2+w+3)//2+4
    return (n//2*x + y//2) * 8 + direction
