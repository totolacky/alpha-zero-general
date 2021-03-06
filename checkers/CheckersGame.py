from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .CheckersLogic import Board
import numpy as np

class CheckersGame(Game):
    square_content = {
        -2: "V",
        -1: "X",
        +0: "-",
        +1: "O",
        +2: "D"
    }

    @staticmethod
    def getSquarePiece(piece):
        return CheckersGame.square_content[piece]

    def __init__(self, n):
        self.n = n
        self.count = 0

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return (np.array(b.pieces), b.count)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        # 2 possible moves per each of the 4 directions, starting at n*n/2 possible positions
        # (i,j) position ->
        return self.n*self.n*4+1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n*4:   # When you don't move
            return ((board[0], board[1]+1), -player)

        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.count = board[1]

        move = self.action2move(action)
        if player == -1:
            (x,y),(z,w) = move
            move = ((self.n-1-x, self.n-1-y),(-z, -w))
        b.execute_move(move, player)
        return ((b.pieces, board[1]+1), -player)

    ''' 
    Move direction represented by action%8:
    4   5
     0 1
      X
     2 3
    6   7
    Coordinate of X: (i,j) -> action = (n//2*i + j//2) * 8 + (action%8)
    '''

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.count = board[1]
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for ((x,y),(z,w)) in legalMoves:
            valids[self.move2action(x,y,z,w)]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.count = board[1]
        if board[1] > 150:
            return 0.01
        return b.game_over()

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        if player == 1:
            return board
        else:
            newB = np.copy(board[0])
            newB = -np.flip(newB, [0,1])
            return (newB, board[1])

    def getSymmetries(self, board, pi):
        # LR mirror only
        assert(len(pi) == self.getActionSize())  # 1 for pass
        l = [(CheckersGame.encodeBoard(board), pi)]

        newB = np.flip(np.copy(board[0]), 1)
        newPi = [0]*self.getActionSize()
        for i in range(self.getActionSize()-1):
            if i%2 == 0:
              newPi[i] = pi[i+1]
            else:
              newPi[i] = pi[i-1]
        newPi[self.getActionSize()-1] = pi[self.getActionSize()-1]
        l += [(CheckersGame.encodeBoard((newB, board[1])), newPi)]
        return l

    def stringRepresentation(self, board):
        return CheckersGame.encodeBoard(board).tostring()
        #return ','.join(str(item) for innerlist in board for item in innerlist)

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board[0] for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board[0])
        b.count = board[1]
        return b.countScore(player)

    @staticmethod
    def display(board):
        n = board[0].shape[0]      
        print("-----------------")
        for y in range(n):
            print("|"+str(n-1-y)+" |", end="")    # print the row #
            for x in range(n):
                piece = board[0][y][n-1-x]    # get the piece to print
                print(CheckersGame.square_content[piece], end=" ")
            print("|")

        print("-----------------")
        print("   |", end="")
        for y in range(n):
            print(y, end=" ")
        print("|")  

        print("   --------------")

    @staticmethod
    def encodeBoard(board):
        b, move_cnt = board
        b1 = np.copy(b) == 1
        b2 = np.copy(b) == 2
        b3 = np.copy(b) == -1
        b4 = np.copy(b) == -2

        n = b.shape[0]
        
        mc = [None]*n
        for i in range(n):
            mc[i] = [board[1]]*n
        mc = np.array(mc)

        res = np.concatenate((b1, b2, b3, b4, mc), 0)

        return res

    def action2move(self, action):
        multiplier = (action//4)%2+1
        xval = (action//8)//(self.n//2)
        yval = (action//8)%(self.n//2)*2 + xval%2
        return ((xval, yval), ((-1+2*(action%2))*multiplier, (-1+2*((action%4)//2))*multiplier))

    def move2action(self, x, y, z, w):
        direction = (z+2*w+3)//2 if abs(z) == 1 else (z//2+w+3)//2+4
        return (self.n//2*x + y//2) * 8 + direction
