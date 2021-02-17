from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .JanggiLogic import Board
import numpy as np

class JanggiGame(Game):
    square_content = {
        # 兵/卒: Soldiers (Byung)
        -7: "b",
        +7: "B",
        # 士: Guards (Sa)
        -6: "s",
        +6: "S",
        # 象: Elephants (Xiang)
        -5: "x",
        +5: "X",
        # 馬: Horses (Ma)
        -4: "m",
        +4: "M",
        # 包: Cannons (Po)
        -3: "p",
        +3: "P",
        # 車: Chariots (Cha)
        -2: "c",
        +2: "C",
        # 漢/楚: General (Goong)
        -1: "g",
        +1: "G",
        # Empty space
        +0: "-",
    }

    # Required
    def __init__(self, c1, c2):
        """
        Charim (initial board state):
            0: SMSM (象馬象馬)
            1: MSMS (馬象馬象)
            2: MSSM (馬象象馬)
            3: SMMS (象馬馬象)
        """
        self.c1 = c1
        self.c2 = c2

        self.nx = 9     # board width
        self.ny = 10    # board height
        self.ns = 16    # number of state planes
        self.na = 58    # number of action planes (excluding turn skip)

    # Required
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # return initial board (numpy board) with repetition count dictionary
        b = Board(self.c1, self.c2)
        d = {}
        return (np.array(b.pieces), d)

    # Required
    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        # (a,b) tuple
        return (9, 10)

    # Required
    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        # return number of actions
        return self.nx*self.ny*self.na + 1

    # Required
    def getNextState(self, board, player, action):
        """
        Input:
            board: current board (pieces)
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.nx*self.ny*self.na:
            return (board, -player)
        b = Board(self.c1, self.c2)
        b.pieces = np.copy(board)
        move = (int(action/(self.nx*self.ny)), int((action%(self.nx*self.ny))/self.ny), action%self.ny) # (actionType, xCoord, yCoord)
        b.execute_move(move, player)
        return (b.pieces, -player)

    # Required
    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    # Required
    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        # return state if player==1, else return -state if player==-1
        return player*board

    # Required
    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    # Required
    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(JanggiGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
