from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .JanggiLogic import Board
import numpy as np
from JanggiConstants import *

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

    # Required
    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # return initial board (numpy board) with repetition count dictionary
        b = Board(self.c1, self.c2)
        return (b.pieces, b.b_params, b.rep_dict)

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
        return CONFIG_X*CONFIG_Y*CONFIG_A + 1

    # Required
    def getNextState(self, board, action):
        """
        Input:
            board: current board (pieces)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
        """
        # if player takes action on board, return next (board,player)
        # action must be a valid move

        b = Board(self.c1, self.c2)
        b.pieces = np.copy(board[0])
        b.b_params = np.copy(board[1])
        b.rep_dict = board[2].copy() # Current board will be added in execute_move()
        
        move = (int(action/(CONFIG_X*CONFIG_Y)), int((action%(CONFIG_X*CONFIG_Y))/CONFIG_Y), action%CONFIG_Y) # (actionType, xCoord, yCoord)
        b.execute_move(move)
        return (b.pieces, b.b_params, b.rep_dict)

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

        b = Board(self.c1, self.c2)
        b.pieces = np.copy(board[0])
        b.b_params = np.copy(board[1])
        b.rep_dict = board[2].copy()

        legalMoves =  b.get_legal_moves()
        for a, x, y in legalMoves:
            valids[a*(CONFIG_X*CONFIG_Y) + x * CONFIG_Y + y] = 1

        return np.array(valids)

    # Required
    def getGameEnded(self, board):
        """
        Input:
            board: current board

        Returns:
            r: 0 if game has not ended.
            Normalized Cho score between -1 ~ 1 if game is over
        """
        # return 0 if not ended, 1 if player Cho won, -1 if player Cho lost
        b = Board(self.c1, self.c2)
        b.pieces = np.copy(board[0])
        b.b_params = np.copy(board[1])
        b.rep_dict = board[2].copy()

        return b.game_ended()

    # Required
    def stringRepresentation(self, board):
        """
        Input:
            board: current board (pieces, b_params, rep_dict)

        Returns:
            boardString: a quick conversion of board (pieces, cur_player, move_cnt) to a string format.
                         Required by MCTS for hashing.
        """
        return np.array([board[0], board[1][N_CUR_PLAYER], board[1][N_MOVE_CNT]]).tostring()

    def getScore(self, board, player):
        b = Board(self.c1, self.c2)
        b.pieces = np.copy(board[0])
        b.b_params = np.copy(board[1])
        b.rep_dict = board[2].copy()

        if player == PLAYER_HAN:
            return b.b_params[N_HAN_SCORE] - b.b_params[N_CHO_SCORE]
        else:
            return b.b_params[N_CHO_SCORE] - b.b_params[N_HAN_SCORE]

    @staticmethod
    def display(board):
        print("   ┌-------------------┐")
        for i in range(10):
            y = 9-i
            print(" ", end="")
            print(y, end=" | ")
            for j in range(9):
                x = j
                print(JanggiGame.square_content[board[0][0][x][y]], end="")
                if (x == 2 or x == 5) and (y >= 7 or y <= 2):
                    print("|", end="")
                else:
                    print(" ", end="")
            print("|")
        print("   └-------------------┘")
        print("     0 1 2 3 4 5 6 7 8")
