from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .JanggiLogic import Board
import numpy as np
from .JanggiConstants import *

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

    def getStateSize(self):
        """
        Returns:
            stateSize: number planes in 1 state
        """
        # return number of actions
        return CONFIG_M*CONFIG_T+CONFIG_L

    # Required
    def getNextState(self, board, action):
        """
        Input:
            board: current board (pieces)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            curPlayer: the player to play
        """
        # if player takes action on board, return next (board,player)
        # action must be a valid move

        b = Board(self.c1, self.c2, True)
        b.pieces = np.copy(board[0])
        b.b_params = np.copy(board[1])
        b.rep_dict = board[2].copy() # Current board will be added in execute_move()
        
        move = (int(action/(CONFIG_X*CONFIG_Y)), int((action%(CONFIG_X*CONFIG_Y))/CONFIG_Y), action%CONFIG_Y) # (actionType, xCoord, yCoord)
        b.execute_move(move)
        return (b.pieces, b.b_params, b.rep_dict)

    # Required
    def getValidMoves(self, board):
        """
        Input:
            board: current board

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board,
                        0 for invalid moves
        """
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()

        b = Board(self.c1, self.c2, True)
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
            Cho score (+-1) if game is over
        """
        # return 0 if not ended, 1 if player Cho won, -1 if player Cho lost
        b = Board(self.c1, self.c2, True)
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
        canonical_board = board[0]
        if (board[1][N_CUR_PLAYER] == PLAYER_HAN):
            canonical_board = np.flip(canonical_board, [1, 2])
        
        # Add player/move count info
        tmp = [0] * CONFIG_X
        for i in range(CONFIG_X):
            tmp[i] = [0] * CONFIG_Y
        tmp[0][0] = board[1][N_CUR_PLAYER]
        tmp[0][1] = board[1][N_MOVE_CNT]
        canonical_board = np.concatenate((canonical_board, [tmp]), 0)

        return np.array(canonical_board).tostring()

    def getScore(self, board):
        b = Board(self.c1, self.c2, True)
        b.pieces = np.copy(board[0])
        b.b_params = np.copy(board[1])
        b.rep_dict = board[2].copy()

        player = b.b_params[N_CUR_PLAYER]

        if player == PLAYER_HAN:
            return b.b_params[N_HAN_SCORE] - b.b_params[N_CHO_SCORE]
        else:
            return b.b_params[N_CHO_SCORE] - b.b_params[N_HAN_SCORE]

    def getPlayer(self, board):
        b_params = board[1]
        return b_params[N_CUR_PLAYER]

    @staticmethod
    def display(board):
        print("   ---------------------")
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
        print("   ---------------------")
        print("     0 1 2 3 4 5 6 7 8")

    @staticmethod
    def display_flat(board):
        print("   ---------------------")
        for i in range(10):
            y = 9-i
            print(" ", end="")
            print(y, end=" | ")
            for j in range(9):
                x = j
                print(JanggiGame.square_content[board[x][y]], end="")
                if (x == 2 or x == 5) and (y >= 7 or y <= 2):
                    print("|", end="")
                else:
                    print(" ", end="")
            print("|")
        print("   ---------------------")
        print("     0 1 2 3 4 5 6 7 8")

    @staticmethod
    def encodeBoard(board):
        encodedBoard = []

        b = Board(0, 0)
        b.pieces = np.copy(board[0])
        b.b_params = np.copy(board[1])
        b.rep_dict = board[2].copy()

        player = b.b_params[N_CUR_PLAYER]
        player_sign = 1 if player == PLAYER_CHO else -1
        move_cnt = b.b_params[N_MOVE_CNT]

        # Set up boards
        for t in range(CONFIG_T):
            pieces_t = b.pieces[t]

            # Create an encoded board
            enc_t = [0] * CONFIG_M
            for i in range(CONFIG_M):
                enc_t[i] = [0] * CONFIG_X
                for j in range(CONFIG_X):
                    enc_t[i][j] = [0] * CONFIG_Y
            
            # Fill in the pieces
            for i in range(CONFIG_X):
                for j in range(CONFIG_Y):
                    if pieces_t[i][j] == 0:
                        continue
                    if np.sign(pieces_t[i][j]) == player_sign:
                        enc_t[abs(pieces_t[i][j]) - 1][i][j] = 1
                    else:
                        enc_t[7 + abs(pieces_t[i][j]) - 1][i][j] = 1
            
            # Fill in repetition count
            canonical_board = pieces_t
            if (player == PLAYER_HAN):
                canonical_board = np.flip(canonical_board, [0, 1])
            repcnt = b.rep_dict[canonical_board.tostring()]
            if (repcnt >= 1):
                enc_t[14] = np.array(enc_t[14]) + 1
            if (repcnt >= 2):
                enc_t[15] = np.array(enc_t[15]) + 1

            # Append to the encodedBoard
            if t == 0:
                encodedBoard = enc_t
            else:
                encodedBoard = np.concatenate((encodedBoard, enc_t), 0)

        # Set up player
        enc_player = [0] * CONFIG_X
        for i in range(CONFIG_X):
            enc_player[i] = [0] * CONFIG_Y
        enc_player += player
        encodedBoard = np.concatenate((encodedBoard, [enc_player]), 0)

        # Set up move_cnt
        enc_mv = [0] * CONFIG_X
        for i in range(CONFIG_X):
            enc_mv[i] = [0] * CONFIG_Y
        enc_mv += move_cnt
        encodedBoard = np.concatenate((encodedBoard, [enc_mv]), 0)

        return np.array(encodedBoard)