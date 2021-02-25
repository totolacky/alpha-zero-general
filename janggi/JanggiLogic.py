'''
Author: Eric P. Nichols
Date: Feb 8, 2008.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''
import numpy as np
from .JanggiConstants import *
from collections import defaultdict

class Board():
    def __init__(self, c1, c2):
        # Done
        "Set up initial board configuration."
        
        """
        board = (pieces, (han_pcs, cho_pcs, move_cnt, curr_player, han_score, cho_score), rep_dict)
        """

        # Create the empty board state.
        self.pieces = [None]*CONFIG_T
        for i in range(CONFIG_T):
            self.pieces[i] = [0]*CONFIG_X
            for j in range(CONFIG_X):
                self.pieces[i][j] = [0]*CONFIG_Y

        # Set up first player's pieces.
        self.pieces[0][4][1] = NK   # K
        self.pieces[0][0][0] = NC   # C
        self.pieces[0][8][0] = NC   # C
        self.pieces[0][1][2] = NP   # P
        self.pieces[0][7][2] = NP   # P

        self.pieces[0][1][0] += NM * int(c1==1 or c1==2)    # M
        self.pieces[0][2][0] += NM * int(c1==0 or c1==3)    # M
        self.pieces[0][6][0] += NM * int(c1==1 or c1==3)    # M
        self.pieces[0][7][0] += NM * int(c1==0 or c1==2)    # M

        self.pieces[0][1][0] += NX * int(c1==0 or c1==3)    # X
        self.pieces[0][2][0] += NX * int(c1==1 or c1==2)    # X
        self.pieces[0][6][0] += NX * int(c1==0 or c1==2)    # X
        self.pieces[0][7][0] += NX * int(c1==1 or c1==3)    # X

        self.pieces[0][3][0] = NS    # S
        self.pieces[0][5][0] = NS    # S
        self.pieces[0][0][3] = NB    # B
        self.pieces[0][2][3] = NB    # B
        self.pieces[0][4][3] = NB    # B
        self.pieces[0][6][3] = NB    # B
        self.pieces[0][8][3] = NB    # B

        # Set up opponent's pieces.
        self.pieces[0][4][8] = -NK    # K
        self.pieces[0][0][9] = -NC    # C
        self.pieces[0][8][9] = -NC    # C
        self.pieces[0][1][7] = -NP    # P
        self.pieces[0][7][7] = -NP    # P

        self.pieces[0][1][9] += -NM * int(c2==0 or c2==3)    # M
        self.pieces[0][2][9] += -NM * int(c2==1 or c2==2)    # M
        self.pieces[0][6][9] += -NM * int(c2==0 or c2==2)    # M
        self.pieces[0][7][9] += -NM * int(c2==1 or c2==3)    # M

        self.pieces[0][1][9] += -NX * int(c2==1 or c2==2)    # X
        self.pieces[0][2][9] += -NX * int(c2==0 or c2==3)    # X
        self.pieces[0][6][9] += -NX * int(c2==1 or c2==3)    # X
        self.pieces[0][7][9] += -NX * int(c2==0 or c2==2)    # X

        self.pieces[0][3][9] = -NS    # S
        self.pieces[0][5][9] = -NS    # S
        self.pieces[0][0][6] = -NB    # B
        self.pieces[0][2][6] = -NB    # B
        self.pieces[0][4][6] = -NB    # B
        self.pieces[0][6][6] = -NB    # B
        self.pieces[0][8][6] = -NB    # B

        # Convert to numpy array
        self.pieces = np.array(self.pieces)

        # Set up params: han_pcs/cho_pcs (bitmap indicating the live pieces), move_cnt and curr_player
        han_pcs = 34133    # 10000/10/10/10/10/10/1
        cho_pcs = 34133    # 10000/10/10/10/10/10/1
        move_cnt = 0
        cur_player = 0
        han_score = 73.5
        cho_score = 72
        captured = False
        is_bic = False
        self.b_params = np.array([han_pcs, cho_pcs, move_cnt, cur_player, han_score, cho_score, captured, is_bic])

        # Create the empty repetition set
        self.rep_dict = {}
        self.rep_dict = defaultdict(lambda:0, self.rep_dict)

    def get_legal_moves(self):
        """Returns all the legal moves for the current board."""
        moves = []  # stores the legal moves.

        legal_sign = 1 if self.b_params[N_CUR_PLAYER] == PLAYER_CHO else -1
        print(self.b_params[N_CUR_PLAYER])

        for y in range(CONFIG_Y):
            for x in range(CONFIG_X):
                if (self.pieces[0][x][y] == 0):
                    continue
                elif abs(self.pieces[0][x][y]) == NK and np.sign(self.pieces[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_K(x,y)
                elif abs(self.pieces[0][x][y]) == NC and np.sign(self.pieces[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_C(x,y)
                elif abs(self.pieces[0][x][y]) == NP and np.sign(self.pieces[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_P(x,y)
                elif abs(self.pieces[0][x][y]) == NM and np.sign(self.pieces[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_M(x,y)
                elif abs(self.pieces[0][x][y]) == NX and np.sign(self.pieces[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_X(x,y)
                elif abs(self.pieces[0][x][y]) == NS and np.sign(self.pieces[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_S(x,y)
                elif abs(self.pieces[0][x][y]) == NB and np.sign(self.pieces[0][x][y]) == legal_sign:
                    moves = moves + self.get_moves_for_B(x,y)
                elif np.sign(self.pieces[0][x][y]) == -legal_sign:
                    continue
                else:
                    assert False

        # Add turn skip
        moves = moves + [(58, 0, 0)]

        return moves

    def get_moves_for_K(self, x, y):
        """Returns all the legal moves of K that use the given square as a base."""
        # Assert that the given piece is a K, and it is in a valid place
        print((x,y))
        assert abs(self.pieces[0][x][y]) == NK
        assert x >= 3 and x <= 5 and y >= 0 and y <= 2

        my_sign = np.sign(self.pieces[0][x][y])

        # Ordinary moves are same as S
        moves = self.get_moves_for_S(x, y)
        
        # Draw move
        for i in range(9):
            if (abs(self.pieces[0][x][y+i+1]) == NK):
                moves.append((16+i, x, y))
            elif (abs(self.pieces[0][x][y+i+1]) != 0):
                break

        # return the generated move list
        return moves
    
    def get_moves_for_C(self, x, y):
        """Returns all the legal moves of C that use the given square as a base."""
        # Assert that the given piece is a C
        assert abs(self.pieces[0][x][y]) == NC

        my_sign = np.sign(self.pieces[0][x][y])

        moves = []
        for i in range(8): # 0 ~ 7: (1, 0) ~ (8, 0)
            if (x+i+1 >= CONFIG_X):
                break
            if (self.pieces[0][x+i+1][y] == 0): # Empty space
                moves.append((i, x, y))
                continue
            elif (np.sign(self.pieces[0][x+i+1][y]) != my_sign):    # Capture opponent piece
                moves.append((i, x, y))
            break

        for i in range(8): # 8 ~ 15: (-1, 0) ~ (-8, 0)
            if (x-i-1 < 0):
                break
            if (self.pieces[0][x-i-1][y] == 0): # Empty space
                moves.append((8+i, x, y))
                continue
            elif (np.sign(self.pieces[0][x-i-1][y]) != my_sign):    # Capture opponent piece
                moves.append((8+i, x, y))
            break

        for i in range(9): # 16 ~ 24: (0, 1) ~ (0, 9)
            if (y+i+1 >= CONFIG_Y):
                break
            if (self.pieces[0][x][y+i+1] == 0): # Empty space
                moves.append((16+i, x, y))
                continue
            elif (np.sign(self.pieces[0][x][y+i+1]) != my_sign):    # Capture opponent piece
                moves.append((16+i, x, y))
            break

        for i in range(9): # 25 ~ 33: (0, -1) ~ (0, -9)
            if (y-i-1 < 0):
                break
            if (self.pieces[0][x][y-i-1] == 0): # Empty space
                moves.append((25+i, x, y))
                continue
            elif (np.sign(self.pieces[0][x][y-i-1]) != my_sign):    # Capture opponent piece
                moves.append((25+i, x, y))
            break

        if ((x == 3 and (y == 0 or y == 7)) or (x == 4 and (y == 1 or y == 8))): # 34: (1, 1)
            if self.pieces[0][x+1][y+1] == 0 or np.sign(self.pieces[0][x+1][y+1]) != my_sign:
                moves.append((34, x, y))

        if (x == 3 and (y == 0 or y == 7)): # 35: (2, 2)
            if self.pieces[0][x+1][y+1] == 0 and (self.pieces[0][x+2][y+2] == 0 or np.sign(self.pieces[0][x+2][y+2]) != my_sign):
                moves.append((35, x, y))

        if ((x == 5 and (y == 0 or y == 7)) or (x == 4 and (y == 1 or y == 8))): # 36: (-1, 1)
            if self.pieces[0][x-1][y+1] == 0 or np.sign(self.pieces[0][x-1][y+1]) != my_sign:
                moves.append((36, x, y))

        if (x == 5 and (y == 0 or y == 7)): # 37: (-2, 2)
            if self.pieces[0][x-1][y+1] == 0 and (self.pieces[0][x-2][y+2] == 0 or np.sign(self.pieces[0][x-2][y+2]) != my_sign):
                moves.append((37, x, y))

        if ((x == 4 and (y == 1 or y == 8)) or (x == 5 and (y == 2 or y == 9))): # 38: (-1, -1)
            if self.pieces[0][x-1][y-1] == 0 or np.sign(self.pieces[0][x-1][y-1]) != my_sign:
                moves.append((38, x, y))

        if (x == 4 and (y == 1 or y == 8)): # 39: (-2, -2)
            if self.pieces[0][x-1][y-1] == 0 and (self.pieces[0][x-2][y-2] == 0 or np.sign(self.pieces[0][x-2][y-2]) != my_sign):
                moves.append((39, x, y))

        if ((x == 3 and (y == 2 or y == 9)) or (x == 4 and (y == 1 or y == 8))): # 40: (1, -1)
            if self.pieces[0][x+1][y-1] == 0 or np.sign(self.pieces[0][x+1][y-1]) != my_sign:
                moves.append((40, x, y))

        if (x == 3 and (y == 2 or y == 9)): # 41: (2, -2)
            if self.pieces[0][x+1][y-1] == 0 and (self.pieces[0][x+2][y-2] == 0 or np.sign(self.pieces[0][x+2][y-2]) != my_sign):
                moves.append((41, x, y))

        # return the generated move list
        return moves

    def get_moves_for_P(self, x, y):
        """Returns all the legal moves of P that use the given square as a base."""
        # Assert that the given piece is a P
        assert abs(self.pieces[0][x][y]) == NP

        my_sign = np.sign(self.pieces[0][x][y])

        moves = []
        done = [False, False, False, False]
        jump = [False, False, False, False]
        for i in range(9):
            for j in range(4):
                if (done[j]):
                    continue

                if (j == 0):    # 0 ~ 7: (1, 0) ~ (8, 0)
                    newx = x+i+1
                    newy = y
                    a = i
                elif (j == 1):  # 8 ~ 15: (-1, 0) ~ (-8, 0)
                    newx = x-i-1
                    newy = y
                    a = 8 + i
                elif (j == 2):  # 16 ~ 24: (0, 1) ~ (0, 9)
                    newx = x
                    newy = y+i+1
                    a = 16 + i
                else:           # 25 ~ 33: (0, -1) ~ (0, -9)
                    newx = x
                    newy = y-i-1
                    a = 25 + i
                
                # Invalid destination
                if (newx >= CONFIG_X or newx < 0 or newy >= CONFIG_Y or newy < 0):
                    done[j] = True
                    continue
                
                # Empty destination
                if (self.pieces[0][newx][newy] == 0): 
                    if (jump[j]):
                        moves.append((a, x, y))
                    continue
                # Nonempty destination
                else:
                    if (not jump[j]):  # The first piece that appear in such direction
                        if (abs(self.pieces[0][newx][newy]) == NP):   # P cannot jump over another P
                            done[j] = True
                            continue
                        else:
                            jump[j] = True
                            continue
                    else:   # The second piece that appears in such direction
                        if (abs(self.pieces[0][newx][newy]) != NP and np.sign(self.pieces[0][newx][newy] != my_sign)):
                            # Capture opponent piece
                            # P cannot capture another P
                            moves.append((a, x, y))
                        done[j] = True
                        continue
        
        if (x == 3 and (y == 0 or y == 7)): # 35: (2, 2)
            if (self.pieces[0][x+1][y+1] != 0 \
                and abs(self.pieces[0][x+1][y+1]) != NP \
                and (self.pieces[0][x+2][y+2] == 0 or np.sign(self.pieces[0][x+2][y+2]) != my_sign)\
                and abs(self.pieces[0][x+2][y+2]) != NP):
                moves.append((35, x, y))

        if (x == 5 and (y == 0 or y == 7)): # 37: (-2, 2)
            if (self.pieces[0][x-1][y+1] != 0 \
                and abs(self.pieces[0][x-1][y+1]) != NP \
                and (self.pieces[0][x-2][y+2] == 0 or np.sign(self.pieces[0][x-2][y+2]) != my_sign)\
                and abs(self.pieces[0][x-2][y+2]) != NP):
                moves.append((37, x, y))

        if (x == 4 and (y == 1 or y == 8)): # 39: (-2, -2)
            if (self.pieces[0][x-1][y-1] != 0 \
                and abs(self.pieces[0][x-1][y-1]) != NP \
                and (self.pieces[0][x-2][y-2] == 0 or np.sign(self.pieces[0][x-2][y-2]) != my_sign)\
                and abs(self.pieces[0][x-2][y-2]) != NP):
                moves.append((39, x, y))

        if (x == 3 and (y == 2 or y == 9)): # 41: (2, -2)
            if (self.pieces[0][x+1][y-1] != 0 \
                and abs(self.pieces[0][x+1][y-1]) != NP \
                and (self.pieces[0][x+2][y-2] == 0 or np.sign(self.pieces[0][x+2][y-2]) != my_sign)\
                and abs(self.pieces[0][x+2][y-2]) != NP):
                moves.append((41, x, y))

        return moves

    def get_moves_for_M(self, x, y):
        """Returns all the legal moves of M that use the given square as a base."""
        # Assert that the given piece is a M
        assert abs(self.pieces[0][x][y]) == NM

        my_sign = np.sign(self.pieces[0][x][y])

        moves = []
        if self._can_M_move(x, y, 2, 1):    # 42: (2, 1)
            moves.append((42, x, y))

        if self._can_M_move(x, y, 2, -1):   # 43: (2, -1)
            moves.append((43, x, y))
        
        if self._can_M_move(x, y, -2, 1):   # 44: (-2, 1)
            moves.append((44, x, y))
        
        if self._can_M_move(x, y, -2, -1):  # 45: (-2, -1)
            moves.append((45, x, y))
        
        if self._can_M_move(x, y, 1, 2):    # 46: (1, 2)
            moves.append((46, x, y))
        
        if self._can_M_move(x, y, 1, -2):   # 47: (1, -2)
            moves.append((47, x, y))
        
        if self._can_M_move(x, y, -1, 2):   # 48: (-1, 2)
            moves.append((48, x, y))
        
        if self._can_M_move(x, y, -1, -2):  # 49: (-1, -2)
            moves.append((49, x, y))

        return moves

    def _can_M_move(self, x, y, dx, dy):
        midx = int(x + dx/2) if (abs(dx) == 2) else x
        midy = y if (abs(dx) == 2) else int(y + dy/2)
        finx = x + dx
        finy = y + dy

        # Cannot move if the final position is invalid
        if finx < 0 or finx >= CONFIG_X or finy < 0 or finy >= CONFIG_Y:
            return False
        
        # Cannot move if a piece is in the way
        if self.pieces[0][midx][midy] != 0:
            return False

        # Cannot move if there is my piece in the final position
        if self.pieces[0][finx][finy] != 0 and np.sign(self.pieces[0][finx][finy]) == np.sign(self.pieces[0][x][y]):
            return False
        
        # Else, return true
        return True

    def get_moves_for_X(self, x, y):
        """Returns all the legal moves of X that use the given square as a base."""
        # Assert that the given piece is a X
        assert abs(self.pieces[0][x][y]) == NX

        moves = []
        if self._can_X_move(x, y, 3, 2):    # 50: (3, 2)
            moves.append((50, x, y))

        if self._can_X_move(x, y, 3, -2):   # 51: (3, -2)
            moves.append((51, x, y))
        
        if self._can_X_move(x, y, -3, 2):   # 52: (-3, 2)
            moves.append((52, x, y))
        
        if self._can_X_move(x, y, -3, -2):  # 53: (-3, -2)
            moves.append((53, x, y))
        
        if self._can_X_move(x, y, 2, 3):    # 54: (2, 3)
            moves.append((54, x, y))
        
        if self._can_X_move(x, y, 2, -3):   # 55: (2, -3)
            moves.append((55, x, y))
        
        if self._can_X_move(x, y, -2, 3):   # 56: (-2, 3)
            moves.append((56, x, y))
        
        if self._can_X_move(x, y, -2, -3):  # 57: (-2, -3)
            moves.append((57, x, y))

        return moves

    def _can_X_move(self, x, y, dx, dy):
        midx1 = int(x + dx/3) if (abs(dx) == 3) else x
        midy1 = y if (abs(dx) == 3) else int(y + dy/3)
        midx2 = int(x + dx/3*2) if (abs(dx) == 3) else int(x + dx/2)
        midy2 = int(y + dy/2) if (abs(dx) == 3) else int(y + dy/3*2)
        finx = x + dx
        finy = y + dy

        # Cannot move if the final position is invalid
        if finx < 0 or finx >= CONFIG_X or finy < 0 or finy >= CONFIG_Y:
            return False
        
        # Cannot move if a piece is in the way
        if self.pieces[0][midx1][midy1] != 0 or self.pieces[0][midx2][midy2] != 0:
            return False

        # Cannot move if there is my piece in the final position
        if self.pieces[0][finx][finy] != 0 and np.sign(self.pieces[0][finx][finy]) == np.sign(self.pieces[0][x][y]):
            return False
        
        # Else, return true
        return True

    def get_moves_for_S(self, x, y):
        """Returns all the legal moves of S that use the given square as a base."""
        # Assert that the given piece is a S, and it is in a valid place
        assert abs(self.pieces[0][x][y]) == NS or abs(self.pieces[0][x][y]) == NK
        assert x >= 3 and x <= 5 and y >= 0 and y <= 2

        my_sign = np.sign(self.pieces[0][x][y])

        moves = []
        if (x < 5): # 0: (1, 0)
            if self.pieces[0][x+1][y] == 0 or np.sign(self.pieces[0][x+1][y]) != my_sign:
                moves.append((0, x, y))
        if (x > 3): # 8: (-1, 0)
            if self.pieces[0][x-1][y] == 0 or np.sign(self.pieces[0][x-1][y]) != my_sign:
                moves.append((8, x, y))
        if (y < 2): # 16: (0, 1)
            if self.pieces[0][x][y+1] == 0 or np.sign(self.pieces[0][x][y+1]) != my_sign:
                moves.append((16, x, y))
        if (y > 0): # 25: (0, -1)
            if self.pieces[0][x][y-1] == 0 or np.sign(self.pieces[0][x][y-1]) != my_sign:
                moves.append((25, x, y))
        if ((x == 3 and y == 0) or (x == 4 and y == 1)): # 34: (1, 1)
            if self.pieces[0][x+1][y+1] == 0 or np.sign(self.pieces[0][x+1][y+1]) != my_sign:
                moves.append((34, x, y))
        if ((x == 5 and y == 0) or (x == 4 and y == 1)): # 36: (-1, 1)
            if self.pieces[0][x-1][y+1] == 0 or np.sign(self.pieces[0][x-1][y+1]) != my_sign:
                moves.append((36, x, y))
        if ((x == 4 and y == 1) or (x == 5 and y == 2)): # 38: (-1, -1)
            if self.pieces[0][x-1][y-1] == 0 or np.sign(self.pieces[0][x-1][y-1]) != my_sign:
                moves.append((38, x, y))
        if ((x == 3 and y == 2) or (x == 4 and y == 1)): # 40: (1, -1)
            if self.pieces[0][x+1][y-1] == 0 or np.sign(self.pieces[0][x+1][y-1]) != my_sign:
                moves.append((40, x, y))

        # return the generated move list
        return moves

    def get_moves_for_B(self, x, y):
        """Returns all the legal moves of B that use the given square as a base."""
        # Assert that the given piece is a B, and it is in a valid place
        assert abs(self.pieces[0][x][y]) == NB

        my_sign = np.sign(self.pieces[0][x][y])

        moves = []
        if (x < CONFIG_X - 1): # 0: (1, 0)
            if self.pieces[0][x+1][y] == 0 or np.sign(self.pieces[0][x+1][y]) != my_sign:
                moves.append((0, x, y))
        if (x > 0): # 8: (-1, 0)
            if self.pieces[0][x-1][y] == 0 or np.sign(self.pieces[0][x-1][y]) != my_sign:
                moves.append((8, x, y))
        if (y < CONFIG_Y - 1): # 16: (0, 1)
            if self.pieces[0][x][y+1] == 0 or np.sign(self.pieces[0][x][y+1]) != my_sign:
                moves.append((16, x, y))
        if ((x == 3 and y == 7) or (x == 4 and y == 8)): # 34: (1, 1)
            if self.pieces[0][x+1][y+1] == 0 or np.sign(self.pieces[0][x+1][y+1]) != my_sign:
                moves.append((34, x, y))
        if ((x == 5 and y == 7) or (x == 4 and y == 8)): # 36: (-1, 1)
            if self.pieces[0][x-1][y+1] == 0 or np.sign(self.pieces[0][x-1][y+1]) != my_sign:
                moves.append((36, x, y))

        # return the generated move list
        return moves

    def execute_move(self, move):
        """Perform the given move on the board; catch pieces as necessary.
        color gives the color pf the piece to play
        """
        player = self.b_params[N_CUR_PLAYER]   # 0: Cho, 1: Han
        (a, x, y) = move    # a: action type, (x,y): board position where the action starts

        assert 0 <= a and a <= 58

        ## Duplicate the last board configuration and shift self.pieces
        self.pieces = np.delete(self.pieces, CONFIG_T - 1, 0)
        self.pieces = np.concatenate(([self.pieces[0].copy()], self.pieces), 0)

        assert self.pieces.shape == (CONFIG_T, CONFIG_X, CONFIG_Y)

        ## Update current player & move count
        self.b_params[N_CUR_PLAYER] = PLAYER_HAN if self.b_params[N_CUR_PLAYER] == PLAYER_CHO else PLAYER_CHO  # change current player
        self.b_params[N_MOVE_CNT] += 1  # increment move count

        ## Rotate board, set captured to false and return if the action is turn skip
        if (a == 58):
            self.pieces = np.flip(self.pieces, [1,2])
            self.b_params[N_CAPTURED] = False
            return

        ## Otherwise, add the current board to the rep_dict
        canonical_board = self.pieces[0]
        if (player == PLAYER_HAN):
            canonical_board = np.flip(canonical_board, [0, 1])
        self.rep_dict[canonical_board.tostring()] += 1

        ## Move the pieces. First, assert that the moving piece is present.
        assert (self.pieces[0][x][y] != 0)
        
        ## Move the piece according to the given action
        if (a <= 7):
            newx = x + (a + 1)
            newy = y
        elif (a <= 15):
            newx = x - (a - 7)
            newy = y
        elif (a <= 24):
            newx = x
            newy = y + (a - 15)
        elif (a <= 33):
            newx = x
            newy = y - (a - 24)
        elif (a <= 35):
            newx = x + (a - 33)
            newy = y + (a - 33)
        elif (a <= 37):
            newx = x - (a - 35)
            newy = y + (a - 35)
        elif (a <= 39):
            newx = x - (a - 37)
            newy = y - (a - 37)
        elif (a <= 41):
            newx = x + (a - 39)
            newy = y - (a - 39)
        elif (a <= 43):
            newx = x + 2
            newy = y + 1 if (a == 42) else y - 1
        elif (a <= 45):
            newx = x - 2
            newy = y + 1 if (a == 44) else y - 1
        elif (a <= 47):
            newx = x + 1
            newy = y + 2 if (a == 46) else y - 2
        elif (a <= 49):
            newx = x - 1
            newy = y + 2 if (a == 48) else y - 2
        elif (a <= 51):
            newx = x + 3
            newy = y + 2 if (a == 50) else y - 2
        elif (a <= 53):
            newx = x - 3
            newy = y + 2 if (a == 52) else y - 2
        elif (a <= 55):
            newx = x + 2
            newy = y + 3 if (a == 54) else y - 3
        elif (a <= 57):
            newx = x - 2
            newy = y + 3 if (a == 56) else y - 3
        else:
            # This should not happen. Panic.
            assert False

        assert x >= 0 and x < CONFIG_X and y >= 0 and y < CONFIG_Y
        assert newx >= 0 and newx < CONFIG_X and newy >= 0 and newy < CONFIG_Y
        
        moving_piece = self.pieces[0][x][y]
        captured_piece = self.pieces[0][newx][newy]
        self.pieces[0][x][y] = 0
        self.pieces[0][newx][newy] = moving_piece

        ## Update han_pcs and cho_pcs
        if (captured_piece != NULL):
            if (player == PLAYER_HAN):  # Han captured Cho
                self.b_params[N_CHO_PCS] = self.remove_piece(self.b_params[N_CHO_PCS], abs(captured_piece))
            else:                       # Cho captured Han
                self.b_params[N_HAN_PCS] = self.remove_piece(self.b_params[N_HAN_PCS], abs(captured_piece))
        
        ## Update han_score and cho_score
        if (captured_piece != NULL):
            if (player == PLAYER_HAN):  # Han captured Cho
                self.b_params[N_CHO_SCORE] -= self._piece_score(abs(captured_piece))
            else:                       # Cho captured Han
                self.b_params[N_HAN_SCORE] -= self._piece_score(abs(captured_piece))

        ## Update captured
        self.b_params[N_CAPTURED] = captured_piece != NULL

        ## Update is_bic
        if (abs(moving_piece) == NK and abs(captured_piece) == NK):
            self.b_params[N_IS_BIC] = True
        
        ## Flip board and return
        self.pieces = np.flip(self.pieces, [1,2])
        return

    def remove_piece(self, pcs, cap_piece):
        """ Given han_pcs or cho_pcs, remove one of the captured piece"""
        assert (1 <= cap_piece and cap_piece <= 7)

        if (cap_piece == NK):    # G
            return pcs & ~(1<<0)
        elif (cap_piece == NC):  # C
            if (pcs & (1<<1) != 0):
                return pcs & ~(1<<1)
            elif (pcs & (1<<2) != 0):
                return (pcs & ~(1<<2)) | (1<<1)
            else:
                assert False
        elif (cap_piece == NP):  # P
            if (pcs & (1<<3) != 0):
                return pcs & ~(1<<3)
            elif (pcs & (1<<4) != 0):
                return (pcs & ~(1<<4)) | (1<<3)
            else:
                assert False
        elif (cap_piece == NM):  # M
            if (pcs & (1<<5) != 0):
                return pcs & ~(1<<5)
            elif (pcs & (1<<6) != 0):
                return (pcs & ~(1<<6)) | (1<<5)
            else:
                assert False
        elif (cap_piece == NX):  # X
            if (pcs & (1<<7) != 0):
                return pcs & ~(1<<7)
            elif (pcs & (1<<8) != 0):
                return (pcs & ~(1<<8)) | (1<<7)
            else:
                assert False
        elif (cap_piece == NS):  # S
            if (pcs & (1<<9) != 0):
                return pcs & ~(1<<9)
            elif (pcs & (1<<10) != 0):
                return (pcs & ~(1<<10)) | (1<<9)
            else:
                assert False
        elif (cap_piece == NB):  # B
            if (pcs & (1<<11) != 0):
                return pcs & ~(1<<11)
            elif (pcs & (1<<12) != 0):
                return (pcs & ~(1<<12)) | (1<<11)
            elif (pcs & (1<<13) != 0):
                return (pcs & ~(1<<13)) | (1<<12)
            elif (pcs & (1<<14) != 0):
                return (pcs & ~(1<<14)) | (1<<13)
            elif (pcs & (1<<15) != 0):
                return (pcs & ~(1<<15)) | (1<<14)
            else:
                assert False

    def _get_piece_num(self, pcs, query_piece):
        """ Given han_pcs or cho_pcs, get the number of query_pieces"""
        assert (1 <= query_piece and query_piece <= 7)

        pcs = int(pcs)
        num = 0

        if (query_piece == NK):    # G
            num = pcs & (1<<0)
        elif (query_piece == NC):  # C
            num = (pcs & (3<<1))>>1
        elif (query_piece == NP):  # P
            num = (pcs & (3<<3))>>3
        elif (query_piece == NM):  # M
            num = (pcs & (3<<5))>>5
        elif (query_piece == NX):  # X
            num = (pcs & (3<<7))>>7
        elif (query_piece == NS):  # S
            num = (pcs & (3<<9))>>9
        elif (query_piece == NB):  # B
            num = (pcs & (31<<11))>>11
        else:
            assert False
        
        if (num == 0):
            return 0
        else:
            return int(np.log2(num) + 1)
    
    def game_ended(self):
        """ Return Cho score if the game is over.
            Return 0 otherwise. """
        # Compare these with han(cho)_pcs & ATTACK_MASK
        cannot_win_yangsa = [
                    # ㅁㅁㅁ: BBBBB/SS/XX/MM/PP/CC/K
            # 대삼능
            264,    # 포양상: 00000/00/10/00/01/00/0
            192,    # 양마상: 00000/00/01/10/00/00/0
            288,    # 마양상: 00000/00/10/01/00/00/0

            # 소삼능
            2088,   # 포마졸: 00001/00/00/01/01/00/0
            2184,   # 포상졸: 00001/00/01/00/01/00/0
            2112,   # 양마졸: 00001/00/00/10/00/00/0
            2304,   # 양상졸: 00001/00/10/00/00/00/0
            4104,   # 포양졸: 00010/00/00/00/01/00/0
            4128,   # 마양졸: 00010/00/00/01/00/00/0
            4224,   # 상마졸: 00010/00/01/00/00/00/0
            8192,   # ㅁ삼졸: 00100/00/00/00/00/00/0

            # 차삼능
            4106,   # 차이졸: 00010/00/00/00/01/01/0

            # 차이능/
            10,     # ㅁ차포: 00000/00/00/00/01/01/0
            34,     # ㅁ차마: 00000/00/00/01/00/01/0
            130,    # ㅁ차상: 00000/00/01/00/00/01/0
            2050,   # ㅁ차졸: 00001/00/00/00/00/01/0
        ]

        cannot_win_wesa = [
            2050,   # ㅁ차졸: 00001/00/00/00/00/01/0
        ]

        # The player that just made a move
        last_player = PLAYER_HAN if self.b_params[N_CUR_PLAYER] == PLAYER_CHO else PLAYER_CHO;

        # If the game just ended with a bic, end the game
        if self.b_params[N_IS_BIC]:
            return 1 if self.b_params[N_CHO_SCORE] > self.b_params[N_HAN_SCORE] else -1

        # Game is over if a K is captured
        if self._get_piece_num(self.b_params[N_HAN_PCS], NK) == 0:
            return 1
        if self._get_piece_num(self.b_params[N_CHO_PCS], NK) == 0:
            return -1

        # Game is over if no player can win
        if ((self._get_piece_num(self.b_params[N_HAN_PCS], NS) == 2 and (int(self.b_params[N_CHO_PCS]) & int(ATTACK_MASK)) in cannot_win_yangsa) \
                or (self._get_piece_num(self.b_params[N_HAN_PCS], NS) == 1 and (int(self.b_params[N_CHO_PCS]) & int(ATTACK_MASK)) in cannot_win_wesa)) \
            and ((self._get_piece_num(self.b_params[N_CHO_PCS], NS) == 2 and (int(self.b_params[N_HAN_PCS]) & int(ATTACK_MASK)) in cannot_win_yangsa) \
                or (self._get_piece_num(self.b_params[N_CHO_PCS], NS) == 1 and (int(self.b_params[N_HAN_PCS]) & int(ATTACK_MASK)) in cannot_win_wesa)):
            return 1 if self.b_params[N_CHO_SCORE] > self.b_params[N_HAN_SCORE] else -1

        # Game is over if repetition happens 3 times
        canonical_board = self.pieces[0]
        if (self.b_params[N_CUR_PLAYER] == PLAYER_HAN):
            canonical_board = np.flip(canonical_board, [0, 1])
        canon_string = canonical_board.tostring()

        if (self.rep_dict[canon_string] >= 2):
            # If both scores are under 30, check score
            if (self.b_params[N_CHO_SCORE] < 30 and self.b_params[N_HAN_SCORE] < 30):
                return 1 if self.b_params[N_CHO_SCORE] > self.b_params[N_HAN_SCORE] else -1
            # Otherwise, the last player lose
            else:
                return 1 if last_player == PLAYER_HAN else -1
        
        # For simplicity, the game is over when move_cnt hits 250.
        if self.b_params[N_MOVE_CNT] >= 250:
            return 1 if self.b_params[N_CHO_SCORE] > self.b_params[N_HAN_SCORE] else -1
        
        # Game is over if bic is called when a player has score >= 30.
        # The last player lose.
        if (self.b_params[N_HAN_SCORE] >= 30 or self.b_params[N_CHO_SCORE] >= 30):
            if (self._get_bic_called()):
                return 1 if last_player == PLAYER_HAN else -1
        
        # If the game is not over, return 0.
        return 0
    
    def _get_bic_called(self):
        """ Return true if two K's are facing each other """
        for i in range(3):
            for j in range(3):
                x = i+3
                y = j

                if (abs(self.pieces[0][x][y]) == NK):
                    for k in range(9):
                        newx = x
                        newy = y+k+1
                        
                        if (newy >= CONFIG_Y):
                            return False
                        
                        if (abs(self.pieces[0][newx][newy]) == NK):
                            # Two K's are facing each other
                            return True
                        elif (abs(self.pieces[0][newx][newy]) != 0):
                            # Two K's are not directly facing each other
                            return False
    
    def _piece_score(self, piece):
        if piece == NK:
            return 0
        elif piece == NC:
            return 13
        elif piece == NP:
            return 7
        elif piece == NM:
            return 5
        elif piece == NX:
            return 3
        elif piece == NS:
            return 3
        elif piece == NB:
            return 2
        else:
            assert False
    
    @staticmethod
    def _action_to_dxdy(a):
        if (a <= 7):
            return (a+1, 0)
        elif (a <= 15):
            return (-a+7, 0)
        elif (a <= 24):
            return (0, a-15)
        elif (a <= 33):
            return (0, -a+24)
        elif (a <= 35):
            return (a-33, a-33)
        elif (a <= 37):
            return (-a+35, a-35)
        elif (a <= 39):
            return (-a+37, -a+37)
        elif (a <= 41):
            return (a-39, -a+39)
        elif (a <= 43):
            return (2, 1 if a == 42 else -1)
        elif (a <= 45):
            return (-2, 1 if (a == 44) else -11)
        elif (a <= 47):
            return (1, 2 if (a == 46) else -2)
        elif (a <= 49):
            return (-1, 2 if (a == 48) else -2)
        elif (a <= 51):
            return (3, 2 if (a == 50) else -2)
        elif (a <= 53):
            return (-2, 2 if (a == 52) else -2)
        elif (a <= 55):
            return (2, 3 if (a == 54) else -3)
        elif (a <= 57):
            return (-2, 3 if (a == 56) else -3)
        elif (a == 58):
            return (0, 0)
        else:
            # This should not happen. Panic.
            assert False
    
    @staticmethod
    def _dxdy_to_action(dx, dy):
        for a in range(CONFIG_X*CONFIG_Y*CONFIG_A + 1):
            if (dx, dy) == Board._action_to_dxdy(a):
                return a
        return -1