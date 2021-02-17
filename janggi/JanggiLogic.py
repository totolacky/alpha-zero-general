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

class Board():
    def __init__(self, c1, c2):
        # Done
        "Set up initial board configuration."
        
        """
        board = (pieces, rep_set)
            pieces = [arr[nm, nx, ny] for timestep t CAT ... CAT arr[nm, nx, ny] for timestep t-(T-1) CAT ones(nm, nx, ny)*current player CAT ones(nm, nx, ny)*move count]
            rep_set: the set of all previous board states, that can possibly be repeated (wiped out when B moves forward, or a piece is captured)
            canonical_rep_set: rep_set, but with canonical board representations

        canonical board = board
            Do not use canonical boards. This is because 楚 gets an additional 1.5 points, making it assymetric with 漢.

        The network input form of the board should be...
            config[0] ~ config[T-1], ones(nm, nx, ny)*pieces[T], ones(nm, nx, ny)*pieces[T+1] concatenated
            with possible rotations to keep the current player's K at the bottom
        """

        self.nx = 9     # board width
        self.ny = 10    # board height
        self.nm = 14    # number of state planes per board (7 for 楚, 7 for 漢)
        self.nt = 4     # number of timesteps recorded
        self.nl = 2     # number of aux. info (current player & move count)
        self.ns = self.nm * self.nt + self.nl   # number of state planes
        self.na = 58    # number of action planes (excluding turn skip)

        # Create the empty board state.
        self.pieces = [None]*self.ns
        for i in range(self.ns):
            self.pieces[i] = [0]*self.nx
            for j in range(self.nx):
                self.pieces[i][j] = [0]*self.ny

        # Create the empty repetition sets.
        self.rep_set = {}
        
        # move count are already set to 0

        # Set up first player's pieces.
        self.pieces[0][4][1] = 1    # K
        self.pieces[1][0][0] = 1    # C
        self.pieces[1][8][0] = 1    # C
        self.pieces[2][1][2] = 1    # P
        self.pieces[2][7][2] = 1    # P

        self.pieces[3][1][0] = int(c1==1 or c1==2)    # M
        self.pieces[3][2][0] = int(c1==0 or c1==3)    # M
        self.pieces[3][6][0] = int(c1==1 or c1==3)    # M
        self.pieces[3][7][0] = int(c1==0 or c1==2)    # M

        self.pieces[4][1][0] = int(c1==0 or c1==3)    # X
        self.pieces[4][2][0] = int(c1==1 or c1==2)    # X
        self.pieces[4][6][0] = int(c1==0 or c1==2)    # X
        self.pieces[4][7][0] = int(c1==1 or c1==3)    # X

        self.pieces[5][3][0] = 1    # S
        self.pieces[5][5][0] = 1    # S
        self.pieces[6][0][3] = 1    # B
        self.pieces[6][2][3] = 1    # B
        self.pieces[6][4][3] = 1    # B
        self.pieces[6][6][3] = 1    # B
        self.pieces[6][8][3] = 1    # B

        # Set up opponent's pieces.
        self.pieces[7][4][8] = 1    # K
        self.pieces[8][0][9] = 1    # C
        self.pieces[8][8][9] = 1    # C
        self.pieces[9][1][7] = 1    # P
        self.pieces[9][7][7] = 1    # P

        self.pieces[10][1][9] = int(c1==0 or c1==3)    # M
        self.pieces[10][2][9] = int(c1==1 or c1==2)    # M
        self.pieces[10][6][9] = int(c1==0 or c1==2)    # M
        self.pieces[10][7][9] = int(c1==1 or c1==3)    # M

        self.pieces[11][1][9] = int(c1==1 or c1==2)    # X
        self.pieces[11][2][9] = int(c1==0 or c1==3)    # X
        self.pieces[11][6][9] = int(c1==1 or c1==3)    # X
        self.pieces[11][7][9] = int(c1==0 or c1==2)    # X

        self.pieces[12][3][9] = 1    # S
        self.pieces[12][5][9] = 1    # S
        self.pieces[13][0][6] = 1    # B
        self.pieces[13][2][6] = 1    # B
        self.pieces[13][4][6] = 1    # B
        self.pieces[13][6][6] = 1    # B
        self.pieces[13][8][6] = 1    # B

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def countDiff(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    count += 1
                if self[x][y]==-color:
                    count -= 1
        return count

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, color):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    if len(newmoves)>0:
                        return True
        return False

    def get_moves_for_square(self, square):
        """Returns all the legal moves that use the given square as a base.
        That is, if the given square is (3,4) and it contains a black piece,
        and (3,5) and (3,6) contain white pieces, and (3,7) is empty, one
        of the returned moves is (3,7) because everything from there to (3,4)
        is flipped.
        """
        (x,y) = square

        # determine the color of the piece.
        color = self[x][y]

        # skip empty source squares.
        if color==0:
            return None

        # search all possible directions.
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves

    def execute_move(self, move, player):
        """Perform the given move on the board; catch pieces as necessary.
        color gives the color pf the piece to play (1=楚,-1=漢)
        """
        assert player == self.pieces[self.nm*self.nt][0][0] # assert that the correct player is given as input

        (a, x, y) = move    # a: action type, (x,y): board position where the action starts

        assert 0 <= a and a <= 58

        ## Duplicate the last board configuration and shift self.pieces
        newboard = self.pieces[:self.nm].copy()
        self.pieces = np.delete(self.pieces, self.nm*(self.nt-1):self.nm*self.nt, 0)
        self.pieces = np.concatenate((newboard, self.pieces), 0)

        assert self.pieces.shape == (self.ns, self.nx, self.ny)

        ## Update current player & move count
        self.pieces[self.nm*self.nt] *= -1      # change current player
        self.pieces[self.nm*self.nt+1] += 1     # increment move count

        ## Return if the action is turn skip
        if (a == 58):
            return

        ## Otherwise, move the pieces if the action is not a turn skip
        ## First, find the moving piece

        found = 0           # For debugging
        piecetype = 0       # state plane idx of the piece
        opp_piecetype = 0   # corresponding state plane idx of the opponent

        for i in range(7):
            # search for state planes 7~13 if player == -1
            if (player == -1):
                i = i+7

            if (self.pieces[i][x][y] == 1):
                found = found + 1   # For debugging
                piecetype = i
                opp_piecetype = if (player == 1) i+7 else i-7
                self.pieces[i][x][y] = 0
        
        assert found == 1   # debugging: ensure that only 1 piece is found
        
        ## Move the piece according to the given action

        # Flags for emptying rep_set
        captured = False        
        B_irreversible = piecetype % 7 == 6

        if (a <= 7):
            move_horizontal = False
            x = x + (a + 1)
        elif (a <= 15):
            move_horizontal = False
            x = x - (a - 7)
        elif (a <= 24):
            y = y + (a - 15)
        elif (a <= 33):
            y = y - (a - 24)
        elif (a <= 35):
            x = x + (a - 33)
            y = y + (a - 33)
        elif (a <= 37):
            x = x - (a - 35)
            y = y + (a - 35)
        elif (a <= 39):
            x = x - (a - 37)
            y = y - (a - 37)
        elif (a <= 41):
            x = x + (a - 39)
            y = y - (a - 39)
        elif (a <= 43):
            x = x + 2
            y = if (a == 42) y + 1 else y - 1
        elif (a <= 45):
            x = x - 2
            y = if (a == 44) y + 1 else y - 1
        elif (a <= 47):
            x = x + 1
            y = if (a == 46) y + 2 else y - 2
        elif (a <= 49):
            x = x - 1
            y = if (a == 48) y + 2 else y - 2
        elif (a <= 51):
            x = x + 3
            y = if (a == 50) y + 2 else y - 2
        elif (a <= 53):
            x = x - 3
            y = if (a == 52) y + 2 else y - 2
        elif (a <= 55):
            x = x + 2
            y = if (a == 54) y + 3 else y - 3
        elif (a <= 57):
            x = x - 2
            y = if (a == 56) y + 3 else y - 3
        else:
            # This should not happen
            assert False

        assert x >= 0 and x < nx and y >= 0 and y < ny
        
        self.pieces[piecetype][x][y] = 1
        captured = self.pieces[opp_piecetype][x][y]
        self.pieces[opp_piecetype][x][y] = 0

        ## 

    def _discover_move(self, origin, direction):
        """ Returns the endpoint for a legal move, starting at the given origin,
        moving by the given increment."""
        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):
        """ Gets the list of flips for a vertex and direction to use with the
        execute_move function """
        #initialize variables
        flips = [origin]

        for x, y in Board._increment_move(origin, direction, self.n):
            #print(x,y)
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
                #print(flips)
                return flips

        return []

    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)): 
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])
    
    def recentConfigToString(self):
        """
        Gives the string representation of the most recent board
        """
        return np.array2string(self.pieces[:self.nm])
