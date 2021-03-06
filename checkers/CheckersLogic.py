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

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,-1),(-1,-1),(-1,1)]

    def __init__(self, n):
        "Set up initial board configuration."

        self.square_content = {
            -2: "V",
            -1: "X",
            +0: "-",
            +1: "O",
            +2: "D"
        }

        self.n = n
        self.count = 0
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n


        # Set up the initial pieces.
        for i in range(self.n):
            if (i+0)%2 == 0:
                self.pieces[0][i] = 1
            else:
                self.pieces[1][i] = 1
            if (i+self.n-1)%2 == 0:
                self.pieces[self.n-1][i] = -1
            else:
                self.pieces[self.n-2][i] = -1
        '''
        for i in range(self.n):
            for j in range(self.n):
                if (i+j)%2==0:
                    self.pieces[i][j] = 2
                    self.display()
                    print(self.get_legal_moves(1))
                    self.pieces[i][j] = 0
        '''


    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def countScore(self, color):
        """Counts the # pieces of the given color
        (1 for white, -1 for black, 0 for empty spaces)"""
        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]*color > 0:
                    count += abs(self[x][y])
                if self[x][y]*color < 0:
                    count -= abs(self[x][y])
        return count

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color or self[x][y]==color*2:
                    newmoves = self.get_moves_for_square((x,y))
                    moves.update(newmoves)
        #print(moves)
        return list(moves)

    def game_over(self):
        player1 = 0 # player '1'
        player2 = 0 # player '-1'
        for i in range(self.n):
            for j in range(self.n):
                if self[i][j] > 0:
                    player1 = player1 + 1
                elif self[i][j] < 0:
                    player2 = player2 + 1
        if player1 == 0:
            return -1
        elif player2 == 0:
            return 1
        elif self.count >= 150:
            if self.countScore(1) > 0:
                return 1
            elif self.countScore(1) == 0:
                return 0.01
            else:
                return -1
        else:
            return 0

    #def has_legal_moves(self, color):
    #    for y in range(self.n):
    #        for x in range(self.n):
    #            if self[x][y]==color:
    #                newmoves = self.get_moves_for_square((x,y))
    #                if len(newmoves)>0:
    #                    return True
    #    return False

    def get_moves_for_square(self, square):
        """Returns all the legal moves from the given square.
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
            if (color * direction[0] > 0 or abs(color) == 2) and (0 <= x+direction[0] and self.n-1 >= x+direction[0] and 0 <= y+direction[1] and self.n-1 >= y+direction[1]):
                if self[x+direction[0]][y+direction[1]] == 0:
                    moves.append(((x,y),direction))
                elif 0 <= x+2*direction[0] and self.n-1 >= x+2*direction[0] and 0 <= y+2*direction[1] and self.n-1 >= y+2*direction[1]:
                    if self[x+direction[0]][y+direction[1]] * color < 0 and self[x+2*direction[0]][y+2*direction[1]] == 0:
                        moves.append(((x,y),tuple(2*i for i in direction)))
        # return the generated move list
        return moves

    def execute_move(self, move, color):
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        #Much like move generation, start at the new piece's square and
        #follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.

        #print("Let's MOVE!!!")
        #print(move)
        #print("THAT's how we're gonna move")
        ((x,y),(z,w)) = move
        #print(x,y,z,w)
        self[x+z][y+w] = self[x][y]
        self[x][y] = 0
        self.count += 1

        if abs(z) == 2:
            self[x+z//2][y+w//2] = 0

        if (x+z==0 or x+z==self.n-1) and abs(self[x+z][y+w]) == 1:
            self[x+z][y+w] = 2*self[x+z][y+w]

    def display(self):
        print("   ", end="")
        for y in range(self.n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(self.n):
            print(y, "|", end="")    # print the row #
            for x in range(self.n):
                piece = self[y][x]    # get the piece to print
                print(self.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
