import numpy as np
import time
import pickle
import random
from pprint import pprint
import os
import pandas as pd
from tqdm import tqdm

# BOARD VALUE
EMPTY = 0
WHITE = 1
BLACK = -1
WALL = 2
BOARD_SIZE = 8

# DIRECTIONS
NONE = 0
LEFT = 2**0  # =1
TOP_LEFT = 2**1  # =2
TOP = 2**2  # =4
TOP_RIGHT = 2**3  # =8
RIGHT = 2**4  # =16
BOTTOM_RIGHT = 2**5  # =32
BOTTOM = 2**6  # =64
BOTTOM_LEFT = 2**7  # =128

# DISPLAY AXIS
HORIZONTAL = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
VERTICAL = ['1', '2', '3', '4', '5', '6', '7', '8']

# OTHER
MAX_TURNS = 60
FIRST_TURN = BLACK
DISPLAY = False
SHOW_HELP = True
SHOW_SLOW = False


def change(board):

    board[1:9, 1: BOARD_SIZE + 1] = BLACK
    board[8] = np.array([
        WALL, BLACK, BLACK, BLACK, WHITE, NONE, NONE, WHITE, BLACK, WALL
    ])


# REVERSI GAME
class REVERSI:

    
    def __init__(self,p1,p2):

        self.player = {
            BLACK: p1,
            WHITE: p2
        }
        self.p1 = p1
        self.p2 = p2
        self.reset()

    """ Reset the game status """
    def reset(self):
        
        self.board = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
        # Walls
        self.board[0, :] = WALL
        self.board[:, 0] = WALL
        self.board[BOARD_SIZE + 1, :] = WALL
        self.board[:, BOARD_SIZE + 1] = WALL

        # Initial Placement
        self.board[4, 4] = WHITE
        self.board[5, 5] = WHITE
        self.board[4, 5] = BLACK
        self.board[5, 4] = BLACK

        # change(self.board)
        # No. of turns
        self.turns = 0
        self.moves = {
            BLACK: [],
            WHITE: []
        }
        

        if self.player[BLACK] != self.p1:
            self.player[BLACK] = self.p1
            self.player[WHITE] = self.p2
            self.p1.color = BLACK
            self.p2.color = WHITE
        else:
            self.player[BLACK] = self.p2
            self.player[WHITE] = self.p1
            self.p2.color = BLACK
            self.p1.color = WHITE
        
        self.p1.reset()
        self.p2.reset()

        # Current Player Color
        self.color = FIRST_TURN

        self.dir_available = np.zeros(
            (BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
        self.pos_available = np.zeros(
            (BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)

        self.choice_available = self.check_available()

    """ Check specific direction available """
    def check_alignment(self, x, x_update, y, y_update, dir, check_dir, color):

        # Check until the other players coin ends
        while self.board[x, y] == - color:
            x += x_update
            y += y_update

        # Update the dir if current player coin exist on the direction
        if self.board[x, y] == color:
            return (dir | check_dir)

        return dir

    """ Check all direction available """
    def check_flip_dir(self, x, y, color):

        # Flippable Direction
        dir = 0

        # If already placed
        if(self.board[x, y] != EMPTY):
            return dir

        # Left
        if(self.board[x - 1, y] == - color):
            dir = self.check_alignment(x-2, -1, y, 0, dir, LEFT, color)

        # Top Left
        if(self.board[x - 1, y - 1] == - color):
            dir = self.check_alignment(x-2, -1, y-2, -1, dir, TOP_LEFT, color)

        # Top
        if(self.board[x, y - 1] == - color):
            dir = self.check_alignment(x, 0, y-2, -1, dir, TOP, color)

        # Top Right
        if(self.board[x + 1, y - 1] == - color):
            dir = self.check_alignment(x+2, 1, y-2, -1, dir, TOP_RIGHT, color)

        # Right
        if(self.board[x + 1, y] == - color):
            dir = self.check_alignment(x+2, 1, y, 0, dir, RIGHT, color)

        # Bottom Right
        if(self.board[x + 1, y + 1] == - color):
            dir = self.check_alignment(
                x+2, 1, y+2, 1, dir, BOTTOM_RIGHT, color)

        # Bottom
        if(self.board[x, y + 1] == - color):
            dir = self.check_alignment(x, 0, y+2, 1, dir, BOTTOM, color)

        # Bottom Left
        if(self.board[x - 1, y + 1] == - color):
            dir = self.check_alignment(
                x-2, -1, y+2, 1, dir, BOTTOM_LEFT, color)

        return dir

    """ Check position and direction available """
    def check_available(self):

        # Mkae every position to False
        self.pos_available[:, :] = False
        choices = []
        
        # Check every position other than walls
        for x in range(1, BOARD_SIZE + 1):
            for y in range(1, BOARD_SIZE + 1):

                # check every direction
                dir = self.check_flip_dir(x, y, self.color)

                # set possible direction
                self.dir_available[x, y] = dir

                if dir != 0:
                    self.pos_available[x, y] = True
                    choices.append([x, y])

        return choices

    """ Check whether the move is possible """
    def check_move(self, move):

        if not move:
            return False

        if move[0] in HORIZONTAL:
            if move[1] in VERTICAL:
                return True

        return False

    """ Check Game Over """
    def check_game_over(self):

        # More than 60 turns
        if self.turns >= MAX_TURNS:
            return True

        if self.pos_available[:, :].any():
            return False

        for x in range(1, BOARD_SIZE + 1):
            for y in range(1, BOARD_SIZE + 1):

                if self.check_flip_dir(x, y, - self.color) != 0:
                    return False

        return True

    """ Flip one direction  """
    def flip_each(self, x, x_update, y, y_update):

        while self.board[x, y] == - self.color:

            self.board[x, y] = self.color

            x += x_update
            y += y_update

    """ Flip all coins, all dir """
    def flip(self, x, y):
        self.board[x, y] = self.color

        dir = self.dir_available[x, y]

        # Left
        if dir & LEFT:
            self.flip_each(x-1, -1, y, 0)

        # Top Left
        if dir & TOP_LEFT:
            self.flip_each(x-1, -1, y-1, -1)

        # Top
        if dir & TOP:
            self.flip_each(x, 0, y-1, -1)

        # Top Right
        if dir & TOP_RIGHT:
            self.flip_each(x+1, +1, y-1, -1)

        # Right
        if dir & RIGHT:
            self.flip_each(x+1, +1, y, 0)

        # Bottom Right
        if dir & BOTTOM_RIGHT:
            self.flip_each(x+1, +1, y+1, +1)

        # Bottom
        if dir & BOTTOM:
            self.flip_each(x, 0, y+1, +1)

        # Bottom Left
        if dir & BOTTOM_LEFT:
            self.flip_each(x-1, -1, y+1, +1)

    """ Place the coins """
    def move(self, x, y):

        if x < 1 or BOARD_SIZE < x:
            return False
        if y < 1 or BOARD_SIZE < y:
            return False
        if self.pos_available[x, y] == 0:
            return False

        # Flip the coins
        self.flip(x, y)

        # Increase Turns
        self.turns += 1

        self.moves[self.color].append([x, y, self.count_coin(self.color)])

        # Change Color
        self.color = - self.color

        # Store available places
        self.choice_available = self.check_available()

        return True

    """ Display Board """
    def display(self):

        if not DISPLAY:
            return

        os.system('clear')
        
        # Horizontal Axis
        print('  a b c d e f g h')
        for y in range(1, 9):

            # Vertical Axis
            print(y, end=" ")

            for x in range(1, 9):

                grid = self.board[x, y]

                if self.pos_available[x, y] and SHOW_HELP:
                    print('p', end=" ")
                elif grid == EMPTY:
                    print('□', end=" ")
                elif grid == WHITE:
                    print('o', end=" ")
                elif grid == BLACK:
                    print('x', end=" ")

            print()

    """ Count Coin """
    def count_coin(self, color):

        return np.count_nonzero(self.board[:, :] == color)

    """ Count All Coin """
    def count_all(self):

        count_black = self.count_coin(BLACK)
        count_white = self.count_coin(WHITE)

        return count_black, count_white

    """ Play Reversi """
    def play(self,iter=1):

        p1,p2 = 0,0
        for i in range(1,iter+1):
            
            if i % 100 == 0:
                print(f'{i}, P1: {self.p1.wins}({self.p1.wins-p1}), P2: {self.p2.wins}({self.p2.wins-p2}), TIE: {i-self.p1.wins-self.p2.wins}')
                p1,p2 = self.p1.wins,self.p2.wins
            while True:

                self.display()

                if DISPLAY:
                    
                    if self.color == BLACK:
                        print(f"\n{self.player[self.color].name} X's (Black) Turn: ", end="")
                    else:
                        print(f"\n{self.player[self.color].name} O's (White) Turn: ", end="")
                
       
                if self.player[self.color].control=="ai":
                    self.player[self.color].addState(self.board)
                
                choices = self.choice_available
                if not self.pos_available[:, :].any():
                    choices = [[None,None]]

                select = self.player[self.color].choice(
                    choices, self.board
                    )

                if not self.pos_available[:, :].any():
                    
                    self.color = - self.color
                    self.choice_available = self.check_available()

                    if DISPLAY:
                        print('Pass\n')
                    continue

                if self.player[self.color].control == "player":
                    
                    if self.check_move(select):
                        x = int(HORIZONTAL.index(select[0]) + 1)
                        y = int(VERTICAL.index(select[1]) + 1)
                        select = x,y
                    
                    else:    
                        if DISPLAY:
                            print("\nImproper Format")
                        continue

                x, y = select
                if not self.move(x, y):
                    if DISPLAY:
                        print('\nImplaceble')
                    continue

                if self.check_game_over():
                    if DISPLAY:
                        self.display()
                        print('\nGame Over:', end=" ")
                    break
                
                if SHOW_SLOW:
                    time.sleep(0.5)

            count_black, count_white = self.count_all()
            diff = (count_black-count_white)/(count_black+count_white)
            if count_black > count_white:
                
                if DISPLAY:
                    print(self.player[BLACK].name,'Winner Black\n')

                self.player[BLACK].wins +=1
                self.player[BLACK].feedReward(3,diff)
                self.player[WHITE].feedReward(1,diff)

            elif count_white > count_black:
                
                if DISPLAY:
                    print(self.player[WHITE].name,'Winnter White\n')

                self.player[WHITE].wins +=1
                self.player[BLACK].feedReward(1,diff)
                self.player[WHITE].feedReward(3,diff)
            else:
                
                if DISPLAY:
                    print('Tie self\n')

                self.player[BLACK].feedReward(2,diff)
                self.player[WHITE].feedReward(2,diff)
                
            self.reset()

        self.p1.savePolicy()
        self.p2.savePolicy()

# PLAYER
class HUMAN:
    def __init__(self):
        self.name = "human"
        self.wins = 0
        self.control = "player"

    # append a hash state
    def addState(self, _):
        pass
    
    def choice(self, _, __, ___):
        
        choice = input()
        return choice

    def feedReward(self, _):
        pass

    def reset(self):
        pass

    def savePolicy(self):
        pass
    
    def loadPolicy(self):
        pass


# COMPUTER
class AI:

    reward = np.array([
        [5,1,4,3,3,3,4,1,5],
        [1,1,4,2,2,2,4,1,1],
        [4,4,4,1,1,1,4,4,4],
        [3,1,3,1,1,1,3,1,2],
        [3,1,3,1,1,1,3,1,2],
        [3,1,3,1,1,1,3,1,2],
        [4,4,4,1,1,1,4,4,4],
        [1,1,4,2,2,2,4,1,1],
        [5,1,4,3,3,3,4,1,5]])

    def __init__(self, name, exp_rate=0.3, load=False, save=False, algorithm="random"):
        self.name = name
        self.states = []
        self.lr = 0.3
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}
        self.wins = 0
        self.result = []
        self.control = "ai"
        self.algorithm = algorithm
        self.color = None
        self.save = save
        if load:
            self.loadPolicy()

    def reset(self):
        self.states = []

    """ evaluate the scores """
    def evaluate(self,board):

        total = 0
        for i in range(1,BOARD_SIZE-1):
            for j in range(1,BOARD_SIZE-1):
                if board[i,j] == self.color:
                    total += self.lr * self.reward[i,j]
                elif board[i,j] == -self.color:
                    total -= self.lr * self.reward[i,j]
                else:
                    continue
        
        return total

    """ add board status """
    def addState(self, state):
        if self.algorithm=="q-learn":
            self.states.append(np.copy(state))

    """ actions made by the ai """
    def choice(self, choices, current_board):

        # Random Selection
        if self.algorithm == "random":
            select = random.randint(0, len(choices)-1)
            action = choices[select]

        # Q - RL
        if self.algorithm == "q-learn":

            # Selects random choice if exploration rat is higher¥
            if np.random.uniform(0, 1) <= self.exp_rate:
                select = random.randint(0, len(choices)-1)
                action = choices[select]         
            
            # Selects choice previous comparing currentboard and nextboard
            else:
                value_max = -999.0
                for x,y in choices:
                    next_board = np.copy(current_board)
                    next_board[x,y] = self.color
                    key = str(next_board.flatten().tolist())

                    if  key in self.states_value:
                        value = self.states_value[key]
                    else:
                        self.states_value[key] = 0.0
                        value = 0.0

                    if value >= value_max:
                        value_max = value
                        action = x,y

                #print("CHOICE",action,value_max)

        return action

    """ Update Computer """
    def feedReward(self, reward, count):
        
        #pprint(self.states[0]==self.states[1])
        for st in reversed(self.states):
            key = str(st.flatten().tolist())
            if key not in self.states_value:
                self.states_value[key] = 0.0
            
            self.states_value[key] += self.lr * (self.decay_gamma * reward - self.states_value[key])
            reward = self.states_value[key]
            #print(reward)
            self.result.append([self.evaluate(st), reward, count])

        #os.system("clear")

    """ Save learned data """
    def savePolicy(self):

        if self.save and self.algorithm == "q-learn":
            fw = open('policy_'+self.name , 'wb')
            pickle.dump(self.states_value, fw)
            fw.close()

    """ Load learned data """
    def loadPolicy(self):

        if self.algorithm == "q-learn" and os.path.isfile('policy'+self.name):
            fr = open('policy_'+self.name, 'rb')
            self.states_value = pickle.load(fr)
            fr.close()


def main():

    # TRAIN
    """p1,p2 = AI("p1",exp_rate=0.5,algorithm="q-learn",save=True,load=True),AI("p2",exp_rate=0.3,algorithm="q-learn",save=True,load=True)
    game = REVERSI(p1,p2)
    game.play(10**4)"""

    p3 = AI("test")
    p4,p5 = AI("p1",exp_rate=0.5,algorithm="q-learn",save=True,load=True),AI("p2",exp_rate=0.3,algorithm="q-learn",save=True,load=True)

    test = REVERSI(p3,p4)
    test.play(1000)
    
    p6 = AI("test")
    test = REVERSI(p5,p6)
    test.play(1000)

if __name__ == '__main__':
    main()
