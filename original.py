import numpy as np
import time
import pickle
import random
from pprint import pprint
import os
import pandas as pd

# BOARD VALUE
EMPTY        : int =  0
WHITE        : int =  1
BLACK        : int = -1
WALL         : int =  2
BOARD_SIZE   : int =  8

# DIRECTIONS
NONE         : int = 0
LEFT         : int = 2**0  # =1
TOP_LEFT     : int = 2**1  # =2
TOP          : int = 2**2  # =4
TOP_RIGHT    : int = 2**3  # =8
RIGHT        : int = 2**4  # =16
BOTTOM_RIGHT : int = 2**5  # =32
BOTTOM       : int = 2**6  # =64
BOTTOM_LEFT  : int = 2**7  # =128

# DISPLAY AXIS
HORIZONTAL : list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
VERTICAL   : list = ['1', '2', '3', '4', '5', '6', '7', '8']

# OTHER
MAX_TURNS  : int  = 60
FIRST_TURN : int  = BLACK
DISPLAY    : bool = True
SHOW_HELP  : bool = False
SHOW_SLOW  : bool = False
SWAP_COLOR : bool = True

# PLAYER
class HUMAN:

    def __init__(self):
        self.name    : str = "human"
        self.wins    : int = 0
        self.control : str = "player"

    def addState(self, _) -> None:
        pass

    def choice(self) -> str:

        choice = input()
        return choice

    def feedReward(self, _) -> None:
        pass

    def reset(self) -> None:
        pass

    def savePolicy(self) -> None:
        pass

    def loadPolicy(self) -> None:
        pass


# COMPUTER
class AI:

    reward = np.array([
        [5, 1, 4, 3, 3, 3, 4, 1, 5],
        [1, 1, 4, 2, 2, 2, 4, 1, 1],
        [4, 4, 4, 1, 1, 1, 4, 4, 4],
        [3, 1, 3, 1, 1, 1, 3, 1, 2],
        [3, 1, 3, 1, 1, 1, 3, 1, 2],
        [3, 1, 3, 1, 1, 1, 3, 1, 2],
        [4, 4, 4, 1, 1, 1, 4, 4, 4],
        [1, 1, 4, 2, 2, 2, 4, 1, 1],
        [5, 1, 4, 3, 3, 3, 4, 1, 5]])

    def __init__(self, name: str, exp_rate: float =0.3, load: bool =False, save: bool =False, algorithm: str ="random") -> None:

        self.name         : str = name
        self.states       : list = []
        self.lr           : float = 0.3
        self.exp_rate     : float = exp_rate
        self.decay_gamma  : float = 0.9
        self.states_value : dict = {}
        self.wins         : int = 0
        self.result       : list = []
        self.control      : str = "ai"
        self.algorithm    : str = algorithm
        self.color        : str = ""
        self.save         : bool = save

        if load:
            self.loadPolicy()

    def reset(self) -> None:
        self.states = []

    """ evaluate the scores """
    def evaluate(self, board: np.ndarray) -> int:

        total = 0
        for i in range(1, BOARD_SIZE-1):
            for j in range(1, BOARD_SIZE-1):
                if board[i, j] == self.color:
                    total += self.lr * self.reward[i, j]
                elif board[i, j] == -self.color:
                    total -= self.lr * self.reward[i, j]
                else:
                    continue

        return total

    """ add board status """
    def addState(self, state: np.ndarray) -> None:
        if self.algorithm == "q-learn":
            self.states.append(np.copy(state))

    """ Update Computer """
    def feedReward(self, reward: int, count: int) -> None:

        for st in reversed(self.states):
            key = str(st.flatten().tolist())
            if key not in self.states_value:
                self.states_value[key] = 0.0

            self.states_value[key] += self.lr * \
                (self.decay_gamma * reward - self.states_value[key])
            reward = self.states_value[key]
            self.result.append([self.evaluate(st), reward, count])

    """ Save learned data """
    def savePolicy(self) -> None:

        if self.save and self.algorithm == "q-learn":
            fw = open('policy_'+self.name, 'wb')
            pickle.dump(self.states_value, fw)
            fw.close()

    """ Load learned data """
    def loadPolicy(self) -> None:

        if self.algorithm == "q-learn" and os.path.isfile('policy'+self.name):
            fr = open('policy_'+self.name, 'rb')
            self.states_value = pickle.load(fr)
            fr.close()


# REVERSI GAME
class REVERSI:

    def __init__(self, p1: AI or HUMAN, p2: AI or HUMAN) -> None:

        # Color -> Player
        self.players = {
            BLACK: p1,
            WHITE: p2
        }

        # Players
        self.p1 = p1
        self.p2 = p2

        self.reset()

    """ Reset the game status """
    def reset(self) -> None:

        # Board Matrix
        self.board : np.ndarray = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
        
        # Walls
        self.board[0, :]              = WALL
        self.board[:, 0]              = WALL
        self.board[BOARD_SIZE + 1, :] = WALL
        self.board[:, BOARD_SIZE + 1] = WALL

        # Initial Placement
        self.board[4, 4] = WHITE
        self.board[5, 5] = WHITE
        self.board[4, 5] = BLACK
        self.board[5, 4] = BLACK

        # No. of turns
        self.turns = 0

        # Set colors to player
        if SWAP_COLOR:

            if self.players[BLACK] != self.p1:
                self.players[BLACK] = self.p1
                self.players[WHITE] = self.p2
                self.p1.color = BLACK
                self.p2.color = WHITE
            else:
                self.players[BLACK] = self.p2
                self.players[WHITE] = self.p1
                self.p2.color = BLACK
                self.p1.color = WHITE

        else:
            self.players[BLACK] = self.p1
            self.players[WHITE] = self.p2
            self.p1.color = BLACK
            self.p2.color = WHITE

        self.p1.reset()
        self.p2.reset()

        # Current Player Color
        self.color = FIRST_TURN

        # Possible Direction to Flip
        self.dir_available : np.ndarray = np.zeros(
            (BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
        
        # Possible Position to Move
        self.pos_available : np.ndarray = np.zeros(
            (BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)

        # Possible Choices
        self.choice_available : list = self.check_available()

    """ Check specific direction available """
    def check_alignment(self, x: int, x_update: int, y: int, y_update: int, dir: int, check_dir: int, color: int) -> int :

        # Check until the other players coin ends
        while self.board[x, y] == - color:
            x += x_update
            y += y_update

        # Update the dir if current player coin exist on the direction
        if self.board[x, y] == color:
            return (dir | check_dir)

        return dir

    """ Check all direction available """
    def check_flip_dir(self, x: int, y: int, color: int) -> int:

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
    def check_available(self) -> list:

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
    def check_move(self, move: list[str, str]) -> bool:

        if not move:
            return False

        if move[0] in HORIZONTAL:
            if move[1] in VERTICAL:
                return True

        return False

    """ Check Game Over """
    def check_game_over(self) -> bool:

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
    def flip_each(self, board: np.ndarray, x: int, x_update: int, y: int, y_update: int) -> None:

        while board[x, y] == - self.color:

            board[x, y] = self.color

            x += x_update
            y += y_update

    """ Flip all coins, all dir """
    def flip(self, board: np.ndarray, x: int, y: int) -> np.ndarray:
        
        board[x, y] = self.color
        dir = self.dir_available[x, y]

        # Left
        if dir & LEFT:
            self.flip_each(board, x-1, -1, y, 0)

        # Top Left
        if dir & TOP_LEFT:
            self.flip_each(board, x-1, -1, y-1, -1)

        # Top
        if dir & TOP:
            self.flip_each(board, x, 0, y-1, -1)

        # Top Right
        if dir & TOP_RIGHT:
            self.flip_each(board, x+1, +1, y-1, -1)

        # Right
        if dir & RIGHT:
            self.flip_each(board, x+1, +1, y, 0)

        # Bottom Right
        if dir & BOTTOM_RIGHT:
            self.flip_each(board, x+1, +1, y+1, +1)

        # Bottom
        if dir & BOTTOM:
            self.flip_each(board, x, 0, y+1, +1)

        # Bottom Left
        if dir & BOTTOM_LEFT:
            self.flip_each(board, x-1, -1, y+1, +1)

        return board

    """ Place the coins """
    def move(self, x: int, y: int) -> bool:

        if x < 1 or BOARD_SIZE < x:
            return False
        if y < 1 or BOARD_SIZE < y:
            return False
        if self.pos_available[x, y] == 0:
            return False

        # Flip the coins
        self.board = self.flip(self.board, x, y)

        # Increase Turns
        self.turns += 1

        # Change Color
        self.color = - self.color

        # Store available places
        self.choice_available = self.check_available()

        return True

    """ Display Board """
    def display(self, iter: int, report: str) -> None:

        if not DISPLAY:
            return

        os.system('clear')

        # Horizontal Axis
        print("=== WINNING TOTAL ===\n")
        print(
            f"P1: {self.p1.wins}, P2: {self.p2.wins}, TIE: {iter - 1 - self.p1.wins-self.p2.wins}\n")
        print(report)
        print("=====================\n")

        print("GAME No.:", iter)
        print('\n  a b c d e f g h')
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
    def count_coin(self, color: int) -> int:

        return np.count_nonzero(self.board[:, :] == color)

    """ Count All Coin """
    def count_all(self) -> tuple[int,int] :

        count_black = self.count_coin(BLACK)
        count_white = self.count_coin(WHITE)

        return count_black, count_white

    """ Choice of Player """
    def choice(self, choices: list, player: AI or HUMAN, board: np.ndarray) -> tuple[int or None,int or None] :

        if (choices[0][0], choices[0][1]) == (None, None):
            if DISPLAY:
                print("PASS")
            return choices[0]

        if DISPLAY:
            print("\nCHOICES:", choices)

        # Random Selection
        if player.algorithm == "random":
            select = random.randint(0, len(choices)-1)
            action = choices[select]

            if DISPLAY:
                print("METHOD: RANDOM")

        if player.algorithm == "q-learn":
            # Selects random choice if exploration rat is higher¥
            if np.random.uniform(0, 1) <= player.exp_rate:
                select = random.randint(0, len(choices)-1)
                action = choices[select]

                if DISPLAY:
                    print("METHOD: RANDOM")

            # Selects choice previous comparing currentboard and nextboard
            else:
                value_max = -999.0

                for x, y in choices:
                    next_board = self.flip(board.copy(), x, y)
                    key = str(next_board.flatten().tolist())

                    if key in player.states_value:
                        value = player.states_value[key]
                    else:
                        player.states_value[key] = 0.0
                        value = 0.0

                    if value >= value_max:
                        value_max = value
                        action = [x, y]

                if DISPLAY:

                    print("METHOD: Q-RL")
                    print("VALUE:", value_max)

        if DISPLAY:

            print(f"SELECTS: {action}, (INDEX: {choices.index(action) + 1})")

        return action

    """ Play Reversi """
    def start(self, iter: int=1) -> None:

        p1, p2 = 0, 0
        report = ""
        for i in range(1, iter+1):

            if i % iter%(iter/10) == 0:
                report += f'At{i}, P1: {self.p1.wins}({self.p1.wins-p1}), P2: {self.p2.wins}({self.p2.wins-p2}), TIE: {i-1-self.p1.wins-self.p2.wins}\n'
                p1, p2 = self.p1.wins, self.p2.wins
            while True:

                self.display(i,report)

                if DISPLAY:

                    if self.color == BLACK:
                        print(
                            f"\n{self.players[self.color].name} X's (Black) Turn: ", end="")
                    else:
                        print(
                            f"\n{self.players[self.color].name} O's (White) Turn: ", end="")

                if self.players[self.color].control == "ai":
                    self.players[self.color].addState(self.board)

                choices = self.choice_available
                if not self.pos_available[:, :].any():
                    choices = [[None, None]]

                if self.players[self.color].control != "player":
                    select = self.choice(
                        choices, self.players[self.color], np.copy(self.board))

                if not self.pos_available[:, :].any():

                    self.color = - self.color
                    self.choice_available = self.check_available()

                    if DISPLAY:
                        print('Pass\n')
                    continue

                if self.players[self.color].control == "player":

                    select = self.players[self.color].choice()
                    if self.check_move(select):
                        x = int(HORIZONTAL.index(select[0]) + 1)
                        y = int(VERTICAL.index(select[1]) + 1)
                        select = x, y

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
                        self.display(i,report)
                    break

                #time.sleep(0.05)
                if SHOW_SLOW:
                    time.sleep(0.5)

            count_black, count_white = self.count_all()

            if DISPLAY:
                print("\nX:", count_black, "O:", count_white)

            diff = (count_black-count_white)/(count_black+count_white)
            if count_black > count_white:

                if DISPLAY:
                    print(self.players[BLACK].name, 'Winner X\n')

                self.players[BLACK].wins += 1
                self.players[BLACK].feedReward(3, diff)
                self.players[WHITE].feedReward(1, diff)

            elif count_white > count_black:

                if DISPLAY:
                    print(self.players[WHITE].name, 'Winnter O\n')

                self.players[WHITE].wins += 1
                self.players[BLACK].feedReward(1, diff)
                self.players[WHITE].feedReward(3, diff)
            else:

                if DISPLAY:
                    print('Tie self\n')

                self.players[BLACK].feedReward(1, diff)
                self.players[WHITE].feedReward(1, diff)

            self.reset()

        self.p1.savePolicy()
        self.p2.savePolicy()


def main():

    # TRAIN
    p1 = AI(
        name="p1",
        exp_rate=0.3,
        algorithm="q-learn",
        save=True,
        load=False
    )
    p2 = AI(
        name="p2",
        exp_rate=0.3,
        algorithm="q-learn",
        save=True,
        load=False
    )

    # TRAIN
    train = REVERSI(p1, p2)
    train.start(100000)

    # TEST
    random_player = AI(name="random")
    test1 = REVERSI(p1, random_player)
    test1.play(500)

    test2 = REVERSI(p2, random_player)
    test2.play(500)
    
    # PLAY
    player = HUMAN()
    play = REVERSI(player,p2)
    #play.start(1)


if __name__ == '__main__':
    main()
