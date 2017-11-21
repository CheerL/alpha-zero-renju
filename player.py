import numpy as np
import config as cfg
from board import Board

class Player(object):
    def __init__(self, color, player_type, size):
        self.player_type = player_type
        self.board = Board(color, size)
        self.oppo_board = None

    def move(self, x, y):
        '''在`(x, y)`处落子'''
        index = self.board.xy2index(x, y)
        assert self.oppo_board is not None, '对手盘面未知'
        assert self.board[index] == 0 and self.oppo_board[index] == 0, '目标位置已经有子'
        self.board[index] = self.board.color
        msg = '{}: ({},{})'.format('Black' if self.board.color is cfg.BLACK else 'White', x, y)
        print(msg)


class GomocupPlayer(Player):
    def __init__(self, color, size):
        super().__init__(color, cfg.GOMOCUP, size)


class HumanPlayer(Player):
    def __init__(self, color, size):
        super().__init__(color, cfg.HUMAN, size)

    def get_move(self):
        pass


class RobotPlayer(Player):
    def __init__(self, color, size):
        super().__init__(color, cfg.ROBOT, size)

    def get_move(self):
        empty_pos = np.where(self.board.board + self.oppo_board == 0)[0]
        index = np.random.choice(empty_pos)
        return self.board.index2xy(index)
