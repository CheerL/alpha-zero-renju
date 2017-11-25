# -*- coding:utf-8 -*-
import logging
import numpy as np
import utils
from utils.logger import Logger
from collections import namedtuple
from board import Board
from mcts import MCT

History = namedtuple('History', ['board', 'x', 'y', 'state'])

class Player(object):
    logger = Logger('game')

    def __init__(self, color, player_type, size):
        self.color = color
        self.player_type = player_type
        self.board = Board(color, size)
        self.oppo_board = None
        self.history = list()
        self.logger.info('Create new {} player'.format(utils.COLOR[self.color]))

    def move(self, x, y):
        '''在`(x, y)`处落子'''
        index = self.board.xy2index(x, y)
        assert self.oppo_board, '对手盘面未知'
        assert self.board.board[index] == 0 and self.oppo_board.board[index] == 0, '目标位置已经有子'
        self.add_history(x, y)
        self.board.board[index] = 1
        # msg = '{}: ({},{})'.format('Black' if self.board.color is utils.BLACK else 'White', x, y)
        # print(msg)

    def undo(self):
        history = self.history.pop()
        self.board.board = history.board
        self.logger.info('{} undo: ({},{})'.format(utils.COLOR[self.color], history.x, history.y))
        return utils.COLOR[self.color], history.x, history.y

    def add_history(self, x, y):
        history = History(self.board.board.copy(), x, y, None)
        self.history.append(history)

    @property
    def show_board(self):
        '''以`size * size`的矩阵形式输出'''
        return self.board.board.reshape(self.board.size, self.board.size) * self.color


class GomocupPlayer(Player):
    def __init__(self, color, size):
        super(GomocupPlayer, self).__init__(color, utils.GOMOCUP, size)


class HumanPlayer(Player):
    def __init__(self, color, size):
        super(HumanPlayer, self).__init__(color, utils.HUMAN, size)


class RandomPlayer(Player):
    def __init__(self, color, size):
        super(RandomPlayer, self).__init__(color, utils.RANDOM, size)

    def get_move(self):
        empty_pos = np.where(self.board.board + self.oppo_board.board == 0)[0]
        index = np.random.choice(empty_pos)
        return self.board.index2xy(index)

class MCTSPlayer(Player):
    def __init__(self, color, size):
        super(MCTSPlayer, self).__init__(color, utils.MCTS, size)
        self.move_probability = None

    def add_history(self, x, y):
        if self.move_probability is not None:
            history = History(self.board.board.copy(), x, y, self.move_probability.copy())
        else:
            history = History(self.board.board.copy(), x, y, None)
        self.history.appendleft(history)

    def get_move(self):
        raise NotImplementedError()
