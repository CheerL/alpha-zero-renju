# -*- coding:utf-8 -*-
from __future__ import unicode_literals

import logging
import numpy as np
import utils
from utils.logger import Logger
from mcts import MCT

class Player(object):
    logger = Logger('game')

    def __init__(self, color, player_type, board):
        self.color = color
        self.color_str = utils.COLOR[color]
        self.player_type = player_type
        self.board = board
        self.logger.info('Create new {} player'.format(self.color_str))

    def move(self, x, y):
        '''在`(x, y)`处落子'''
        index = self.board.xy2index(x, y)
        assert self.board.board[index] == utils.EMPTY, '目标位置已经有子'
        self.board.board[index] = self.color
        self.logger.info('{}: ({}:{})'.format(self.color_str, x, y))

    def undo(self, x, y):
        self.logger.info('{} undo: ({},{})'.format(self.color_str, x, y))

    def win(self):
        self.logger.info('{} win'.format(self.color_str))

    def judge_win(self, x, y):
        return self.board.judge_win(x, y, self.color)

    @property
    def show_board(self):
        '''以`size * size`的矩阵形式输出'''
        return self.board.get_show_board(self.color)


class GomocupPlayer(Player):
    def __init__(self, color, board):
        super(GomocupPlayer, self).__init__(color, utils.GOMOCUP, board)


class HumanPlayer(Player):
    def __init__(self, color, board):
        super(HumanPlayer, self).__init__(color, utils.HUMAN, board)


class RandomPlayer(Player):
    def __init__(self, color, board):
        super(RandomPlayer, self).__init__(color, utils.RANDOM, board)

    def get_move(self):
        index = np.random.choice(self.board.empty_pos)
        return self.board.index2xy(index)


class MCTSPlayer(Player):
    def __init__(self, color, board):
        super(MCTSPlayer, self).__init__(color, utils.MCTS, board)
        self.prob_history = list()
        self.probability = None

    def add_history(self):
        if self.probability is not None:
            self.prob_history.append(self.probability.copy())
        else:
            raise AttributeError('no probability yet')

    def get_move(self):
        raise NotImplementedError()
    
    def undo(self, x, y):
        super(MCTSPlayer, self).undo()
        self.probability = self.prob_history.pop()


def player_generate(player_type, color, board):
    PLAYER_DICT = {
        utils.HUMAN: HumanPlayer,
        utils.GOMOCUP: GomocupPlayer,
        utils.RANDOM: RandomPlayer,
        utils.MCTS: MCTSPlayer
    }

    if player_type not in PLAYER_DICT:
        raise AttributeError('no such player type')

    return PLAYER_DICT[player_type](color, board)
