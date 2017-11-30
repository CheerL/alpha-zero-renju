# -*- coding:utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import utils
from utils.logger import Logger
from functools import partial
from mcts import MCT

class Player(object):
    logger = Logger('game')

    def __init__(self, color, player_type, game):
        self.color = color
        self.color_str = utils.COLOR[color]
        self.player_type = player_type
        self.game = game
        self.board = game.board
        self.logger.info('Create new {} player'.format(self.color_str))

    def move(self, move):
        '''在`(x, y)`处落子'''
        self.board.move(move)
        self.logger.info('{}: ({}:{})'.format(self.color_str, move.x, move.y))

    def undo(self, move):
        self.board.undo()
        self.logger.info('{} undo: ({},{})'.format(self.color_str, move.x, move.y))

    def win(self):
        self.logger.info('{} win'.format(self.color_str))

    def judge_win(self, move):
        return self.board.judge_win(move)

    @property
    def show_board(self):
        '''以`size * size`的矩阵形式输出'''
        return self.board.get_show_board(self.color)

    # def get_board_history(self, round_num=None):
    #     if not round_num:
    #         round_num = self.game.round_num

    #     if round_num < self.board_history_length:
    #         raw_board_history = [np.zeros(self.board.full_size, dtype=np.int)] * (self.board_history_length - round_num - 1)\
    #             + [history.board for history in self.game.history[:round_num + 1]]
    #     else:
    #         raw_board_history = [history.board for history in self.game.history[round_num - self.board_history_length + 1:round_num + 1]]

    #     get_seperate_board = partial(self.board.get_seperate_board, self.color)
    #     return np.concatenate(list(map(get_seperate_board, raw_board_history)))


class GomocupPlayer(Player):
    def __init__(self, color, game):
        super(GomocupPlayer, self).__init__(color, utils.GOMOCUP, game)


class HumanPlayer(Player):
    def __init__(self, color, game):
        super(HumanPlayer, self).__init__(color, utils.HUMAN, game)


class RandomPlayer(Player):
    def __init__(self, color, game):
        super(RandomPlayer, self).__init__(color, utils.RANDOM, game)

    def get_move(self):
        index = np.random.choice(self.board.empty_pos)
        return self.board.index2xy(index)


class MCTSPlayer(Player):
    def __init__(self, color, game):
        super(MCTSPlayer, self).__init__(color, utils.MCTS, game)
        self.prob_history = list()
        self.probability = None
        self.mct = MCT(self.board)

    def add_history(self):
        if self.probability is not None:
            self.prob_history.append(self.probability.copy())
        else:
            raise AttributeError('no probability yet')

    def get_move(self):
        self.mct.play()
        self.probability = self.mct.get_move_probability()
        self.add_history()
        return self.board.index2xy(self.mct.get_move(self.probability))

    def undo(self, move):
        super(MCTSPlayer, self).undo(move)
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
