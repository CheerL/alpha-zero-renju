# -*- coding:utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import utils
from utils.logger import Logger
from functools import partial
from mcts import MCT, MCTNode
from net import write_db

class Player(object):
    logger = Logger('game')

    def __init__(self, color, player_type, game):
        self.color = color
        self.color_str = utils.COLOR[color]
        self.player_type = player_type
        self.game = game
        self.logger.info('Create new {} {}'.format(self.color_str, type(self).__name__))

    def move(self, move):
        '''在`(x, y)`处落子'''
        self.game.board.move(move)
        self.logger.info('Round {}, {} move ({}:{})'.format(
            self.game.board.round_num, self.color_str, move.x, move.y
            ))

    def undo(self, move):
        self.game.board.undo()
        self.logger.info('Round back to {}, {} undo ({},{})'.format(
            self.game.board.round_num, self.color_str, move.x, move.y
            ))

    def win(self):
        self.logger.info('{} win'.format(self.color_str))

    @property
    def show_board(self):
        '''以`size * size`的矩阵形式输出'''
        return self.game.board.get_show_board(self.color)

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
        index = np.random.choice(self.game.board.empty_pos)
        return self.game.board.index2xy(index)


class MCTSPlayer(Player):
    def __init__(self, color, game):
        super(MCTSPlayer, self).__init__(color, utils.MCTS, game)
        self.prob_history = list()
        self.probability = None
        self.is_only_mctsplayer = True
        
        if color is utils.WHITE and isinstance(self.game.black_player, MCTSPlayer):
            self.mct = self.game.black_player.mct
            self.game.black_player.is_only_mctsplayer = False
            self.is_only_mctsplayer = False
        else:
            self.mct = MCT(self.game.board)

    def add_history(self):
        if self.probability is not None:
            self.prob_history.append(self.probability.copy())
        else:
            raise AttributeError('no probability yet')

    def get_move(self):
        if self.game.history:
            if self.is_only_mctsplayer:
                self.mct.update(*self.game.history[-2:])
            else:
                self.mct.update_one(self.game.history[-1])

        self.mct.play()
        self.probability = self.mct.get_move_probability()
        self.add_history()
        return self.game.board.index2xy(self.mct.get_move(self.probability))

    def undo(self, move):
        super(MCTSPlayer, self).undo(move)
        self.probability = self.prob_history.pop()
        if self.mct.root.parent and self.mct.root.parent.parent:
            self.mct.root = self.mct.root.parent.parent
        else:
            self.mct.root = MCTNode(None, 1.0)

    def win(self):
        super(MCTSPlayer, self).win()
        history_length = len(self.prob_history)

        feature = np.concatenate(
            [self.game.board.get_feature(self.color, i) for i in range(history_length)],
            )
        expect = np.array(self.prob_history, dtype=np.float32)
        reward = np.ones(history_length, dtype=np.float32)

        write_db('minidb', 'train_1.minidb', feature, expect, reward)
        return feature, expect, reward



def player_generate(player_type, color, game):
    PLAYER_DICT = {
        utils.HUMAN: HumanPlayer,
        utils.GOMOCUP: GomocupPlayer,
        utils.RANDOM: RandomPlayer,
        utils.MCTS: MCTSPlayer
    }

    if player_type not in PLAYER_DICT:
        raise AttributeError('no such player type')

    return PLAYER_DICT[player_type](color, game)
