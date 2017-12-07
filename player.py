# -*- coding:utf-8 -*-
from __future__ import unicode_literals

import os
import time
import numpy as np
import utils
from utils.tfrecord import generate_example, generate_writer
from utils.logger import Logger
from functools import partial
from mcts import MCT, MCTNode
# from net import write_db

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

    def reset(self):
        pass

    @property
    def show_board(self):
        '''以`size * size`的矩阵形式输出'''
        return self.game.board.get_show_board(self.color)


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
        self.mct = MCT(self.game.board)

    def add_history(self):
        if self.probability is not None:
            self.prob_history.append(self.probability.copy())
        else:
            raise AttributeError('no probability yet')

    def get_move(self):
        if self.game.history:
            if len(self.game.history) is 1:
                self.mct.update_one(self.game.history[0])
            else:
                self.mct.update(*self.game.history[-2:])

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

        tfr_name = 'game{}.tfrecord'.format(time.time())
        # tfr_path = os.path.join(utils.DB_PATH, tfr_name)
        if utils.USE_PAI:
            tfr_path = os.path.join(utils.PAI_DB_PATH, tfr_name)
        else:
            tfr_path = os.path.join(utils.DB_PATH, tfr_name)
        tfr_writer = generate_writer(tfr_path)

        for i, expect in enumerate(self.prob_history):
            feature = self.game.board.get_feature(self.color, i)
            reward = 1
            example = generate_example(feature, expect, reward)
            tfr_writer.write(example.SerializeToString())

        oppo_player = self.game.black_player if self.color is utils.WHITE else self.game.white_player
        if oppo_player.player_type is utils.MCTS:
            for i, expect in enumerate(oppo_player.prob_history):
                feature = oppo_player.game.board.get_feature(oppo_player.color, i)
                reward = -1
                example = generate_example(feature, expect, reward)
                tfr_writer.write(example.SerializeToString())

        tfr_writer.close()

    def reset(self):
        self.prob_history = list()
        self.probability = None
        self.mct.reset()


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
