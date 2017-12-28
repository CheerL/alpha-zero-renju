# -*- coding:utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

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
        self.size = game.board.size
        self.logger.info('Create new {} {}'.format(self.color_str, type(self).__name__))

    def move(self, index):
        '''在`(x, y)`处落子'''
        self.game.board.move(index)
        self.logger.info('Round {}, {} move ({}:{})'.format(
            self.game.board.round_num, self.color_str, index % self.size, index // self.size
            ))

    def undo(self, index):
        self.game.board.undo()
        self.logger.info('Round back to {}, {} undo ({},{})'.format(
            self.game.board.round_num, self.color_str, index % self.size, index // self.size
            ))

    def win(self):
        self.logger.info('{} win'.format(self.color_str))

    def lose(self):
        pass

    def reset(self):
        pass

    def get_model_num(self):
        raise NotImplementedError()

    @property
    def show_board(self):
        '''以`size * size`的矩阵形式输出'''
        return self.game.board.get_show_board(self.color)


class GomocupPlayer(Player):
    def __init__(self, color, game, *args):
        super(GomocupPlayer, self).__init__(color, utils.GOMOCUP, game)


class HumanPlayer(Player):
    def __init__(self, color, game, *args):
        super(HumanPlayer, self).__init__(color, utils.HUMAN, game)


class RandomPlayer(Player):
    def __init__(self, color, game, *args):
        super(RandomPlayer, self).__init__(color, utils.RANDOM, game)

    def get_move(self):
        index = np.random.choice(self.game.board.empty_pos)
        return index


class MCTSPlayer(Player):
    def __init__(self, color, game, model_num):
        super(MCTSPlayer, self).__init__(color, utils.MCTS, game)
        self.prob_history = list()
        self.probability = None
        self.mct = MCT(self.game.board, model_num)

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
                self.mct.update(self.game.history[-2], self.game.history[-1])

        self.mct.play()
        self.probability = self.mct.get_move_probability()
        self.add_history()
        return self.mct.get_move(self.probability)

    def undo(self, index):
        super(MCTSPlayer, self).undo(index)
        self.probability = self.prob_history.pop()
        if self.mct.root.parent and self.mct.root.parent.parent:
            self.mct.root = self.mct.root.parent.parent
        else:
            self.mct.root = MCTNode(None, 1.0)

    def win(self):
        super(MCTSPlayer, self).win()
        self.save_history_to_tfrecord(1)

    def lose(self):
        self.save_history_to_tfrecord(-1)

    def get_model_num(self):
        return self.mct.net.get_model_num()

    def save_history_to_tfrecord(self, reward):
        if utils.SAVE_RECORD:
            net_model_num = self.mct.net.get_model_num()
            verification_pattern = os.path.join(
                utils.PAI_DB_PATH if utils.USE_PAI else utils.DB_PATH,
                'game-verification-{}*'.format(net_model_num)
                )
            if len(utils.pai_find_path(verification_pattern)) < 2 * utils.VERIFICATION_GAME_NUM:
                tfr_name = 'game-verification-{}-{}-{}.tfrecord'.format(net_model_num, time.time(), self.color_str)
            else:
                tfr_name = 'game-{}-{}-{}.tfrecord'.format(net_model_num, time.time(), self.color_str)
            tfr_path = os.path.join(utils.PAI_DB_PATH if utils.USE_PAI else utils.DB_PATH, tfr_name)
            tfr_writer = generate_writer(tfr_path)

            for i, expect in enumerate(self.prob_history):
                feature = self.game.board.get_feature(self.color, i)
                example = generate_example(feature, expect, reward)
                tfr_writer.write(example.SerializeToString())

            # for i, base_expect in enumerate(self.prob_history):
            #     base_feature = self.game.board.get_feature(self.color, i)
            #     base_expect = np.reshape(base_expect, (self.size, self.size))
            #     for rot_num in range(8):
            #         rot = self.mct.net.rot[rot_num]
            #         feature = rot(base_feature, (1, 2))
            #         expect = rot(base_expect).reshape(self.game.board.full_size)
            #         example = generate_example(feature, expect, reward)
            #         tfr_writer.write(example.SerializeToString())

            tfr_writer.close()

    def reset(self):
        self.prob_history = list()
        self.probability = None
        self.mct.reset()


class TransPlayer(Player):
    def __init__(self, color, game, *args):
        super(TransPlayer, self).__init__(color, utils.TRANS, game)
        self.prob_history = list()
        self.probability = None
        self.record = None

    def add_history(self):
        if self.probability is not None:
            self.prob_history.append(self.probability.copy())
        else:
            raise AttributeError('no probability yet')

    def add_record(self, path):
        self.logger.info('Add {}'.format(path))
        with utils.pai_open(path, 'r') as file:
            self.record = iter(file.readlines()[1:-3])

        if self.color is utils.WHITE:
            next(self.record)

    def get_move(self):
        try:
            move = next(self.record)
            try:
                next(self.record)
            except:
                pass

            x, y, _ = map(lambda x: int(x) - 1, move.split(','))
            index = x + y * self.size
            self.logger.info('{} {} {}'.format(x, y, index))
            self.probability = np.zeros(self.game.board.full_size, np.float32)
            self.probability[index] = 1.0
            self.add_history()
            return index
        except Exception as e:
            self.logger.error(e)
            self.logger.error('Come to the end of the record')

    def undo(self, index):
        super(TransPlayer, self).undo(index)
        self.probability = self.prob_history.pop()

    def win(self):
        super(TransPlayer, self).win()
        self.save_history_to_tfrecord(1)

    def lose(self):
        self.save_history_to_tfrecord(-1)

    def save_history_to_tfrecord(self, reward):
        if utils.SAVE_RECORD:
            net_model_num = 0
            verification_pattern = os.path.join(
                utils.PAI_DB_PATH if utils.USE_PAI else utils.DB_PATH,
                'game-verification-{}*'.format(net_model_num)
                )
            if len(utils.pai_find_path(verification_pattern)) < 2 * utils.VERIFICATION_GAME_NUM:
                tfr_name = 'game-verification-{}-{}-{}.tfrecord'.format(net_model_num, time.time(), self.color_str)
            else:
                tfr_name = 'game-{}-{}-{}.tfrecord'.format(net_model_num, time.time(), self.color_str)
            tfr_path = os.path.join(utils.PAI_DB_PATH if utils.USE_PAI else utils.DB_PATH, tfr_name)
            tfr_writer = generate_writer(tfr_path)

            for i, expect in enumerate(self.prob_history):
                feature = self.game.board.get_feature(self.color, i)
                example = generate_example(feature, expect, reward)
                tfr_writer.write(example.SerializeToString())

            tfr_writer.close()

    def reset(self):
        self.prob_history = list()
        self.probability = None


def player_generate(player_type, color, game, model_num=None):
    PLAYER_DICT = {
        utils.HUMAN: HumanPlayer,
        utils.GOMOCUP: GomocupPlayer,
        utils.RANDOM: RandomPlayer,
        utils.MCTS: MCTSPlayer,
        utils.TRANS: TransPlayer
    }

    if player_type not in PLAYER_DICT:
        raise AttributeError('no such player type')

    return PLAYER_DICT[player_type](color, game, model_num)
