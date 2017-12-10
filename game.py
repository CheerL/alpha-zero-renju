# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import time
import utils
from utils.logger import Logger
from player import player_generate
from board import Board

import numpy as np


class Game(object):
    logger = Logger('game')
    # logger.remove_stream_handler()

    def __init__(self, black_player_type=utils.RANDOM,
                 white_player_type=utils.RANDOM, size=utils.SIZE):
        self.logger.info('Start new game. Board size: {} * {}'.format(size, size))
        self.board = Board(size)
        self.black_player = player_generate(black_player_type, utils.BLACK, self)
        self.white_player = player_generate(white_player_type, utils.WHITE, self)
        self.run = True
        self.history = list()

    @property
    def now_player_color(self):
        return self.board.now_color

    @property
    def now_player(self):
        if self.now_player_color is utils.BLACK:
            return self.black_player
        else:
            return self.white_player

    @property
    def last_player(self):
        if self.now_player_color is utils.BLACK:
            return self.white_player
        else:
            return self.black_player

    def add_history(self, index):
        self.history.append(index)

    def round_back(self):
        if self.board.round_num > 0:
            last_index = self.history.pop()
            self.now_player.undo(last_index)
            del last_index
        else:
            self.logger.warning('That is the first round')

    def round_process(self, index=None):
        if not index:
            if self.now_player.player_type is not utils.GOMOCUP:
                index = self.now_player.get_move()
            else:
                raise AttributeError('gomocup player does not get move')

        self.now_player.move(index)
        self.add_history(index)

        if self.board.judge_win(index):
            self.game_over()
        elif self.board.judge_round_up():
            self.game_over()
        else:
            self.board.round_change(1)
        return index

    def start(self):
        while self.run:
            self.round_process()

    def reset(self, black_player_type=None, white_player_type=None):
        self.run = True
        self.history = list()

        self.board.reset()
        if black_player_type and black_player_type != self.black_player.player_type:
            self.black_player = player_generate(black_player_type, utils.BLACK, self)
        else:
            self.black_player.reset()

        if white_player_type and white_player_type != self.white_player.player_type:
            self.white_player = player_generate(white_player_type, utils.BLACK, self)
        else:
            self.white_player.reset()

        utils.CLEAR()
        self.logger.info('Reset game')

    def game_over(self):
        self.logger.info('Game over')
        self.run = False

        if self.board.winner is utils.EMPTY:
            self.logger.info('End in a draw')
        else:
            self.now_player.win()
            self.last_player.lose()
            if utils.SAVE_PSQ:
                self.save_record()

    def show(self, board):
        show_board = board.astype(object)
        show_board[show_board == utils.BLACK] = '㊣'
        show_board[show_board == utils.WHITE] = '〇'
        show_board[show_board == utils.EMPTY] = '　'

        for line in show_board:
            print(''.join(line))

    def save_record(self):
        def get_path_from_format(formater, suffix=''):
            path = formater.format(suffix)
            if os.path.exists(path):
                return get_path_from_format(formater, suffix + '_')
            else:
                return path

        time_suffix = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        record_filename = 'record-{}.psq'.format(time_suffix + '{}')
        record_path_format = os.path.join(utils.RECORD_PATH, record_filename)
        record_path = get_path_from_format(record_path_format)

        with open(record_path, 'w+') as file:
            file.write('Piskvorky {}x{}, 0:0, 1\n'.format(self.board.size, self.board.size))

            for index in self.history:
                x, y = index % self.board.size, index // self.board.size
                file.write('{},{},0\n'.format(x + 1, y + 1))

            file.write('-1\n')

        self.logger.info('Save record to {}'.format(record_path))

        if utils.USE_PAI:
            record_filename = record_path.split('/')[-1]
            pai_record_path = os.path.join(utils.PAI_RECORD_PATH, record_filename)
            utils.pai_copy(record_path, pai_record_path)

def main():
    if utils.USE_PAI:
        game = Game(utils.MCTS, utils.MCTS)
        model_num = game.black_player.mct.net.get_model_num()
        db_pattern = os.path.join(utils.PAI_DB_PATH, 'game-{}-*'.format(model_num))
        while len(utils.pai_find_path(db_pattern)) / 2 < utils.TRAIN_EPOCH_GAME_NUM:
            game.logger.info('There are {} records now'.format(len(utils.pai_find_path(db_pattern))/ 2))
            game.start()
            game.reset()
    else:
        game = Game()
        game.start()

def compare(compare_model_num):
    if utils.USE_PAI:
        game = Game(utils.MCTS, utils.MCTS)
        best_model_num = game.black_player.mct.net.get_model_num()
        if compare_model_num != best_model_num:
            utils.pai_model_copy(compare_model_num)

        while True:
            win, total = utils.pai_read_compare_record(best_model_num, compare_model_num)
            game.logger.info('Now compare result: {}-{}'.format(win, total))
            if total > utils.COMPARE_TIME:
                break

            black_as_best = np.random.choice([True, False])
            if black_as_best:
                game.logger.info('Black as best')
                game.black_player.mct.reset_net(best_model_num)
                game.white_player.mct.reset_net(compare_model_num)
            else:
                game.logger.info('White as best')
                game.black_player.mct.reset_net(compare_model_num)
                game.white_player.mct.reset_net(best_model_num)

            game.start()
            winner = game.board.winner
            if winner is utils.EMPTY:
                pass
            elif (winner is utils.BLACK and black_as_best) or (winner is utils.WHITE and not black_as_best):
                utils.pai_write_compare_record(best_model_num, compare_model_num, False)
            else:
                utils.pai_write_compare_record(best_model_num, compare_model_num, True)
            game.reset()

        if win / total > utils.COMPARE_WIN_RATE:
            utils.pai_change_best(compare_model_num)
            game.logger.info('Change best model to {}'.format(compare_model_num))
        else:
            game.logger.info('Best model does not change')


if __name__ == '__main__':
    main()
