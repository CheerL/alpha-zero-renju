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

    def add_history(self, move):
        self.history.append(move)

    def round_back(self):
        if self.board.round_num > 0:
            last_move = self.history.pop()
            self.now_player.undo(last_move)
            del last_move
        else:
            self.logger.warning('That is the first round')

    def round_process(self, move=None):
        if not move:
            if self.now_player.player_type is not utils.GOMOCUP:
                move = self.now_player.get_move()
            else:
                raise AttributeError('gomocup player does not get move')

        self.now_player.move(move)
        self.add_history(move)

        if self.board.judge_win(move):
            self.game_over()
        elif self.board.judge_round_up():
            self.game_over()
        else:
            self.board.round_change(1)
        return move

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

            for move in self.history:
                file.write('{},{},0\n'.format(move.x + 1, move.y + 1))

            file.write('-1\n')

        self.logger.info('Save record to {}'.format(record_path))

        if utils.USE_PAI:
            record_filename = record_path.split('/')[-1]
            pai_record_path = os.path.join(utils.PAI_RECORD_PATH, record_filename)
            utils.pai_copy(record_path, pai_record_path)

def main():
    game = Game(utils.MCTS, utils.MCTS)
    if utils.USE_PAI:
        db_pattern = os.path.join(utils.PAI_DB_PATH, '*')
        while len(utils.pai_find_path(db_pattern)) < utils.TRAIN_EPOCH_GAME_NUM:
            game.logger.info('There are {} records now'.format(len(utils.pai_find_path(db_pattern))))
            game.start()
            game.reset()
    else:
        game.start()

if __name__ == '__main__':
    main()
