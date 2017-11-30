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
        self.board = Board(size)
        self.players = {
            utils.BLACK: player_generate(black_player_type, utils.BLACK, self),
            utils.WHITE: player_generate(white_player_type, utils.WHITE, self)
        }
        self.size = size
        self.full_size = self.board.full_size
        self.round_num = 0
        self.winner = utils.EMPTY
        self.run = True
        self.history = list()

        self.logger.info(
            'Start new game. Board size: %d * %d, Black: %s, White: %s'
            % (size, size, type(self.black_player).__name__, type(self.white_player).__name__)
            )

    @property
    def now_player_color(self):
        return self.board.now_color

    @property
    def black_player(self):
        return self.players[utils.BLACK]

    @property
    def white_player(self):
        return self.players[utils.WHITE]

    @property
    def now_player(self):
        return self.players[self.now_player_color]

    @property
    def last_player(self):
        return self.players[-self.now_player_color]

    def add_history(self, move):
        self.history.append(move)

    def round_back(self):
        if self.round_num > 0:
            last_move = self.history.pop()
            self.now_player.undo(last_move)
            self.round_num -= 1
            self.logger.info('Round back to {}'.format(self.round_num))
            del last_move
        else:
            self.logger.warning('That is the first round')

    def round_process(self, move=None):
        self.logger.info('Round {}'.format(self.round_num))
        if not move:
            if self.now_player.player_type is not utils.GOMOCUP:
                move = self.now_player.get_move()
            else:
                raise AttributeError('gomocup player does not get move')

        self.now_player.move(move)
        self.add_history(move)

        if self.now_player.judge_win(move):
            self.winner = self.now_player_color
            self.game_over()
            # 输出结果
        elif self.round_num is self.full_size - 1:
        # elif self.round_num is 100:
            self.game_over()
        else:
            self.round_num += 1
        return move

    def start(self):
        while self.run:
            self.round_process()

    def restart(self, black_player_type=None, white_player_type=None, size=None):
        if not black_player_type:
            black_player_type = self.black_player.player_type
        if not white_player_type:
            white_player_type = self.white_player.player_type
        if not size:
            size = self.size

        self.__init__(black_player_type, white_player_type, size)
        self.start()

    def game_over(self):
        self.logger.info('Game over')
        if self.winner is utils.EMPTY:
            self.logger.info('End in a draw')
        else:
            self.now_player.win()

        self.save_record()
        self.run = False

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
            file.write('Piskvorky {}x{}, 0:0, 1\n'.format(self.size, self.size))

            for move in self.history:
                file.write('{},{},0\n'.format(move.x + 1, move.y + 1))

            file.write('-1\n')

        self.logger.info('Save record to {}'.format(record_path))


def main():
    for _ in range(100):
        game = Game()
        game.start()

if __name__ == '__main__':
    main()
