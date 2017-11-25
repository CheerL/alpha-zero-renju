# -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import os
import time
import utils
from utils.logger import Logger
from utils.database import LMDB
from player_single import HumanPlayer, RandomPlayer, GomocupPlayer, MCTSPlayer
from board_single import Board
from collections import namedtuple

History = namedtuple('History', ['board', 'x', 'y'])

class Game(object):
    RESULT = {
        utils.BLACK: 'Black win',
        utils.WHITE: 'White win',
        utils.EMPTY: 'end in a draw'
    }
    PLAYER = {
        utils.HUMAN: HumanPlayer,
        utils.GOMOCUP: GomocupPlayer,
        utils.MCTS: MCTSPlayer,
        utils.RANDOM: RandomPlayer
    }
    logger = Logger('game', handlers=['File'])

    def __init__(self, black_player_type=utils.RANDOM,
                 white_player_type=utils.RANDOM, size=utils.SIZE):
        self.board = Board(size)
        self.players = {
            utils.BLACK: self.PLAYER[black_player_type](utils.BLACK, self.board),
            utils.WHITE: self.PLAYER[white_player_type](utils.WHITE, self.board)
        }
        self.size = size
        self.full_size = size ** 2        
        self.round_num = 0
        self.player_color = utils.BLACK
        self.winner = utils.EMPTY
        self.run = True
        self.history = list()

        self.add_history(None, None)
        self.logger.info(
            'Start new game. Board size: %d * %d, Black: %s, White: %s'
            % (size, size, type(self.black_player).__name__, type(self.white_player).__name__)
            )

    @property
    def black_player(self):
        return self.players[utils.BLACK]

    @property
    def white_player(self):
        return self.players[utils.WHITE]

    @property
    def now_player(self):
        return self.players[self.player_color]

    @property
    def last_player(self):
        return self.players[-self.player_color]

    def add_history(self, x, y):
        history = History(self.board.board.copy, x, y)
        self.history.append(history)

    def round_back(self):
        if self.round_num > 0:
            history = self.history.pop()
            self.now_player.undo(history.x, history.y)
            self.board.set_board(history.board)
            self.round_num -= 1
            self.player_color *= -1
            self.logger.info('Round back to {}'.format(self.round_num))
        else:
            self.logger.warn('That is the first round')

    def round_process(self, move=None):
        now_player = self.now_player

        if not move:
            if now_player.player_type is not utils.GOMOCUP:
                move = now_player.get_move()
            else:
                msg = 'gomocup player does not get move'
                self.logger.error(msg)
                raise AttributeError(msg)

        now_player.move(*move)
        self.add_history(*move)
        self.logger.info('Round {}, {}: ({}:{})'.format(
            self.round_num, utils.COLOR[self.player_color], *move
        ))

        if now_player.judge_win(*move):
            self.winner = self.player_color
            self.game_over()
            # 输出结果
        elif self.round_num is 100:
        # elif self.round_num is self.full_size - 1:
            self.game_over()
        else:
            self.player_color *= -1
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
        self.run = False
        self.logger.info('Game over, {}'.format(self.RESULT[self.winner]))
        self.save_record()

    def show(self, board):
        show_board = board.astype(object)
        show_board[show_board == utils.BLACK] = '㊣'
        show_board[show_board == utils.WHITE] = '〇'
        show_board[show_board == utils.EMPTY] = '　'
        # show_board[show_board == utils.BLACK] = '★'
        # show_board[show_board == utils.WHITE] = '☆'
        # show_board[show_board == utils.EMPTY] = '　'

        for line in show_board:
            print(''.join(line))

    def save_record(self):
        time_suffix = time.strftime('%Y%m%d-%H%M%S',time.localtime())
        record_filename = 'record-{}.psq'.format(time_suffix)
        record_path = os.path.join(utils.RECORD_PATH, record_filename)
        while os.path.exists(record_path):
            record_filename = '_' + record_filename
            record_path = os.path.join(utils.RECORD_PATH, record_filename)

        with open(record_path, 'w+') as file:
            file.write('Piskvorky {}x{}, 0:0, 1\n'.format(self.size, self.size))

            for history in self.history[1:]:
                file.write('{},{},0\n'.format(history.x + 1, history.y + 1))

            file.write('-1\n')

        self.logger.info('Save record to {}'.format(record_filename))

def main():
    game = Game()
    game.start()

if __name__ == '__main__':
    main()
