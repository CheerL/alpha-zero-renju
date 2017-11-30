# -*- coding:utf-8 -*-
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import utils

class Board(object):
    '''棋盘类'''
    WIN_PATTERN = np.ones(utils.WIN_NUM, np.int)

    def __init__(self, size=utils.SIZE):
        self.size = size
        self.full_size = self.size ** 2
        self.board = np.zeros(self.full_size, np.int)
        self.board_history_length = utils.BOARD_HISTORY_LENGTH
        self.black_board_history = [np.zeros(self.full_size, dtype=np.int)] * self.board_history_length
        self.white_board_history = [np.zeros(self.full_size, dtype=np.int)] * self.board_history_length

    def __getitem__(self, index):
        return self.board[index]

    def __setitem__(self, index, val):
        self.board[index] = val

    def get_round(self):
        return (self.board != 0).sum()

    def get_color(self):
        if self.get_round() % 2:
            return utils.BLACK
        else:
            return utils.WHITE

    def xy2index(self, move):
        '''将坐标`(x, y)`变为序号`index`'''
        return move.x + move.y * self.size

    def index2xy(self, index):
        '''将序号`index`变为坐标`(x, y)`'''
        return utils.Move(index % self.size, index // self.size)

    def row(self, move):
        '''获取第`y`行'''
        return self.board[move.y * self.size: (move.y + 1) * self.size]

    def col(self, move):
        '''获取第`x`列'''
        return self.board[move.x::self.size]

    def diag(self, move):
        '''获取`(x, y)`处对角线'''
        if move.x >= move.y:
            return self.board[move.x - move.y:(self.size - (move.x - move.y)) * self.size:self.size + 1]
        else:
            return self.board[(move.y - move.x) * self.size:self.full_size:self.size + 1]

    def back_diag(self, move):
        '''获取`(x, y)`处反对角线'''
        if move.x + move.y < 15:
            return self.board[move.x + move.y:(move.x + move.y) * self.size + 1:self.size - 1]
        else:
            return self.board[(move.x + move.y - self.size + 2) * self.size - 1:self.full_size:self.size - 1]

    def judge_win(self, move, color):
        '''检查`(x, y)`周围情况, 判断是否获胜'''
        target = utils.WIN_NUM * color
        lines = [self.row(move), self.col(move), self.diag(move), self.back_diag(move)]
        for line in lines:
            if line.size < utils.WIN_NUM:
                continue
            else:
                if target  in np.correlate(line, self.WIN_PATTERN):
                    return True
        return False

    def move(self, move_or_index, color):
        if color not in [utils.BLACK, utils.WHITE]:
            raise AttributeError('given color is not black or white')

        if isinstance(move_or_index, utils.Move):
            index = self.xy2index(move_or_index)
        elif isinstance(move_or_index, int):
            index = move_or_index
        else:
            raise AttributeError('neither move or index')

        assert 0 <= index <self.full_size, 'index out of range'
        assert self.board[index] == utils.EMPTY, '目标位置已经有子'
        self.board[index] = color

        if color is utils.BLACK:
            self.black_board_history.append(self.black_board)
        else:
            self.white_board_history.append(self.white_board)

    def undo(self, color):
        if color is utils.BLACK:
            if len(self.black_board_history) > self.board_history_length:
                self.board = self.black_board_history.pop() * utils.BLACK + self.white_board_history[-1] * utils.WHITE
        elif color is utils.WHITE:
            if len(self.white_board_history) > self.board_history_length:
                self.board = self.black_board_history[-1] * utils.BLACK + self.white_board_history.pop() * utils.WHITE
        else:
            raise AttributeError('given color is not black or white')


    def get_feature(self, color):
        if color is utils.BLACK:
            return np.array(self.black_board_history[-self.board_history_length:]\
                + self.white_board_history[-self.board_history_length:]\
                + [np.ones(self.full_size)])
        elif color is utils.WHITE:
            return np.array(self.white_board_history[-self.board_history_length:]\
                + self.black_board_history[-self.board_history_length:]\
                + [np.zeros(self.full_size)])
        else:
            raise AttributeError('given color is not black or white')

    def get_color_board(self, color, board=None):
        if board is None:
            board = self.board
        elif not isinstance(board, np.ndarray) or not board.size == self.full_size:
            raise AttributeError('given board is not np.array or does not match the size')

        if color not in [utils.BLACK, utils.WHITE]:
            raise AttributeError('given color is not black or white')

        color_board = np.zeros(self.full_size, np.int)
        color_board[board == color] = 1
        return color_board

    def get_show_board(self, color=None):
        if color:
            show_board = self.get_color_board(color)
            return show_board.reshape(self.size, self.size)
        else:
            return self.board.reshape(self.size, self.size)

    @property
    def show_board(self):
        return self.get_show_board()

    @property
    def black_board(self):
        return self.get_color_board(utils.BLACK)

    @property
    def white_board(self):
        return self.get_color_board(utils.WHITE)

    @property
    def empty_pos(self):
        return np.where(self.board == utils.EMPTY)[0]

    # def generate_matrix_trans(self):
    #     rotat_0 = lambda m: m
    #     rotat_90 = np.rot90
    #     rotat_180 = lambda m: np.rot90(m, 2)
    #     rotat_270 = lambda m: np.rot90(m, 3)
    #     reflect_0 = np.fliplr
    #     reflect_90 = lambda m: np.fliplr(rotat_90(m))
    #     reflect_180 = lambda m: np.fliplr(rotat_180(m))
    #     reflect_270 = lambda m: np.fliplr(rotat_270(m))

    #     return {
    #         0: rotat_0,
    #         1: rotat_90,
    #         2: rotat_180,
    #         3: rotat_270,
    #         4: reflect_0,
    #         5: reflect_90,
    #         6: reflect_180,
    #         7: reflect_270
    #     }


if __name__ == '__main__':
    pass
