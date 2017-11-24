# -*- coding:utf-8 -*-
from __future__ import division

import numpy as np
import utils

class Board(object):
    WIN_PATTERN = np.ones(utils.WIN_NUM, np.int)

    def __init__(self, color, size=utils.SIZE):
        self.color = color
        self.size = size
        self.full_size = self.size ** 2
        self.board = np.zeros(self.full_size, np.int)
        # self.matrix_trans = self.generate_matrix_trans()

    def __getitem__(self, index):
        return self.board[index]

    def __setitem__(self, index, val):
        self.board[index] = val

    def xy2index(self, x, y):
        '''将坐标`(x, y)`变为序号`index`'''
        return x + y * self.size

    def index2xy(self, index):
        '''将序号`index`变为坐标`(x, y)`'''
        return index % self.size, index // self.size

    def row(self, y):
        '''获取第`y`行'''
        return self.board[y * self.size: (y + 1) * self.size]

    def col(self, x):
        '''获取第`x`列'''
        return self.board[x::self.size]

    def diag(self, x, y):
        '''获取`(x, y)`处对角线'''
        if x >= y:
            return self.board[x - y:(self.size - (x - y)) * self.size:self.size + 1]
        else:
            return self.board[(y - x) * self.size:self.full_size:self.size + 1]

    def back_diag(self, x, y):
        '''获取`(x, y)`处反对角线'''
        if x + y < 15:
            return self.board[x + y:(x + y) * self.size + 1:self.size - 1]
        else:
            return self.board[(x + y - self.size + 2) * self.size - 1:self.full_size:self.size - 1]

    def judge_win(self, x, y):
        '''检查`(x, y)`周围情况, 判断是否获胜'''
        lines = [self.row(y), self.col(x), self.diag(x, y), self.back_diag(x, y)]
        for line in lines:
            if line.size < utils.WIN_NUM:
                continue
            else:
                if utils.WIN_NUM in np.correlate(line, self.WIN_PATTERN):
                    return True
        return False

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
    white_board = Board(utils.WHITE)
    black_board = Board(utils.BLACK)
