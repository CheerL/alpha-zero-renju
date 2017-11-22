import config as cfg
from collections import deque
from board import Board
from player import HumanPlayer, RobotPlayer, GomocupPlayer
from utils.logger import LogHandler

class Game(object):
    RESULT = {
        cfg.BLACK: '黑棋获胜',
        cfg.WHITE: '白棋获胜',
        cfg.EMPTY: '和棋'
    }
    PLAYER = {
        cfg.HUMAN: HumanPlayer,
        cfg.ROBOT: RobotPlayer,
        cfg.GOMOCUP: GomocupPlayer
    }
    logger = LogHandler('game')

    def __init__(self, black_player_type=cfg.ROBOT, white_player_type=cfg.ROBOT, size=cfg.SIZE):
        black_player = self.PLAYER[black_player_type](cfg.BLACK, size)
        white_player = self.PLAYER[white_player_type](cfg.WHITE, size)
        black_player.oppo_board = white_player.board.board
        white_player.oppo_board = black_player.board.board
        self.size = size
        self.players = {
            cfg.BLACK: black_player,
            cfg.WHITE: white_player
        }
        self.round_num = 0
        self.player_color = cfg.BLACK
        self.winner = cfg.EMPTY
        self.run = True
        self.board_history = deque([white_player.board.board, black_player.board.board])
        self.logger.info(
            'Start new game. Board size: %d * %d, Black: %s, White: %s'
            % (size, size, type(black_player).__name__, type(white_player).__name__)
            )

    def round_process(self, move=None):
        player = self.players[self.player_color]
        if player.player_type is not cfg.GOMOCUP:
            move = player.get_move()
        else:
            if not move:
                msg = 'gomocup player没有获得落子点'
                self.logger.error(msg)
                raise NotImplementedError(msg)

        player.move(*move)
        self.board_history.appendleft(player.board.board)
        self.logger.info('Round %d, %s: (%d:%d)' % (
            self.round_num,
            'Black' if self.player_color is cfg.BLACK else 'White',
            *move
        ))
        if player.board.judge_win(*move):
            self.winner = self.player_color
            self.game_over()
            # 输出结果
        elif self.round_num is player.board.full_size - 1:
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
            black_player_type = self.players[cfg.BLACK].player_type
        if not white_player_type:
            white_player_type = self.players[cfg.WHITE].player_type
        if not size:
            size = self.size
        self.__init__(black_player_type, white_player_type, size)
        self.start()

    def game_over(self):
        self.run = False
        msg = '游戏结束, {}'.format(self.RESULT[self.winner])
        self.logger.info(msg)

    def show(self):
        print(self.players[cfg.BLACK].board.show() + self.players[cfg.WHITE].board.show())

if __name__ == '__main__':
    game = Game()
