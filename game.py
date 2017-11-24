import os
import time
import config as cfg
from player import HumanPlayer, RandomPlayer, GomocupPlayer, MCTSPlayer
from utils import RECORD_PATH
from utils.logger import Logger

class Game(object):
    RESULT = {
        cfg.BLACK: 'Black win',
        cfg.WHITE: 'White win',
        cfg.EMPTY: 'end in a draw'
    }
    PLAYER = {
        cfg.HUMAN: HumanPlayer,
        cfg.GOMOCUP: GomocupPlayer,
        cfg.MCTS: MCTSPlayer,
        cfg.RANDOM: RandomPlayer
    }
    logger = Logger('game')

    def __init__(self, black_player_type=cfg.RANDOM, white_player_type=cfg.RANDOM, size=cfg.SIZE):
        black_player = self.PLAYER[black_player_type](cfg.BLACK, size)
        white_player = self.PLAYER[white_player_type](cfg.WHITE, size)
        black_player.oppo_board = white_player.board
        white_player.oppo_board = black_player.board
        self.size = size
        self.players = {
            cfg.BLACK: black_player,
            cfg.WHITE: white_player
        }
        self.round_num = 0
        self.player_color = cfg.BLACK
        self.winner = cfg.EMPTY
        self.run = True
        self.logger.info(
            'Start new game. Board size: %d * %d, Black: %s, White: %s'
            % (size, size, type(black_player).__name__, type(white_player).__name__)
            )

    @property
    def black_player(self):
        return self.players[cfg.BLACK]

    @property
    def white_player(self):
        return self.players[cfg.WHITE]

    @property
    def now_player(self):
        return self.players[self.player_color]

    @property
    def last_player(self):
        return self.players[-self.player_color]

    def round_back(self):
        undo = self.now_player.undo()
        oppo_undo = self.last_player.undo()
        self.round_num -= 2
        self.logger.info('Round back to {}'.format(self.round_num))
        return undo, oppo_undo

    def round_process(self, move=None):
        now_player = self.now_player
        if not move:
            if now_player.player_type is not cfg.GOMOCUP:
                move = now_player.get_move()
            else:
                msg = 'gomocup player does not get move'
                self.logger.error(msg)
                raise NotImplementedError(msg)

        now_player.move(*move)
        self.logger.info('Round {}, {}: ({}:{})'.format(
            self.round_num, cfg.COLOR[self.player_color], *move
        ))
        if now_player.board.judge_win(*move):
            self.winner = self.player_color
            self.game_over()
            # 输出结果
        elif self.round_num is now_player.board.full_size - 1:
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

    @property
    def show_board(self):
        return self.black_player.show_board + self.white_player.show_board

    def show(self, board):
        show_board = board.astype(object)
        show_board[show_board == cfg.BLACK] = '㊣'
        show_board[show_board == cfg.WHITE] = '〇'
        show_board[show_board == cfg.EMPTY] = '　'
        # show_board[show_board == cfg.BLACK] = '★'
        # show_board[show_board == cfg.WHITE] = '☆'
        # show_board[show_board == cfg.EMPTY] = '　'

        for line in show_board:
            print(''.join(line))

    def save_record(self):
        time_suffix = time.strftime('%Y%m%d-%H%M%S',time.localtime())
        record_filename = 'record-{}.psq'.format(time_suffix)
        record_path = os.path.join(RECORD_PATH, record_filename)

        with open(record_path, 'w+') as file:
            file.write('Piskvorky {}x{}, 0:0, 1\n'.format(self.size, self.size))
            while True:
                try:
                    history = self.black_player.history.pop()
                    file.write('{},{},0\n'.format(history.x + 1, history.y + 1))
                    history = self.white_player.history.pop()
                    file.write('{},{},0\n'.format(history.x + 1, history.y + 1))
                except IndexError:
                    break
            file.write('-1\n')

        self.logger.info('Save record to {}'.format(record_filename))

if __name__ == '__main__':
    from game import Game
    import config as cfg
    g = Game(cfg.RANDOM, cfg.RANDOM, cfg.SIZE)
