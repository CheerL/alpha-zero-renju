import config as cfg
from board import Board
from player import HumanPlayer, RobotPlayer, GomocupPlayer

class Game(object):
    RESULT = {
        cfg.BLACK: '黑棋获胜',
        cfg.WHITE: '白棋获胜',
        cfg.EMPTY: '和棋'
    }
    def __init__(self, black_player_type=cfg.ROBOT, white_player_type=cfg.ROBOT, size=cfg.SIZE):
        black_player = self.generate_player(cfg.BLACK, black_player_type, size)
        white_player = self.generate_player(cfg.WHITE, white_player_type, size)
        black_player.oppo_board = white_player.board.board
        white_player.oppo_board = black_player.board.board

        self.players = {
            cfg.BLACK: black_player,
            cfg.WHITE: white_player
        }
        self.round_num = 0
        self.player_color = cfg.BLACK
        self.winner = cfg.EMPTY
        self.run = True

    def generate_player(self, color, player_type, size):
        if player_type is cfg.HUMAN:
            return HumanPlayer(color, size)
        elif player_type is cfg.ROBOT:
            return RobotPlayer(color, size)
        elif player_type is cfg.GOMOCUP:
            return GomocupPlayer(color, size)

    def round_process(self, move=None):
        player = self.players[self.player_color]
        if player.player_type is not cfg.GOMOCUP:
            move = player.get_move()
        elif move is not None:
            pass
        else:
            raise Exception('gomocup player没有获得落子点')
        player.move(*move)
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
    
    def restart(self):
        self.run = True
        self.winner = cfg.EMPTY
        self.round_num = 0
        self.player_color = cfg.BLACK
        for _, player in self.players.items():
            player.board.empty()

        self.start()

    def game_over(self):
        self.run = False
        msg = '游戏结束, {}'.format(self.RESULT[self.winner])
        print(msg)

    def show(self):
        print(self.players[cfg.BLACK].board.show() + self.players[cfg.WHITE].board.show())

if __name__ == '__main__':
    game = Game()
