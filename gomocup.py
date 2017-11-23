import time
import config as cfg
from queue import Queue
from game import Game
from utils.socket import SocketClient, SOCKET_INIT_PARA
from utils.logger import Logger

class Gomocup(object):
    logger = Logger('gomocup')

    def __init__(self, size=None):
        self.get_board = False
        self.board_queue = Queue()
        self.end = False
        self.start = False
        self.begin = False
        self.robot_color = None
        self.size = size
        self.game = None
        self.messages = list()

    def start_game(self, color):
        if self.start:
            self.begin = True
            self.robot_color = color
            if color is cfg.BLACK:
                self.game = Game(white_player_type=cfg.GOMOCUP, size=self.size)
            else:
                self.game = Game(black_player_type=cfg.GOMOCUP, size=self.size)
            self.logger.info('begin play as {}'.format(cfg.COLOR[self.robot_color]))
        else:
            raise AttributeError('game has not started')

    def move(self, move=None):
        if move:
            self.game.round_process(move)
        else:
            move = self.game.round_process()
        self.logger.info('{}: ({},{})'.format(cfg.COLOR[self.robot_color], *move))
        return move

    def oppo_move(self, move=None):
        if move:
            self.game.round_process(move)
        else:
            raise AttributeError('gomocup player does not get move')
        self.logger.info('{}: ({},{})'.format(cfg.COLOR[-self.robot_color], *move))

    def do_command(self, cmd):
        try:
            cmds = cmd.strip().split(' ')
            if not cmds:
                raise AttributeError('no command')

            if self.get_board:
                if cmds[0] == 'DONE':
                    if not self.board_queue.qsize() % 2:
                        self.start_game(cfg.BLACK)
                    else:
                        self.start_game(cfg.WHITE)
                    while not self.board_queue.empty():
                        move, color = self.board_queue.get()
                        if color is 1:
                            self.move(move)
                        else:
                            self.oppo_move(move)
                    move = self.move()
                    self.get_board = False
                    return '{},{}'.format(*move)
                else:
                    move_color = list(map(int, cmd.split(',')))
                    self.board_queue.put((move_color[:2], move_color[2]))
                    self.logger.info('get move {}'.format(str(move_color)))
                    return 'None'

            if cmds[0] == 'START':
                try:
                    self.size = int(cmds[1])
                    self.start = True
                    self.logger.info('start game')
                    return 'OK'
                except:
                    raise AttributeError('unsupported size or other error')

            elif cmds[0] == 'BEGIN':
                self.start_game(cfg.BLACK)
                move = self.game.round_process()
                self.logger.info('{}: ({},{})'.format(cfg.COLOR[self.robot_color], *move))
                return '{},{}'.format(*move)

            elif cmds[0] == 'TURN':
                if not self.begin:
                    self.start_game(cfg.WHITE)
                oppo_move = list(map(int, cmds[1].split(',')))
                self.logger.info(oppo_move)
                self.oppo_move(oppo_move)
                move = self.move()
                return '{},{}'.format(*move)

            elif cmds[0] == 'RESTART':
                self.logger.info('restart game')
                self.__init__(self.size)
                self.start = True
                return 'OK'

            elif cmds[0] == 'ABOUT':
                self.logger.info('ask about info')
                return 'name="alpha-renju-zero", version="0.0", author="Cheer.L", country="China"'

            elif cmds[0] == 'END':
                self.logger.info('end')
                self.end = True
                return 'None'

            elif cmds[0] == 'INFO':
                self.logger.info('recieve info %s' % ' '.join(cmds[1:]))
                return 'None'

            elif cmds[0] == 'BOARD':
                self.logger.info('start recieve board')
                self.get_board = True
                return 'None'

            elif cmds[0] == 'RECTSTART':
                raise AttributeError('unsupported')

            elif cmds[0] == 'TAKEBACK':
                undo, oppo_undo = self.game.round_back()
                self.logger.info('{} undo: ({},{})'.format(*undo))
                self.logger.info('{} undo: ({},{})'.format(*oppo_undo))
                return 'OK'

            else:
                return 'None'
        except Exception as error:
            self.logger.error(error)
            return str(error)

    def process_cmd(self, cmd):
        response = self.do_command(cmd)
        self.messages.append(cmd)
        self.messages.append(response)
        return response

def main():
    gomocup = Gomocup()
    logger = gomocup.logger
    socket = SocketClient(*SOCKET_INIT_PARA)
    while True:
        try:
            socket.bind_addr(cfg.HOST, cfg.PORT)
            logger.info('socket bind success')
            break
        except:
            logger.error('socket connect Failed')
            time.sleep(1)
    try:
        while not gomocup.end:
            try:
                cmd = socket.recv_msg()
                response = gomocup.process_cmd(cmd)
                socket.send_msg(response)
            except Exception as e:
                logger.error(e)
                time.sleep(1)
    finally:
        socket.close()

if __name__ == '__main__':
    main()
