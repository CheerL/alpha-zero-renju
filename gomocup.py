import time
import config as cfg
from game import Game
from utils.socket import SocketClient, SOCKET_INIT_PARA
from utils.logger import LogHandler

class Gomocup(object):
    COLOR = {
        cfg.BLACK: 'Black',
        cfg.WHITE: 'White'
    }
    def __init__(self, size=None):
        self.end = False
        self.start = False
        self.begin = False
        self.robot_color = None
        self.size = size
        self.game = None
        self.messages = list()
        self.logger = LogHandler('gomocup')

    def do_command(self, cmd):
        cmds = cmd.strip().split(' ')
        if not cmds:
            self.logger.error('no command')
            return 'ERROR no command'

        if cmds[0] == 'START':
            try:
                self.size = int(cmds[1])
                self.start = True
                self.logger.info('start game')
                return 'OK'
            except:
                self.logger.error('unsupported size or other error')
                return 'ERROR unsupported size or other error'
        elif cmds[0] == 'BEGIN':
            if self.start:
                self.begin = True
                self.robot_color = cfg.BLACK
                self.game = Game(cfg.ROBOT, cfg.GOMOCUP, self.size)
                move = self.game.round_process()
                self.logger.info('begin play as {}'.format(self.COLOR[self.robot_color]))
                self.logger.info('{}: ({},{})'.format(self.COLOR[self.robot_color], *move))
                return '{},{}'.format(*move)
            else:
                self.logger.error('game has not started')
                return 'ERROR game has not started'
        elif cmds[0] == 'TURN':
            if self.begin:
                gomocup_move = [int(each) for each in cmds[1].split(',')]
                self.game.round_process(gomocup_move)
                try:
                    move = self.game.round_process()
                    self.logger.info('{}: ({},{})'.format_map(self.COLOR[-self.robot_color], *gomocup_move))
                    self.logger.info('{}: ({},{})'.format_map(self.COLOR[self.robot_color], *move))
                except Exception as e:
                    self.logger.error(e)
                    raise Exception(e)
                return '{},{}'.format(*move)
            elif self.start:
                self.begin = True
                self.robot_color = cfg.WHITE
                self.game = Game(cfg.GOMOCUP, cfg.ROBOT, self.size)
                gomocup_move = [int(each) for each in cmds[1].split(',')]
                self.game.round_process(gomocup_move)
                move = self.game.round_process()
                self.logger.info('begin play as {}'.format(self.COLOR[self.robot_color]))
                self.logger.info('{}: ({},{})'.format(self.COLOR[-self.robot_color], *gomocup_move))
                self.logger.info('{}: ({},{})'.format(self.COLOR[self.robot_color], *move))
                return '{},{}'.format(*move)
            else:
                self.logger.error('game has not started')
                return 'ERROR game has not started'
        elif cmds[0] == 'RESTART':
            self.logger.info('restart game')
            self.__init__(self.size)
            self.start = True
            return 'OK'
        elif cmds[0] == 'ABOUT':
            self.logger.info('ask about info')
            return 'name="alpha-zero-renju", version="0.0", author="Cheer.L", country="China"'
        elif cmds[0] == 'END':
            self.logger.info('end')
            self.__init__()
            self.end = True
            return 'None'
        elif cmds[0] == 'INFO':
            self.logger.info('recieve info %s' % ' '.join(cmds[1:]))
            return 'None'
        else:
            return 'None'

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
