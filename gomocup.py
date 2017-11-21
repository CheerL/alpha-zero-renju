import time
import config as cfg
from game import Game
from utils.socket import SocketClient, SOCKET_INIT_PARA
from utils.logger import LogHandler

class Gomocup(object):
    def __init__(self):
        self.end = False
        self.start = False
        self.begin = False
        self.robot_color = None
        self.size = None
        self.game = None
        self.messages = list()

    def do_command(self, cmd):
        cmds = cmd.strip().split(' ')
        if not cmds:
            return 'ERROR no command'
        if cmds[0] == 'START':
            try:
                self.size = int(cmds[1])
                self.start = True
                return 'OK'
            except:
                return 'ERROR unsupported size or other error'
        elif cmds[0] == 'BEGIN':
            if self.start:
                self.begin = True
                self.robot_color = cfg.BLACK
                self.game = Game(cfg.ROBOT, cfg.GOMOCUP, self.size)
                move = self.game.round_process()
                return '{},{}'.format(*move)
            else:
                return 'ERROR game has not started'
        elif cmds[0] == 'TURN':
            if self.begin:
                gomocup_move = [int(each) for each in cmds[1].split(',')]
                self.game.round_process(gomocup_move)
                move = self.game.round_process()
                return '{},{}'.format(*move)
            elif self.start:
                self.begin = True
                self.robot_color = cfg.WHITE
                self.game = Game(cfg.GOMOCUP, cfg.ROBOT, self.size)
                gomocup_move = [int(each) for each in cmds[1].split(',')]
                self.game.round_process(gomocup_move)
                move = self.game.round_process()
                return '{},{}'.format(*move)
            else:
                return 'ERROR game has not started'
        elif cmds[0] == 'RESTART':
            self.__init__()
            return 'OK'
        elif cmds[0] == 'ABOUT':
            return 'name="alpha-zero-renju", version="0.0", author="Cheer.L", country="China"'
        elif cmds[0] == 'END':
            self.__init__()
            self.end = True
            return 'None'
        elif cmds[0] == 'INFO':
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
    logger = LogHandler('gomocup')
    socket = SocketClient(*SOCKET_INIT_PARA)
    while True:
        try:
            socket.bind_addr(cfg.HOST, cfg.PORT)
            logger.info('Socket bind success')
            break
        except:
            logger.error('Connect Failed')
            time.sleep(1)
    try:
        while not gomocup.end:
            try:
                cmd = socket.recv_msg()
                logger.info(cmd)
                response = gomocup.process_cmd(cmd)
                socket.send_msg(response)
                logger.info(response)
            except:
                time.sleep(1)
    finally:
        socket.close()

if __name__ == '__main__':
    main()
