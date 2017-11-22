import os
import sys
import subprocess
import win32console
import pywintypes
import config as cfg
from utils.socket import SocketServer, SOCKET_INIT_PARA
from utils.logger import LogHandler
from utils import ROOT_PATH, LOG_PATH

def recv_msg():
    """read a line from sys.stdin"""
    return sys.stdin.readline().strip()

def send_msg(msg):
    """write a line to sys.stdout"""
    print(msg)
    sys.stdout.flush()

def main():
    """main function for AI console application"""
    handle = win32console.GetStdHandle(win32console.STD_INPUT_HANDLE)
    try:
        if handle.GetConsoleMode():
            send_msg("MESSAGE Gomoku AI should not be started directly. Please install gomoku manager (http://sourceforge.net/projects/piskvork). Then enter path to this exe file in players settings.")
    except pywintypes.error:
        pass

    file_name = os.path.join(LOG_PATH, 'temp')
    pipe = open(file_name, 'w+')
    process = subprocess.Popen(
        ['python', os.path.join(ROOT_PATH, 'gomocup.py')],
        stderr=pipe,
        stdout=pipe
    )
    logger = LogHandler('manager', handlers=['File'])
    logger.info('start background process')
    socket = SocketServer(*SOCKET_INIT_PARA)
    socket.bind_addr(cfg.HOST, cfg.PORT)
    logger.info('socket bind success')
    # a = process.communicate()

    try:
        # now keep talking with the client
        while True:
            cmd = recv_msg()
            logger.info(cmd)
            socket.send_msg(cmd)
            msg = socket.recv_msg()
            if msg != 'None':
                send_msg(msg)
                logger.info(msg)

    finally:
        pipe.close()
        socket.close()
        process.kill()
        os.remove(file_name)

if __name__ == '__main__':
    main()
