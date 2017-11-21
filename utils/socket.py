import sys
import socket

SOCKET_INIT_PARA = [socket.AF_INET, socket.SOCK_STREAM]

class SocketClient(socket.socket):
    host, port = None, None

    def bind_addr(self, host, port):
        self.host = host
        self.port = port
        try:
            self.connect((self.host, self.port))
        except socket.error:
            # print('Connect failed')
            sys.exit()

    def send_msg(self, msg):
        self.send(msg.encode())

    def recv_msg(self, buffer_size=1024):
        return self.recv(buffer_size).decode()

class SocketServer(socket.socket):
    conn = None
    # def __init__(self):
    #     super().__init__(socket.AF_INET, socket.SOCK_DGRAM)
    #     self.bind((cfg.HOST, cfg.PORT))
    #     self.conn = None

    def bind_addr(self, host, port):
        try:
            self.bind((host, port))
            # print('Socket bind complete')
            self.check_conn()
        except socket.error:
            # print('Bind failed')
            sys.exit()

    def send_msg(self, msg):
        self.check_conn()
        self.conn.send(msg.encode())

    def recv_msg(self, buffersize=1024):
        self.check_conn()
        return self.conn.recv(buffersize).decode()

    def check_conn(self):
        if self.conn is None:
            self.listen(5)
            # print('Wait connection')
            self.conn, addr = self.accept()
            # print('Connected with {}:{}'.format(*addr))
