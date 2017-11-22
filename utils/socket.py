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
            self.settimeout(5)
        except socket.error:
            # print('Connect failed')
            sys.exit()

    def send_msg(self, msg):
        self.send(msg.encode())

    def recv_msg(self, buffer_size=1024, retry=0):
        try:
            return self.recv(buffer_size).decode()
        except socket.timeout:
            if retry < 3:
                return self.recv_msg(buffer_size, retry + 1)
            else:
                raise socket.timeout

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

    def recv_msg(self, buffersize=1024, retry=0):
        self.check_conn()
        try:
            return self.conn.recv(buffersize).decode()
        except socket.timeout:
            if retry < 3:
                return self.recv_msg(buffersize, retry + 1)
            else:
                raise socket.timeout
            

    def check_conn(self):
        if self.conn is None:
            self.listen(5)
            # print('Wait connection')
            self.conn, _ = self.accept()
            self.conn.settimeout(5)
            # print('Connected with {}:{}'.format(*addr))
