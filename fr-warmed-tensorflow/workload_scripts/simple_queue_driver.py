#!/usr/bin/env python3

import socket
import os

class GenericConnectionDriver:
    def __init__(self, name):
        self.name = name

    def push(self, data):
        print("\tpush::parent")
        pass

    def get(self):
        print("\tget::parent")
        pass


class SimpleQueueDriver(GenericConnectionDriver):
    def __init__(self, host, port, recv_bufsize=4096):
        GenericConnectionDriver.__init__(self, "SimpleQueue")
        self._client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._client_socket.connect((host, port))
        self._recv_bufsize = recv_bufsize

    def push(self, data):
        self._client_socket.sendall(data)
        #print("\tpush::" + self.__class__.__name__ + " push " + str(data.decode()) + " pid " + str(os.getpid()))

    def get(self):
        data = ""
        while True:
            x = self._client_socket.recv(self._recv_bufsize)
            if not x:
                break
            data += x
        #print("\tget::" + self.__class__.__name__ + " data " + data + " pid " + str(os.getpid()))
        return data

    def set_timeout(self, sec):
        self._client_socket.settimeout(sec)

    def close(self):
        self._client_socket.close()
