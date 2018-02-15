from __future__ import division
import Queue as queue
import socket
import threading
import sys
import errno
import pickle
import time
from timeit import default_timer as timer


push_port = 50000
get_port = 50001

class LockingList:
    def __init__(self):
        self._lock = threading.Lock()
        self._list = []

    def append(self, image):
        self._lock.acquire()
        self._list.append(image)
        self._lock.release()

    def pickle_and_empty(self):
        self._lock.acquire()
        ret = pickle.dumps(self._list)
        self._list = []
        self._lock.release()
        return ret

    def list_length(self):
        self._lock.acquire()
        size = len(self._list)
        self._lock.release()
        return size


class SimpleQueueServer(object):
    def __init__(self, push_port, get_port, nr_of_producers, nr_of_consumers, recv_bufsize, initial_batch_size,
                 dynamic_batching, max_batch_size, min_batch_size):
        self._queue = queue.Queue()
        self._push_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._push_socket.bind(('', push_port))
        self._get_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._get_socket.bind(('', get_port))
        self._recv_bufsize = recv_bufsize
        self._nr_of_producers = nr_of_producers
        self._nr_of_consumers = nr_of_consumers
        self._dynamic_batching = dynamic_batching
        self._batch_size = initial_batch_size
        self._max_batch_size = max_batch_size
        self._min_batch_size = min_batch_size


    def handler_producer(self, unpickled_batch, c, addr):
        pickled_tup = bytes()
        while True:
            x = c.recv(recv_bufsize)
            pickled_tup += x
            if not x:
                break

        unpickled_tup = pickle.loads(pickled_tup)
        if "EndOfBatch" in unpickled_tup[0]:
            unpickled_batch.append(unpickled_tup)
            time.sleep(2)
        else:
            unpickled_batch.append(unpickled_tup)
        c.close()

    def on_push(self):
        print("Server start push")
        self._push_socket.listen(self._nr_of_producers)
        self._push_socket.settimeout(2.0)

        unpickled_batch = LockingList()

        no_of_timeouts = 0
        batch_size = self._batch_size
        five_second_timer = timer()
        five_second_batch = 0
        passed_first_push = False
        while True:
            try:
                c, addr = self._push_socket.accept()
                no_of_timeouts = 0
                print("accepted push {}".format(addr))
                print("queue size is: ", self._queue.qsize())

                handler_thread = threading.Thread(target=self.handler_producer, args=(unpickled_batch, c, addr))
                handler_thread.start()

                if self._dynamic_batching:
                    if timer() - five_second_timer >= 5.0:
                        if (five_second_batch >= batch_size * 2) and self._queue.qsize() > 3:
                            if (batch_size * 2 <= self._max_batch_size):
                                batch_size = batch_size + batch_size // 2
                        if (five_second_batch <= batch_size // 2):
                            if (batch_size // 2 >= self._min_batch_size):
                                batch_size //= 2
                        if self._queue.qsize() < 3:
                            if (batch_size // 2 >= self._min_batch_size):
                                batch_size = batch_size - batch_size // 2
                        five_second_batch = 0
                        five_second_timer = timer()

                if unpickled_batch.list_length() >= batch_size:
                    five_second_batch += unpickled_batch.list_length()
                    self._queue.put(unpickled_batch.pickle_and_empty())
                    passed_first_push = True
            except socket.timeout:
                no_of_timeouts += 1
                if no_of_timeouts > 3:
                    while (passed_first_push == True):
                        unpickled_batch.append(("EndOfBatch",  None))
                        self._queue.put(unpickled_batch.pickle_and_empty())
                        time.sleep(2)

    def handler_consumer(self, c, addr):
        data = self._queue.get()
        try:
            c.sendall(data)
        except socket.error as e:
            if e.errno == errno.EPIPE:
                self._queue.put(data)
                print('Consumer timed out')
        c.close()

    def on_get(self):
        print("Server start get")
        self._get_socket.listen(self._nr_of_producers)
        while True:
            c, addr = self._get_socket.accept()
            print("accepted get {}".format(addr))

            handler_thread = threading.Thread(target=self.handler_consumer, args=(c, addr))
            handler_thread.start()
            #print('get from {} msg: {}'.format(addr, data))

    def start(self):
        push_thread = threading.Thread(target=self.on_push)
        get_thread = threading.Thread(target=self.on_get)
        push_thread.start()
        get_thread.start()

if __name__ == "__main__":
    import os
    push_port = int(os.environ.get("SIMPLE_QUEUE_PUSH_PORT", '50000'))
    get_port = int(os.environ.get("SIMPLE_QUEUE_GET_PORT", '50001'))
    max_conn_push = int(os.environ.get("SIMPLE_QUEUE_NR_OF_PRODUCERS", '10'))
    max_conn_get = int(os.environ.get("SIMPLE_QUEUE_NR_OF_CONSUMERS", '10'))
    recv_bufsize = int(os.environ.get("SIMPLE_QUEUE_RECV_BUFSIZE", "500000"))
    initial_batch_size = int(os.environ.get("SIMPLE_QUEUE_BATCH_SIZE", "400"))
    max_batch_size = int(os.environ.get("SIMPLE_QUEUE_MAX_BATCH_SIZE", "400"))
    min_batch_size = int(os.environ.get("SIMPLE_QUEUE_MIN_BATCH_SIZE", "15"))
    batching = os.environ.get("SIMPLE_QUEUE_DYNAMIC_BATCH","dynamic")
    if "dynamic" in batching:
        dynamic_batching = True
    else:
        dynamic_batching = False
    server = SimpleQueueServer(push_port, get_port, max_conn_push, max_conn_get, recv_bufsize,
                               initial_batch_size, dynamic_batching, max_batch_size, min_batch_size)
    server.start()