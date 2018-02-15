##############################################################################
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#/*
#* "INTEL CONFIDENTIAL" Copyright 2017 Intel Corporation All Rights
#* Reserved.
#*
#* The source code contained or described herein and all documents related
#* to the source code ("Material") are owned by Intel Corporation or its
#* suppliers or licensors. Title to the Material remains with Intel
#* Corporation or its suppliers and licensors. The Material contains trade
#* secrets and proprietary and confidential information of Intel or its
#* suppliers and licensors. The Material is protected by worldwide copyright
#* and trade secret laws and treaty provisions. No part of the Material may
#* be used, copied, reproduced, modified, published, uploaded, posted,
#* transmitted, distributed, or disclosed in any way without Intel's prior
#* express written permission.
#*
#* No license under any patent, copyright, trade secret or other
#* intellectual property right is granted to or conferred upon you by
#* disclosure or delivery of the Materials, either expressly, by
#* implication, inducement, estoppel or otherwise. Any license under such
#* intellectual property rights must be express and approved by Intel in
#* writing.
#*/
###############################################################################


from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer
from timeit import default_timer as timer
import sys
import numpy as np
import argparse


#hopefully this will get executed on the workers
import socket
import pickle

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


def onPart(part):
    #global sess, pnet, rnet, onet, g
    nr = 0

    for el in part:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_address = '/local_socket'
        sock.connect(server_address)
        filename = str(el[0])
        nimg = cv2.imdecode(np.frombuffer(pickle.loads(el[1]), np.uint8), 1)
        sock.sendall(pickle.dumps((filename,nimg)))
        sock.close()


i = 0
timeout = timer()

def rddFunc(rdd):
    global i
    global timeout
    j = rdd.count()
    print("RDD count = " , j)
    if j:
        timeout = timer()
        start = timer()
        x = 144
        rdd_rep = rdd.repartition(x)
        rdd_rep.foreachPartition(onPart)
        end = timer()
        timeout = timer()
        print("********************************************************")
        print('images/sec = ', (j - 0) / (end - start), "for ", j, " Images")
        print("########################################################")
    else:
        endtime = timer()
        if (endtime - timeout > 100):
            exit()
        else:
            print("Time elapsed without RDD: %r" % (endtime - timeout))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--zkQuorum', type=str,
                        help='<zk:port>', default='kafka-producer:2080')
    parser.add_argument('--kafka_producer', type=str,
    			help='<kafka:port>', default='kafka-producer:9092')
    parser.add_argument('--topic', type=str,
                        help='topic for consumer to listen to', default='topic_1')
    parser.add_argument('--wl_label', type=str,
                        help=' workload label for events', default='facenet-streaming-inference-classify-tf')
    # parser.add_argument('--topic_return', type=str,
    # help='topic for consumer to send images back', default='rec_back')
    parser.add_argument('--tag_dir', type=str,
                        help='<path to tag directory>',
                        default='/mnt/nvme_shared_x01/facenet-streaming-inference/logs/tags_${HOSTNAME}')
    parser.add_argument('--model', type=str,
                        help=' path to the model protobuf (.pb) file', default='/20170512-110547.pb')
    parser.add_argument('--classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.',
                        default='/lfw_classifier.pkl')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    parser.add_argument('--output_dir', type=str,
                        help='Directory with aligned face thumbnails.', default='/')
    parser.add_argument('--queue_server', type=str, help='<queue_server_name>',default="simple-queue-server-0")
    parser.add_argument('--queue_port', type=str, help='<queue_server_port>',default="50000")
    parser.add_argument('--kafka_topic_number', type=str, help="number of topics that Kafka will be streaming on",
                            default="10")
    return parser.parse_args(argv)


if __name__ == "__main__":
    global args
    args = parse_arguments(sys.argv[1:])
    sc = SparkContext(appName="SparkStreamingFaceRecognition")
    ssc = StreamingContext(sc, 3)
    #kvs = KafkaUtils.createStream(ssc, args.zkQuorum, "spark-streaming-consumer", {args.topic: 1})
    print(args.topic)
    print(args.kafka_producer)
    stream_no = int(args.kafka_topic_number)
    streams = []
    for i in range(stream_no):
        stream = KafkaUtils.createDirectStream(ssc, [args.topic + str(i)], {"metadata.broker.list":args.kafka_producer+":9092"})
        streams.append(stream)

    lines = ssc.union(*streams)
    lines.foreachRDD(rddFunc)
    ssc.start()
    ssc.awaitTermination()