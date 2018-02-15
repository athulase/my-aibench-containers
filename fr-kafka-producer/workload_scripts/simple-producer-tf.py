'''
###################################################################################
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
###################################################################################
'''

from kafka import KafkaProducer
import sys
import pickle
import os

images_path = '/mnt/nvme_shared_x01/AIBench_shared_disk/Data/facenet_images_raw'
KAFKA_IP= sys.argv[1]
TOPIC_NAME = sys.argv[2]
DATA_DIR = sys.argv[3]
WL = sys.argv[4]
producer = KafkaProducer(bootstrap_servers=[KAFKA_IP + ':9092'])
writers = int(os.environ.get("KAFKA_TOPIC_NUMBER", '10'))

i = 0
for file in os.listdir(images_path):
    print("Sending " + file)
    string = file
    with open(images_path + "/" + file, 'rb') as f:
        data = f.read()
        data_to_send = pickle.dumps(data)

    producer.send(TOPIC_NAME + str(i % writers), key=string, value=data_to_send)
    i = i + 1

print("Images sent: " + str(i))

exit()

