from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from timeit import default_timer as timer
import math
import pickle
import os
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.python.platform import gfile
import signal
import sys
import simple_queue_driver


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        # print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(image_array, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_array)
    # nrof_samples = 1
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    # print (images)
    for i in range(nrof_samples):
        # print (image_paths)
        # img = misc.imread(image_paths)
        img = image_array[i]
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i, :, :, :] = img
    return images

times = timer()
def signal_handler(signal, frame):
    global times
    print("Elapsed: ",timer() -times)
    sys.exit(0)

def classify(queue_of_batches, imgs_per_process_queue, confidence_queue, classifier_filename, model, image_size,
             batch_max_size, queue_server_name, queue_server_port, queue_recv_size, tf_intra_op, tf_inter_op):
    server_ip = queue_server_name
    port = queue_server_port
    recv_bufsize = queue_recv_size

    image_counter = 0
    confidence = 0
    batch_size = batch_max_size

    sessconfig = tf.ConfigProto(
        intra_op_parallelism_threads=tf_intra_op,
        inter_op_parallelism_threads=tf_inter_op)
    with tf.Graph().as_default():
        with tf.Session(config=sessconfig) as sess:
            import random
            y = random.randint(0,999)
            # Load the model
            # print('Loading feature extraction model')
            load_model(model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename_exp = os.path.expanduser(classifier_filename)

            # print('Testing classifier')
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            with open("/output_inference_"+str(y)+".txt", 'w') as outputfile:
                exitbatch = False
                while True:
                    consumer = simple_queue_driver.SimpleQueueDriver(server_ip, port, recv_bufsize)
                    pickled_batch = consumer.get()
                    unpickled_batch = pickle.loads(pickled_batch)
                    unpickled_images = []
                    unpickled_names = []
                    if len(unpickled_batch) > batch_size:
                        batch_size = len(unpickled_batch)
                    for el in unpickled_batch:
                        if (el[1] is not None and "EndOfBatch" not in el[0]):
                            unpickled_names.append(el[0])
                            unpickled_images.append(el[1])
                        else:
                            exitbatch = True

                    print("New batch arrived! Size is: ", len(unpickled_batch))
                    batch_time = timer()
                    nrof_images = len(unpickled_images)

                    if (nrof_images > 0):
                        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
                        emb_array = np.zeros((nrof_images, embedding_size))

                        for i in range(nrof_batches_per_epoch):
                            start_index = i * batch_size
                            end_index = min((i + 1) * batch_size, nrof_images)
                            # paths_batch = paths[start_index:end_index]
                            # print (paths_batch)
                            images = load_data(unpickled_images, False, False, image_size)
                            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                            # print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        for i in range(len(best_class_indices)):
                            outputfile.write(' %s %s: %.3f \n' % (unpickled_names[i], class_names[best_class_indices[i]], best_class_probabilities[i]))
                            outputfile.flush()
                            confidence+=best_class_probabilities[i]
                        image_counter += nrof_images
                        print("Batch time is:", timer() - batch_time)
                        consumer.close()
                    if exitbatch:
                        imgs_per_process_queue.put(image_counter)
                        confidence_queue.put(confidence / image_counter)
                        queue_of_batches.get(True)
                        queue_of_batches.task_done()
                        return
                        #sys.exit(0)



if __name__ == "__main__":

    #Replace with your specific environment variables
    classifier_filename = os.environ.get("WARMTENSOR_CLASSIFIER", '/lfw_classifier.pkl')
    model = os.environ.get("WARMTENSOR_MODEL",'/20170512-110547.pb')
    logdir = os.environ.get("WARMTENSOR_LOGDIR",'/')
    num_processes = int(os.environ.get("WARMTENSOR_PROCESESS","2"))
    max_batch_size  = int(os.environ.get("WARMTENSOR_MAX_BATCH_SIZE","400"))
    image_size = int(os.environ.get("WARMTENSOR_IMG_SIZE","160"))
    queue_server_name = os.environ.get("WARMTENSOR_SERVER","simple-queue-server-0")
    queue_server_port = int(os.environ.get("WARMTENSOR_SERVER_PORT","50001"))
    queue_recv_size = int(os.environ.get("WARMTENSOR_SERVER_RECV","4000000"))
    tf_intra_op = int(os.environ.get("WARMTENSOR_INTRA_OP","36"))
    tf_inter_op = int(os.environ.get("WARMTENSOR_INTER_OP","1"))

    times = timer()
    from multiprocessing import JoinableQueue
    from multiprocessing import Queue
    from multiprocessing import Pool

    queue_of_batches = JoinableQueue()
    imgs_per_process_queue = Queue()
    confidence_queue = Queue()
    for i in range(int(num_processes)):
        queue_of_batches.put(object())

    tf_pool = Pool(int(num_processes), classify, (queue_of_batches, imgs_per_process_queue,
                                                  confidence_queue, classifier_filename, model, image_size,
                                                  max_batch_size, queue_server_name, queue_server_port,
                                                  queue_recv_size, tf_intra_op, tf_inter_op))
    queue_of_batches.join()
    endtime = timer()-times
    throughput = 0
    avg_confidence = 0
    for i in range(int(num_processes)):
        throughput+=imgs_per_process_queue.get(True)
        avg_confidence+=confidence_queue.get(True)

    with open("/relevant_metrics","w") as relevant_metrics:
        relevant_metrics.write('%.3f ImagesPerSecond \n %.3f AverageConfidence\n' % (throughput / endtime, avg_confidence / num_processes))
        relevant_metrics.flush()


    sys.exit(0)
