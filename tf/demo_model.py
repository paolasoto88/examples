import argparse
import sys

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

"""
An example of model parallelism in tensorflow, i.e., spreading a (huge) model over multiple resources.
"""

########################################################################################################################
# Prepare dataset.
########################################################################################################################

mnist = input_data.read_data_sets("/tfdata/data02/MNIST_data/", one_hot=True)


FLAGS = None

def main(_):
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device("/job:worker/task:{}".format(FLAGS.task_index)):
            # Build model.
            x = tf.placeholder(tf.float32, [None, 784], name="input")
            y_ = tf.placeholder(tf.float32, [None, 10], name="output")

            # Model.
            W1 = tf.Variable(tf.truncated_normal([784, 256], stddev=0.1), name="w1")
            b1 = tf.Variable(tf.zeros([256]), name="b1")
            y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

        with tf.device("/job:worker/task:{}".format(FLAGS.task_index)):
            # Model.
            W2 = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1), name="w2")
            b2 = tf.Variable(tf.zeros([10]), name="b2")
            y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)
            y = tf.argmax(tf.nn.softmax(y2, axis=1))

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y2))

            # Training.
            global_step = tf.train.get_or_create_global_step()
            train_op = tf.train.AdamOptimizer(0.01).minimize(loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [tf.train.StopAtStepHook(last_step=100000)]

    # The MonitoredTrainingSession takes care of session initialization, restoring from a checkpoint, saving to a
    # checkpoint, and closing when done or an error occurs.

    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
        if FLAGS.mode == 'train':
            train_step = 0
            while not mon_sess.should_stop():
                # Run a training step asynchronously. See `tf.train.SyncReplicasOptimizer` for additional details on how
                # to perform *synchronous* training. mon_sess.run handles AbortedError in case of preempted PS.
                batch_xs, batch_ys = mnist.train.next_batch(100)
                _, loss_val = mon_sess.run([train_op, loss], feed_dict={x: batch_xs, y_: batch_ys})
                train_step += 1
                if train_step % 100 == 0:
                    print('{}:{}: step {}: loss = {}'.format(FLAGS.job_name, FLAGS.task_index, train_step, loss_val))
            print("Done.")

        elif FLAGS.mode == 'inference':
            print("Inference time!")
            batch_xs, batch_ys = mnist.test.next_batch(10)
            out = mon_sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})
            print(list(zip(out, np.argmax(batch_ys, axis=1))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    # Flags for defining the operational mode: 'train' or 'inference'
    parser.add_argument(
        "--mode",
        type=str,
        default='train',
        help="Operational mode: 'train' or 'inference'."
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)




