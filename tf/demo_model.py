import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

"""
An example of model parallelism in tensorflow, i.e., spreading a (huge) model over multiple resources.
"""

########################################################################################################################
# Prepare dataset.
########################################################################################################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


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
            W1 = tf.Variable(tf.zeros([784, 10]), name="w1")
            b1 = tf.Variable(tf.zeros([10]), name="b1")
            y1 = tf.matmul(x, W1) + b1

        with tf.device("/job:worker/task:{}".format(FLAGS.task_index)):
            # Model.
            W2 = tf.Variable(tf.zeros([10, 10]), name="w2")
            b2 = tf.Variable(tf.zeros([10]), name="b2")
            y2 = tf.matmul(y1, W2) + b2

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y2))

            # Training.
            global_step = tf.train.get_or_create_global_step()
            train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [tf.train.StopAtStepHook(last_step=10000)]

    # The MonitoredTrainingSession takes care of session initialization, restoring from a checkpoint, saving to a
    # checkpoint, and closing when done or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
        train_step = 0
        while not mon_sess.should_stop():
            print('{}:{}: training step {}'.format(FLAGS.job_name, FLAGS.task_index, train_step))
            # Run a training step asynchronously. See `tf.train.SyncReplicasOptimizer` for additional details on how to
            # perform *synchronous* training. mon_sess.run handles AbortedError in case of preempted PS.
            batch_xs, batch_ys = mnist.train.next_batch(100)
            mon_sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
            train_step += 1

        print("Done.")


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
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
