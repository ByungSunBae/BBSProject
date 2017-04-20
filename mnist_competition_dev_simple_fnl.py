import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib import slim
import sys, os
import time

np.random.seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

flags = tf.app.flags
flags.DEFINE_integer('max_steps', 500000, """Number of max steps to run.""")

flags.DEFINE_string('train_dir', './mnist_mymodel_exp',
                           """Directory where to write event logs """
                           """and checkpoint.""")

flags.DEFINE_integer('batch_size', 100, 
                           """Number of batches.""")

flags.DEFINE_float('keep_drop_value', 0.5, 
                           """(1 - dropout ratio)""")

flags.DEFINE_boolean('only_cpu', False, 
                           """If true, Only use CPU.""")



FLAGS = tf.app.flags.FLAGS

def DnnSkipModule(inputs, num_outputs, n):
    with slim.arg_scope([slim.fully_connected], scope = ("DnnSkipModule"+str(n)),
                      weights_initializer=slim.initializers.xavier_initializer()):
        net1 = slim.fully_connected(inputs=inputs, num_outputs=num_outputs)
        net2 = slim.fully_connected(inputs=net1, num_outputs=num_outputs)
        net3 = slim.fully_connected(inputs=net2, num_outputs=num_outputs, activation_fn=None)
        net4 = tf.add(net3, tf.identity(net1))
        res = tf.nn.relu(net4)
        return res


with tf.device("/cpu:0"):
    x = tf.placeholder(tf.float32, [None, 784])
    keep_prob = tf.placeholder(tf.float32)
    global_step = tf.get_variable(
              'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    with slim.arg_scope([slim.fully_connected],
                      weights_initializer=slim.initializers.xavier_initializer()):
        net = slim.fully_connected(inputs=x, num_outputs=500, scope="net")
        net_drop = slim.dropout(inputs=net, keep_prob=keep_prob, scope="net_drop")
        net1 = slim.fully_connected(inputs=net_drop, num_outputs=400, scope="net1")
        net1_drop =slim.dropout(inputs=net1, keep_prob=keep_prob, scope="net1_drop")
        net1_skip = DnnSkipModule(inputs=net1_drop, num_outputs=400, n=1)
        net2 = DnnSkipModule(inputs=net1_skip, num_outputs=400, n=2)
        net3 = slim.fully_connected(inputs=net2, num_outputs=400, scope = "net3")
        net3_drop = slim.dropout(inputs=net3, keep_prob=keep_prob, scope="net3_drop")
        net4 = slim.fully_connected(inputs=net3_drop, num_outputs=10,
                                     scope = "net4", activation_fn=None)

    y = tf.nn.softmax(net4)

    y_ = tf.placeholder(tf.float32, [None, 10])

    num_batches_per_epoch = 60000 / FLAGS.batch_size
    num_epochs_per_decay = 10.0
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
    learning_rate_decay_rate = 0.2
    init_learning_rate = 0.1
    lr = tf.train.exponential_decay(init_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_rate,
                                    staircase=True)

    tf.summary.scalar('learning_rate',lr)
    opt = tf.train.GradientDescentOptimizer(lr)

    if FLAGS.only_cpu is not True:
        with tf.device("/gpu:0"):
            cross_entropy_model = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
            gvs = opt.compute_gradients(cross_entropy_model)
    else:
        cross_entropy_model = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        gvs = opt.compute_gradients(cross_entropy_model)

    t_list = [grad for grad, _  in gvs]

    global_norm = tf.global_norm(t_list)

    clipped_grads = [(tf.clip_by_average_norm(grad, global_norm), var) for grad, var in gvs]

    train_step = opt.apply_gradients(clipped_grads, global_step = global_step)

    init = tf.global_variables_initializer()

    tf.summary.scalar("Loss", cross_entropy_model)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in clipped_grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=tf.get_default_graph())
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: FLAGS.keep_drop_value})
            duration = time.time() - start_time

            if step % 100 == 0:
                print("Duration per batch size : ", duration, ", Step : ",step, 
                      ", Train Loss", sess.run(cross_entropy_model, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: FLAGS.keep_drop_value}))

            if step % 10 == 0:
                summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: FLAGS.keep_drop_value})
                summary_writer.add_summary(summary_str, step)

            if step % 5000 == 0 or (step+1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = step)

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        feed_train = {x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0}
        feed_test = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}

        print("Finally!!", " Train_ACC : ", sess.run(accuracy, feed_dict=feed_train), 
              ", Test_ACC : ", sess.run(accuracy, feed_dict=feed_test))
