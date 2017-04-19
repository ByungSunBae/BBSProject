import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib import slim
import sys, os
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

flags = tf.app.flags
flags.DEFINE_integer('max_steps', 500000,
                            """Number of max steps to run.""")

flags.DEFINE_string('train_dir', '/home/spark/hard/Deep_Learning_for_all_people/mnist_mymodel_exp5',
                           """Directory where to write event logs """
                           """and checkpoint.""")

flags.DEFINE_integer('batch_size', 100,
                            """Number of batches.""")

FLAGS = tf.app.flags.FLAGS


def DnnSkipModule(inputs, num_outputs, n, auxilary=False):
    with slim.arg_scope([slim.fully_connected], scope = ("DnnSkipModule"+str(n)),
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
                      #weights_regularizer=slim.l2_regularizer(0.0005)):
        net1 = slim.fully_connected(inputs=inputs, num_outputs=num_outputs)
        net2 = slim.fully_connected(inputs=net1, num_outputs=num_outputs)
        if auxilary:
            net2_aux1 = slim.fully_connected(inputs=net2, num_outputs=num_outputs, scope = "aux1")
            net2_aux2 = slim.fully_connected(inputs=net2_aux1, num_outputs=num_outputs, scope = "aux2")
            aux_logit = slim.fully_connected(inputs=net2_aux2, num_outputs=10, scope = "aux_logit")
        net3 = slim.fully_connected(inputs=net2, num_outputs=num_outputs, activation_fn=None)
        net4 = tf.add(net3, tf.identity(net1))
        res = tf.nn.relu(net4)
    if auxilary:
        return res, aux_logit
    else:
        return res



with tf.device("/cpu:0"):
    x = tf.placeholder(tf.float32, [None, 784])
    global_step = tf.get_variable(
              'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    with slim.arg_scope([slim.fully_connected],
                      weights_initializer=slim.initializers.xavier_initializer()):
        net = slim.fully_connected(inputs=x, num_outputs=400, scope="net")
        net1 = DnnSkipModule(inputs=net, num_outputs=400, n=1)
        #net2 = DnnSkipModule(inputs=net1, num_outputs=700, n=2)
        #net3 = DnnSkipModule(inputs=net2, num_outputs=700, n=3)
        #net4, aux_logit = DnnSkipModule(inputs=net3, num_outputs=700, n=4, auxilary=True)
        #net5 = DnnSkipModule(inputs=net4, num_outputs=700, n=5)
        net6 = slim.fully_connected(inputs=net1, num_outputs=200, scope = "net6")
        net7 = slim.fully_connected(inputs=net6, num_outputs=10, 
                                     scope = "net7", activation_fn=None)

    #y_aux = tf.nn.softmax(aux_logit)
    y_model = tf.nn.softmax(net7)

    y_ = tf.placeholder(tf.float32, [None, 10])
    #y_aux_ = tf.placeholder(tf.float32, [None, 10])

    #cross_entropy_aux = -tf.reduce_sum(y_aux_ * tf.log(y_aux), reduction_indices=[1])
    #cross_entropy_aux = tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=aux_logit, 
    #                                labels=y_aux_), 0.4)
    #cross_entropy_model = tf.nn.softmax_cross_entropy_with_logits(logits=y_model, labels=y_)

    #cross_entropy = tf.add(cross_entropy_aux, cross_entropy_model)

    num_batches_per_epoch = 60000 / FLAGS.batch_size
    num_epochs_per_decay = 10.0
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
    learning_rate_decay_rate = 0.1
    init_learning_rate = 0.3
    lr = tf.train.exponential_decay(init_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_rate,
                                    staircase=True)

    tf.summary.scalar('learning_rate',lr)
    opt = tf.train.GradientDescentOptimizer(lr)

    with tf.device("/gpu:0"):
        #cross_entropy = tf.reduce_mean([cross_entropy_aux, cross_entropy_model])
        cross_entropy_model = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_model, labels=y_))
        gvs = opt.compute_gradients(cross_entropy_model)

    t_list = [grad for grad, _  in gvs]

    global_norm = tf.global_norm(t_list)

    clipped_grads = [(tf.clip_by_average_norm(grad, global_norm), var) for grad, var in gvs]

    #clipped_grads = [(tf.clip_by_global_norm(t_list, global_norm), t_vars)]

    train_step = opt.apply_gradients(clipped_grads)

    #train_step = opt.minimize(cross_entropy)

    init = tf.global_variables_initializer()

    tf.summary.scalar("Loss", cross_entropy_model)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in clipped_grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    merged_summary_op = tf.summary.merge_all()

#    saver = tf.train.Saver(tf.global_variables())

#    if tf.gfile.Exists(FLAGS.train_dir):
#        tf.gfile.DeleteRecursively(FLAGS.train_dir)
#
#    tf.gfile.MakeDirs(FLAGS.train_dir)

    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=tf.get_default_graph())
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            duration = time.time() - start_time

            if step % 100 == 0:
                print("Duration : ", duration, ", Step : ",step, 
                      ", Loss", sess.run(cross_entropy_model, feed_dict={x: batch_xs, y_: batch_ys}))

            if step % 10 == 0:
                summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y_: batch_ys})
                summary_writer.add_summary(summary_str, step)

 #           if step % 5000 == 0 or (step+1) == FLAGS.max_steps:
 #               checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
 #               saver.save(sess, checkpoint_path, global_step = step)

        correct_prediction = tf.equal(tf.argmax(y_model,1), tf.argmax(y_,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        feed_train = {x: mnist.train.images, y_: mnist.train.labels}
        feed_test = {x: mnist.test.images, y_: mnist.test.labels}

        print("Finally!!", " Train_ACC : ", sess.run(accuracy, feed_dict=feed_train), 
              ", Test_ACC : ", sess.run(accuracy, feed_dict=feed_test))

