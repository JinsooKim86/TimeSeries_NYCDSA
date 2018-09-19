import tensorflow as tf
import numpy as np
import pandas as pd

# these are the functions that try to train a vanilla NN or CNN on the hourly BTC price/volume data
# to predict next day's movement
# however, all the methods failed

def train_hourly_arc1(epochs,train_1d,train_label,test_1d,test_label,
                      nn_keep_prob = 0.5,optimizer = tf.train.AdamOptimizer):

    '''
    Here we take hourly BTC price and volumes as a 1d vector and just train a NN to predict next day's price movement.
    However, this overfits the train set.
    '''
    tf.reset_default_graph()

    # train_flag and eval_flag control whether we are using train set or test seet and whether we are using dropout or not
    train_flag = tf.placeholder(dtype = tf.bool)
    eval_flag = tf.placeholder(dtype = tf.bool)

    # put data pipeline on CPU to improve speed
    with tf.device('/cpu:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_1d,train_label)).batch(1000).prefetch(1)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_1d,test_label)).batch(100).prefetch(1)
        iter_train = train_dataset.make_initializable_iterator()
        image_train, y_train = iter_train.get_next()
        iter_test = test_dataset.make_initializable_iterator()
        image_test, y_test = iter_test.get_next()

    with tf.device('/gpu:0'):

        # 2 hidden layer NN
        dense1 = tf.layers.Dense(128,activation=tf.nn.tanh)
        dense2 = tf.layers.Dense(16,activation=tf.nn.tanh)
        dense3 = tf.layers.Dense(1)

        dense1_output = tf.cond(train_flag,lambda:dense1(image_train),lambda:dense1(image_test))
        # add a dropout to try to reduce overfitting problem
        dense1_output = tf.cond(eval_flag,lambda:tf.nn.dropout(dense1_output,nn_keep_prob),lambda:dense1_output)
        dense2_output = dense2(dense1_output)
        outputs = dense3(dense2_output)

        opt = optimizer(learning_rate=0.01)

        loss = tf.cond(train_flag,lambda:tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train,logits=outputs),
                                  lambda:tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test,logits=outputs))

        loss = tf.reduce_mean(loss)
        training_op = opt.minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # iterate epochs+1 times
        for ep in range(1,epochs+2):
            # run all initializer when necessary
            sess.run(iter_test.initializer)
            sess.run(iter_train.initializer)
            if ep==1:
                sess.run(tf.global_variables_initializer())

            # run optimization
            sess.run(training_op,feed_dict={train_flag:True,eval_flag:True})

            with tf.variable_scope('eval1', reuse=True):
                # eval train_loss and test_loss
                if ep%1000 == 1:
                    sess.run(iter_train.initializer)
                    sess.run(iter_test.initializer)
                    print('Training Epoch : {}'.format(ep))
                    train_loss = sess.run(loss,feed_dict={train_flag:True,eval_flag:False})
                    sess.run(iter_test.initializer)
                    sess.run(iter_train.initializer)
                    test_loss = sess.run(loss,feed_dict={train_flag:False,eval_flag:False})
                    print('Training Loss at step {} : {}'.format(ep,train_loss))
                    print('Test Loss at step {} : {}'.format(ep,test_loss))

        # save model in the end
        save_path = saver.save(sess, "/train_hourly_arc1/model.ckpt")
        print("Model saved in path: %s" % save_path)


def train_hourly_arc2(epochs,train_image,train_label,test_image,test_label,
                      nn_keep_prob = 0.8,batch_size=100,
                      optimizer = tf.train.AdamOptimizer):
    '''
    Here we take hourly BTC price and volumes as a 1d vector and train a CNN with 1D convolution to predict next day's price movement.
    However, this network does not converge.
    '''

    tf.reset_default_graph()

    train_flag = tf.placeholder(dtype = tf.bool)

    # put data pipeline on CPU
    # Note:: shuffle first!!! then batch / repeat etc...
    with tf.device('/cpu:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_image,train_label)).shuffle(1000).batch(batch_size).prefetch(10)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_image,test_label)).batch(batch_size).prefetch(1)
        iter_train = train_dataset.make_initializable_iterator()
        image_train, y_train = iter_train.get_next()
        iter_test = test_dataset.make_initializable_iterator()
        image_test, y_test = iter_test.get_next()


    with tf.device('/gpu:0'):
        conv1_weights = tf.Variable(tf.truncated_normal([3,2,8],
                                    stddev=0.1,
                                    dtype=tf.float32))
        conv1_biases = tf.Variable(tf.constant(0.01,shape=[8], dtype=tf.float32))
        conv2_weights = tf.Variable(tf.truncated_normal([3,8,32],
                                    stddev=0.1,
                                    dtype=tf.float32))
        conv2_biases = tf.Variable(tf.constant(0.01, shape=[32], dtype=tf.float32))

        # first 1d conv layer
        # input channel 2, output channel 8
        # window width 3
        # stride 1
        # padding : 'same', i.e. the output tensor remains the same shape as the input tensor
        # Note: we are using channel first data format as it is more efficient on GPU
        conv = tf.cond(train_flag,lambda: tf.nn.conv1d(image_train,
                                                       conv1_weights,
                                                       data_format='NCW',
                                                       stride=1,
                                                       padding='SAME'),
                                  lambda:tf.nn.conv1d(image_test,
                                                      conv1_weights,
                                                      data_format='NCW',
                                                      stride=1,
                                                      padding='SAME'))

        # reshape to a 4D tensor in order for the bias_add and max_pool to work
        # just add a dummy dimension with size 1
        conv = tf.reshape(conv,shape = (batch_size,8,1,23))
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases,data_format='NCHW'))
        # perform max pooling on the 4D tensor
        # this just reduce the size of the relevant dimension by half
        # in this case, it reduces 23 to 12
        pool = tf.nn.max_pool(relu,
                              ksize=[1,1,1,2],
                              strides=[1,1,1,2],
                              data_format='NCHW',
                              padding='SAME')
        # remove the extra dimension and make it a 3D tensor to input to the next conv layer
        pool = tf.squeeze(pool)
        # second 1d conv layer
        # input channel 8, output channel 32
        # window width 3
        # stride 1
        # padding: 'same'
        conv = tf.nn.conv1d(pool,
                            conv2_weights,
                            data_format='NCW',
                            stride=1,
                            padding='SAME')
        # reshape to a 4D tensor to do bias_add and max_pool
        conv = tf.reshape(conv,shape = (batch_size,32,1,12))
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases,data_format='NCHW'))
        pool = tf.nn.max_pool(relu,
                              ksize=[1,1,1,2],
                              strides=[1,1,1,2],
                              data_format='NCHW',
                              padding='SAME')

        # flatten the tensor to input to the dense layer
        image_out = tf.reshape(pool,
                               shape = (batch_size,192))

        # dropout if train
        image_out = tf.cond(train_flag,lambda:tf.nn.dropout(image_out, nn_keep_prob),lambda:image_out)

        # 4 hidden layer NN
        dense1 = tf.layers.Dense(256,activation=tf.nn.tanh)
        dense2 = tf.layers.Dense(256,activation=tf.nn.tanh)
        dense3 = tf.layers.Dense(64,activation = tf.nn.tanh)
        dense4 = tf.layers.Dense(16,activation = tf.nn.tanh)
        dense5 = tf.layers.Dense(1)

        dense1_output = dense1(image_out)
        dense2_output = dense2(dense1_output)
        dense3_output = dense3(dense2_output)
        dense4_output = dense4(dense3_output)
        outputs = dense5(dense4_output)

        opt = optimizer(learning_rate=0.03)

        loss = tf.cond(train_flag,lambda:tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train,logits=outputs),
                                  lambda:tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test,logits=outputs))

        loss = tf.reduce_mean(loss)
        training_op = opt.minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # train for epochs+1 steps
        for ep in range(1,epochs+2):

            # initialize variables whenever necessary
            sess.run(iter_test.initializer)
            if ep==1:
                sess.run(tf.global_variables_initializer())
            if ep%10==1:
                sess.run(iter_train.initializer)

            # run optimization
            sess.run(training_op,feed_dict={train_flag:True})

            with tf.variable_scope('eval1', reuse=True):
                # evaluate model every 1000 steps
                if ep%1000 == 1:
                    sess.run(iter_train.initializer)
                    sess.run(iter_test.initializer)
                    print('Training Epoch : {}'.format(ep))
                    train_loss = sess.run(loss,feed_dict={train_flag:True})
                    sess.run(iter_test.initializer)
                    sess.run(iter_train.initializer)
                    test_loss = sess.run(loss,feed_dict={train_flag:False})
                    print('Training Loss at step {} : {}'.format(ep,train_loss))
                    print('Test Loss at step {} : {}'.format(ep,test_loss))

        # save the final model
        save_path = saver.save(sess, "/train_hourly_arc2/model.ckpt")
        print("Model saved in path: %s" % save_path)

def train_hourly_arc3(epochs,train_image,train_label,test_image,test_label,
                      nn_keep_prob = 0.8,batch_size=100,
                      optimizer = tf.train.AdamOptimizer):
    '''
    Here we take hourly BTC price and volumes and artificially transform them to a 2d tensor
    and train a CNN with 2D convolution to predict next day's price movement.
    However, this network does not converge.
    '''

    tf.reset_default_graph()

    train_flag = tf.placeholder(dtype = tf.bool)

    with tf.device('/cpu:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_image,train_label)).shuffle(1000).batch(batch_size).prefetch(10)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_image,test_label)).batch(batch_size).prefetch(1)
        iter_train = train_dataset.make_initializable_iterator()
        image_train, y_train = iter_train.get_next()
        iter_test = test_dataset.make_initializable_iterator()
        image_test, y_test = iter_test.get_next()


    with tf.device('/gpu:0'):
        conv1_weights = tf.Variable(tf.truncated_normal([ 5, 5,2, 16],
                                    stddev=0.1,
                                    dtype=tf.float32))
        conv1_biases = tf.Variable(tf.constant(0.01,shape=[16], dtype=tf.float32))
        conv2_weights = tf.Variable(tf.truncated_normal([5, 5,16,64],
                                    stddev=0.1,
                                    dtype=tf.float32))
        conv2_biases = tf.Variable(tf.constant(0.01, shape=[64], dtype=tf.float32))

        # pretty much standard cnn structure
        # conv layer
        # then relu
        # then max_pool
        # repeat twice
        conv = tf.cond(train_flag,lambda: tf.nn.conv2d(image_train,
                                                       conv1_weights,
                                                       data_format='NCHW',
                                                       strides=[1, 1, 1, 1],
                                                       padding='SAME'),
                                  lambda:tf.nn.conv2d(image_test,
                                                      conv1_weights,
                                                      data_format='NCHW',
                                                      strides=[1, 1, 1, 1],
                                                      padding='SAME'))

        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases,data_format='NCHW'))

        pool = tf.nn.max_pool(relu,
                              ksize=[1,1,2,2],
                              strides=[1,1,2,2],
                              data_format='NCHW',
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            data_format='NCHW',
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases,data_format='NCHW'))
        pool = tf.nn.max_pool(relu,
                              ksize=[1,1,2,2],
                              strides=[1,1,2,2],
                              data_format='NCHW',
                              padding='SAME')

        image_out = tf.reshape(pool,
                               shape = (batch_size,576))

        image_out = tf.cond(train_flag,lambda:tf.nn.dropout(image_out, nn_keep_prob),lambda:image_out)

        # NN with 3 hidden layers
        dense1 = tf.layers.Dense(256,activation=tf.nn.tanh)
        dense2 = tf.layers.Dense(128,activation=tf.nn.tanh)
        dense3 = tf.layers.Dense(16,activation=tf.nn.tanh)
        dense4 = tf.layers.Dense(1)

        dense1_output = dense1(image_out)
        dense2_output = dense2(dense1_output)
        dense3_output = dense3(dense2_output)
        outputs = dense4(dense3_output)

        opt = optimizer(learning_rate=0.05)

        loss = tf.cond(train_flag,lambda:tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train,logits=outputs),
                                  lambda:tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test,logits=outputs))

        loss = tf.reduce_mean(loss)
        training_op = opt.minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        for ep in range(1,epochs+2):
            sess.run(iter_test.initializer)
            if ep==1:
                sess.run(tf.global_variables_initializer())
            if ep%10==1:
                sess.run(iter_train.initializer)


            sess.run(training_op,feed_dict={train_flag:True})

            with tf.variable_scope('eval1', reuse=True):
                if ep%1000 == 1:
                    sess.run(iter_train.initializer)
                    sess.run(iter_test.initializer)
                    print('Training Epoch : {}'.format(ep))
                    train_loss = sess.run(loss,feed_dict={train_flag:True})
                    sess.run(iter_test.initializer)
                    sess.run(iter_train.initializer)
                    test_loss = sess.run(loss,feed_dict={train_flag:False})
                    print('Training Loss at step {} : {}'.format(ep,train_loss))
                    print('Test Loss at step {} : {}'.format(ep,test_loss))


        save_path = saver.save(sess, "/train_hourly_arc3/model.ckpt")
        print("Model saved in path: %s" % save_path)
