import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import re
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split

# a list of initializer to try:
# tf.glorot_normal_initializer
# tf.glorot_uniform_initializer
# a list of activation to try:
# tf.nn.tanh
# tf.nn.relu
# tf.nn.elu
# tf.nn.sigmoid
# a list of hub_module to try:
# "https://tfhub.dev/google/nnlm-en-dim128/1"
# "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"
# "https://tfhub.dev/google/nnlm-en-dim50/1"
# "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1"
# a list of optimizer to try:
# tf.train.AdagradOptimizer
# tf.train.AdadeltaOptimizer
# tf.train.RMSPropOptimizer

def train(epochs,features_complete,total_picture,window_size=100,state_num1=25,state_num2=75,
            hub_module = "https://tfhub.dev/google/nnlm-en-dim50/1",
            initializer = tf.glorot_normal_initializer(),
            activation = tf.nn.tanh,
            optimizer = tf.train.AdamOptimizer):
    if hub_module in ["https://tfhub.dev/google/nnlm-en-dim128/1","https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"]:
        text_length = 128
    elif hub_module in ["https://tfhub.dev/google/nnlm-en-dim50/1","https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1"]:
        text_length = 50
    else:
        raise ValueError('{} not supported!'.format(hub_module))

    tf.reset_default_graph()


    prob = tf.placeholder(dtype = tf.float32)

    y = tf.placeholder(dtype = tf.float32, shape = (window_size,1))
    y_in = tf.expand_dims(y, 0)
    raw_input = tf.placeholder(dtype = tf.float32, shape = (window_size, 15))
    image_data = tf.placeholder(dtype = tf.float32, shape = (window_size, 12, 12, 2))
    sentences = tf.placeholder(dtype = tf.string, shape = (window_size,))

    kwargs = {'initializer':initializer, 'activation':activation, 'use_peepholes':True, 'reuse':tf.AUTO_REUSE}
    basic_cell1 = tf.nn.rnn_cell.LSTMCell(num_units = state_num1, **kwargs)
    basic_cell2 = tf.nn.rnn_cell.LSTMCell(num_units = state_num2, **kwargs)
    basic_cell1_w = tf.nn.rnn_cell.DropoutWrapper(basic_cell1, input_keep_prob = prob, output_keep_prob = prob, state_keep_prob= prob, input_size=text_length+159, variational_recurrent=True, dtype=tf.float32)
    basic_cell2_w = tf.nn.rnn_cell.DropoutWrapper(basic_cell2, input_keep_prob = prob, output_keep_prob = prob, state_keep_prob= prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([basic_cell1_w,basic_cell2_w])
  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.global_variables_initializer().run()}
    conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 2 , 8],  # 5x5 filter, depth 8.
                                stddev=0.1,
                                dtype=tf.float32))
    conv1_biases = tf.Variable(tf.zeros([8], dtype=tf.float32))
    conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 8, 16],
                                stddev=0.1,
                                dtype=tf.float32))
    conv2_biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32))

    pretrained_module = hub.Module(hub_module)

    conv = tf.nn.conv2d(image_data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool,
                         [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    image_out = tf.nn.dropout(reshape, prob)
    text_input = pretrained_module(sentences)
    input = tf.concat([raw_input,image_out,text_input],1)
    input = tf.expand_dims(input, 0)
    dense_final = tf.layers.Dense(1, activation=tf.nn.sigmoid)
    opt = optimizer(learning_rate=0.01)

    with tf.Session() as sess:
        with tf.variable_scope('test', reuse=tf.AUTO_REUSE):
            for ep in range(epochs):
                print('Training Epoch : {}'.format(ep))

                p = 0.5

                for t in range(20,25):
                    feed_dict = {prob:p,
                                 y:np.array(features_complete['y'])[70+t*window_size:70+(t+1)*window_size].reshape(-1,1),
                                 raw_input:np.array(features_complete.iloc[69+t*window_size:69+(t+1)*window_size,0:15]),
                                 image_data:total_picture[69+t*window_size:69+(t+1)*window_size],
                                 sentences:list(features_complete['Text'][69+t*window_size:69+(t+1)*window_size])}

                    if t == 20:
                        initial_state = cell.zero_state(tf.shape(input)[0],dtype=tf.float32)
                        rnn_output, states = tf.nn.dynamic_rnn(cell, input, time_major=False, dtype=tf.float32, initial_state=initial_state)
                    else:
                        rnn_output, states = tf.nn.dynamic_rnn(cell, input, time_major=False, dtype=tf.float32, initial_state=states)
 # dense_final is a callable

                    outputs = dense_final(rnn_output)

                    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_in,
                                                                   logits=outputs)
                    loss = tf.reduce_mean(loss)
        # this is equivalent to tf.reduce_mean(tf.square(outputs-y),(0,1,2))...
                             #Adam gradient descent method
                    training_op = opt.minimize(loss)

                    if ep==0 and t==20:
                        print('Initialize variables!')
                        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer(), tf.tables_initializer()])
                    if t==24:
                        _,train_loss,state_f = sess.run([training_op,loss,states],feed_dict=feed_dict)
                    else:
                        _,train_loss = sess.run([training_op,loss],feed_dict=feed_dict)
                    print('Training Loss at step {} : {}'.format(t,train_loss))
                    del feed_dict

            # test
                total_loss = 0
                for t in range(25,29):
                    feed_dict = {prob:1.0,
                                 y:np.array(features_complete['y'])[70+t*window_size:70+(t+1)*window_size].reshape(-1,1),
                                 raw_input:np.array(features_complete.iloc[69+t*window_size:69+(t+1)*window_size,0:15]),
                                 image_data:total_picture[69+t*window_size:69+(t+1)*window_size],
                                 sentences:list(features_complete['Text'][69+t*window_size:69+(t+1)*window_size])}

                    states_test = state_f
                    rnn_output_test, states_test = tf.nn.dynamic_rnn(cell, input, time_major=False, dtype=tf.float32, initial_state=states_test)
                    outputs_test = dense_final(rnn_output_test)
                    loss_test = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_in,
                                                                        logits=outputs_test)
                    loss_test = tf.reduce_mean(loss_test)

                    if ep==0 and t==25:
                        print('Initialize variables!')
                        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer(), tf.tables_initializer()])


                    loss_local = sess.run(loss_test,feed_dict = feed_dict)
                    total_loss = total_loss+loss_local
                    del feed_dict
                total_loss = total_loss/4
                print('Test Loss at step : {}'.format(total_loss))
