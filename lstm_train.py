import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

# in this script, we propose 3 architectures that combine time series data with text data
# to predict the price movement
# but they all overfit the train set

# a helper function to get accuracy
def get_accuracy(x,y,p=0.5):
    '''
    x,y are tensors
    x is the sigmoid output, y is the label
    p is the threshold
    first convert x to 0,1 using p
    then calculate accuracy
    '''
    # flatten everything
    x = np.reshape(x,(-1))
    y = np.reshape(y,(-1))
    return ((x>=p) == y).mean()

# a list of initializers to try:
# tf.glorot_normal_initializer
# tf.glorot_uniform_initializer

def train_arc3(epochs,train,train_label,test,test_label,train_step,
               rnn_drop_prob = 0.5,window_size=100,num_layers=2,num_units=25,
               initializer = tf.glorot_normal_initializer(),
               optimizer = tf.train.AdamOptimizer):
    '''
    Given train set, test set (including both time series data and text data), train a LSTM model which chain
    all the time together, but update weights every window_size time stamps.
    Then the final state is passed to generate output for test set.
    The features for text data can be found in './text_feature.pkl'
    '''

    tf.reset_default_graph()

    with tf.device('/cpu:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((train,train_label)).batch(window_size).prefetch(10)
        test_dataset = tf.data.Dataset.from_tensor_slices((test,test_label)).batch(window_size).prefetch(1)
        iter_train = train_dataset.make_initializable_iterator()
        X_train, y_train = iter_train.get_next()
        # we need to expand_dims at index 1 because we need a 3D tensor of shape (time_len,batch_size,input_size)
        # instead of a 2D tensor created by tf.data.Dataset shape (time_len,input_size)
        # note: batch_size here is just 1, so it is just a dummy dimension
        X_train = tf.expand_dims(X_train,1)
        y_train = tf.expand_dims(y_train,1)
        iter_test = train_dataset.make_initializable_iterator()
        X_test, y_test = iter_test.get_next()
        X_test = tf.expand_dims(X_test,1)
        y_test = tf.expand_dims(y_test,1)

    cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers,num_units,dropout=rnn_drop_prob,kernel_initializer=initializer)

    dense_final = tf.layers.Dense(1)

    opt = optimizer(learning_rate=0.03)

    # use python list to hold all losses and accuracies during training and evaluation
    loss_train = []
    loss_test = []
    accuracy_train = []
    accuracy_test = []

    with tf.Session() as sess:
        with tf.variable_scope('train_arc3', reuse=tf.AUTO_REUSE):
            for ep in range(epochs):
                print('Training Epoch : {}'.format(ep))
                sess.run(iter_test.initializer)
                sess.run(iter_train.initializer)
                # training
                for t in range(train_step):
                    if t == 0:
                        rnn_output, states = cell(X_train,training=True)
                    else:
                        rnn_output, states = cell(X_train,initial_state=states,training=True)

                    outputs = dense_final(rnn_output)


                    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train,
                                                                   logits=outputs)

                    loss = tf.reduce_mean(loss)
                    pred = tf.sigmoid(outputs)
                    pred = tf.squeeze(pred)

                    training_op = opt.minimize(loss)

                    if ep==0 and t==0:
                        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer(), tf.tables_initializer()])
                    if t==train_step-1:
                        _,train_loss,train_pred,state_f = sess.run([training_op,loss,pred,states])
                    else:
                        _,train_loss,train_pred = sess.run([training_op,loss,pred])
                    loss_train.append(train_loss)
                    train_accuracy = get_accuracy(train_pred,train_label[t*window_size:t*window_size+window_size])
                    accuracy_train.append(train_accuracy)

                # evaluation
                sess.run(iter_test.initializer)
                sess.run(iter_train.initializer)
                states_test = state_f
                rnn_output_test, states_test = cell(X_test,initial_state=states_test,training=False)
                outputs_test = dense_final(rnn_output_test)
                pred_test = tf.sigmoid(outputs_test)
                pred_test = tf.squeeze(pred_test)
                loss_t = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test,
                                                                   logits=outputs_test)
                loss_t = tf.reduce_mean(loss_t)



                loss_local,test_pred = sess.run([loss_t,pred_test])
                loss_test.append(loss_local)
                test_accuracy = get_accuracy(test_pred,test_label)
                accuracy_test.append(test_accuracy)


                # save model every 100 epoches
                if ep%100 == 0:
                    save_path = tf.train.Saver().save(sess, "/training_arc3/model{}.ckpt".format(ep))
                    print("Model saved in path: %s" % save_path)

    return loss_train,loss_test,accuracy_train,accuracy_test






def train_arc2(epochs,train,train_label,test,test_label,
               drop_prob1 = 0.5, drop_prob2 = 0.5, num_units1 = 25, num_units2 = 25,
               optimizer = tf.train.AdamOptimizer):
        '''
        Given train set and test set (including both time series data and text data), we train a LSTM model that
        uses all the time stamps before updating the weights. ( so this is very similar to the arc3 , the only difference
        is when we update the weights )
        Then the final state is passed to generate output for test set.

        Note that tf.contrib.cudnn_rnn.CudnnLSTM does not support variable time_len. Therefore, we use tf.keras.layers.CuDNNLSTM here.
        Also note that tensors feeded to tf.contrib.cudnn_rnn.CudnnLSTM need to be in time-major form,
        while tensors feeded to tf.keras.layers.CuDNNLSTM need to be in batch-major form.
        '''

        tf.reset_default_graph()

        # one shot approach
        with tf.device('/cpu:0'):
            train_dataset = tf.data.Dataset.from_tensor_slices((train,train_label)).batch(1000).repeat().prefetch(10)
            test_dataset = tf.data.Dataset.from_tensor_slices((test,test_label)).batch(100).repeat().prefetch(1)
            iter_train = train_dataset.make_one_shot_iterator()
            X_train, y_train = iter_train.get_next()
            X_train = tf.expand_dims(X_train,0)
            y_train = tf.expand_dims(y_train,0)
            iter_test = test_dataset.make_one_shot_iterator()
            X_test, y_test = iter_test.get_next()
            X_test = tf.expand_dims(X_test,0)
            y_test = tf.expand_dims(y_test,0)

        cell1 = tf.keras.layers.CuDNNLSTM(num_units1,kernel_initializer='glorot_uniform',recurrent_initializer='glorot_uniform',
                                         return_sequences=True,
                                         return_state=True)
        dropout1 = tf.keras.layers.Dropout(drop_prob1)
        cell2 = tf.keras.layers.CuDNNLSTM(num_units2,kernel_initializer='glorot_uniform',recurrent_initializer='glorot_uniform',
                                          return_sequences=True,
                                          return_state=True)
        dropout2 = tf.keras.layers.Dropout(drop_prob2)
        dense_final = tf.layers.Dense(1)
        opt = optimizer(learning_rate=0.05)


        loss_train = []
        loss_test = []
        accuracy_train = []
        accuracy_test = []
        with tf.Session() as sess:
            with tf.variable_scope('train_arc2', reuse=tf.AUTO_REUSE):
                for ep in range(epochs):
                    # training

                    # note that hidden state and cell state need to be caught separately
                    # i.e. it is not recognized as a tuple ...
                    rnn_output1, hstate1,cstate1 = cell1(X_train)
                    rnn_output1 = dropout1(rnn_output1,training = True)
                    rnn_output2, hstate2,cstate2 = cell2(rnn_output1)
                    rnn_output2 = dropout2(rnn_output2,training = True)
                    outputs = dense_final(rnn_output2)
                    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train,
                                                                   logits=outputs)
                    loss = tf.reduce_mean(loss)

                    pred = tf.sigmoid(outputs)
                    pred = tf.squeeze(pred)


                    training_op = opt.minimize(loss)

                    if ep==0:
                        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])


                    _,train_loss,final_hstate1,final_cstate1,final_hstate2,final_cstate2,train_pred = sess.run([training_op,loss,
                                                                                                                hstate1,cstate1,
                                                                                                                hstate2,cstate2,
                                                                                                                pred])

                    loss_train.append(train_loss)
                    train_accuracy = get_accuracy(train_pred,train_label)
                    accuracy_train.append(train_accuracy)

                    # evaluating
                    rnn_output_test1, _, _ = cell1(X_test,initial_state=(tf.convert_to_tensor(final_hstate1),tf.convert_to_tensor(final_cstate1)))
                    rnn_output_test1 = tf.multiply(rnn_output_test1,tf.constant(1-drop_prob1,shape = [1,100,num_units1]))
                    rnn_output_test2, _, _ = cell2(rnn_output_test1,initial_state = (tf.convert_to_tensor(final_hstate2),tf.convert_to_tensor(final_cstate2)))
                    rnn_output_test2 = tf.multiply(rnn_output_test2,tf.constant(1-drop_prob2,shape = [1,100,num_units2]))
                    outputs_test = dense_final(rnn_output_test2)
                    loss_t = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test,
                                                                       logits=outputs_test)
                    loss_t = tf.reduce_mean(loss_t)

                    pred_test = tf.sigmoid(outputs_test)
                    pred_test = tf.squeeze(pred_test)

                    loss_local,test_pred = sess.run([loss_t,pred_test])
                    loss_test.append(loss_local)
                    test_accuracy = get_accuracy(test_pred,test_label)
                    accuracy_test.append(test_accuracy)

                    # report loss every 10 epochs
                    if ep%10==0:
                        print('Train Loss at epoch {} : {}'.format(ep,train_loss))
                        print('Test Loss at epoch {} : {}'.format(ep,loss_local))
                    # save model every 100 epochs
                    if ep%100 == 0:
                        save_path = tf.train.Saver().save(sess, "/training_arc2/model{}.ckpt".format(ep))
                        print("Model saved in path: %s" % save_path)

        return loss_train,loss_test,accuracy_train,accuracy_test


def train_arc1(epochs,train,train_label,test,test_label,train_text,test_text,train_step,
               rnn_drop_prob = 0.5,window_size=100,num_layers=2,num_units=25,
               hub_module = "https://tfhub.dev/google/nnlm-en-dim50/1",
               initializer = tf.glorot_normal_initializer(),
               optimizer = tf.train.AdamOptimizer):
    '''
    Given time series data and text data, train a LSTM model which chain
    all the time together, but update weights every window_size time stamps.
    This is similar to train_arc3, but instead of train the text feature separately, we train it
    within the network.
    Then the final state is passed to generate output for test set.
    '''
    tf.reset_default_graph()

    train_flag = tf.placeholder(dtype = tf.bool)

    with tf.device('/cpu:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((train,train_text,train_label)).batch(window_size).prefetch(10)
        test_dataset = tf.data.Dataset.from_tensor_slices((test,test_text,test_label)).batch(window_size).prefetch(1)
        iter_train = train_dataset.make_initializable_iterator()
        X_train,sentences_train, y_train = iter_train.get_next()
        y_train = tf.expand_dims(y_train,1)
        iter_test = train_dataset.make_initializable_iterator()
        X_test,sentences_test, y_test = iter_test.get_next()
        y_test = tf.expand_dims(y_test,1)

    cell = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers,num_units,dropout=rnn_drop_prob,kernel_initializer=initializer)

    # use pretrained text embedding
    pretrained_module = hub.Module(hub_module)
    text_input = tf.cond(train_flag,lambda:pretrained_module(sentences_train),lambda:pretrained_module(sentences_test))

    with tf.device('/cpu:0'):
        # just add text embedding to our features
        input = tf.cond(train_flag,lambda:tf.concat([X_train,text_input],1),lambda:tf.concat([X_test,text_input],1))
        input = tf.expand_dims(input,1)

    dense_final = tf.layers.Dense(1)  # no activation
    opt = optimizer(learning_rate=0.03)

    loss_train = []
    loss_test = []
    accuracy_train = []
    accuracy_test = []

    with tf.Session() as sess:
        with tf.variable_scope('train_arc1', reuse=tf.AUTO_REUSE):
            for ep in range(epochs):
                print('Training Epoch : {}'.format(ep))
                sess.run(iter_test.initializer)
                sess.run(iter_train.initializer)
                # training
                for t in range(train_step):
                    if t == 0:
                        rnn_output, states = cell(input,training=True)
                    else:
                        rnn_output, states = cell(input,initial_state=states,training=True)

                    outputs = dense_final(rnn_output)


                    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train,
                                                                   logits=outputs)

                    loss = tf.reduce_mean(loss)
                    pred = tf.sigmoid(outputs)
                    pred = tf.squeeze(pred)

                    training_op = opt.minimize(loss)

                    if ep==0 and t==0:
                        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer(), tf.tables_initializer()])
                    if t==train_step-1:
                        _,train_loss,train_pred,state_f = sess.run([training_op,loss,pred,states],feed_dict={train_flag:True})
                    else:
                        _,train_loss,train_pred = sess.run([training_op,loss,pred],feed_dict={train_flag:True})
                    loss_train.append(train_loss)
                    train_accuracy = get_accuracy(train_pred,train_label[t*window_size:t*window_size+window_size])
                    accuracy_train.append(train_accuracy)

                # evaluating
                sess.run(iter_test.initializer)
                sess.run(iter_train.initializer)
                states_test = state_f
                rnn_output_test, states_test = cell(input,initial_state=states_test,training=False)
                outputs_test = dense_final(rnn_output_test)
                pred_test = tf.sigmoid(outputs_test)
                pred_test = tf.squeeze(pred_test)
                loss_t = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test,
                                                                   logits=outputs_test)
                loss_t = tf.reduce_mean(loss_t)

                loss_local,test_pred = sess.run([loss_t,pred_test],feed_dict = {train_flag:False})
                loss_test.append(loss_local)
                test_accuracy = get_accuracy(test_pred,test_label)
                accuracy_test.append(test_accuracy)

                if ep%100 == 0:
                    # save model each 100 epochs
                    save_path = tf.train.Saver().save(sess, "/training_arc1/model{}.ckpt".format(ep))
                    print("Model saved in path: %s" % save_path)

    return loss_train,loss_test,accuracy_train,accuracy_test
