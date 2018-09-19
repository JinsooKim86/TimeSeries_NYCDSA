import tensorflow as tf
import numpy as np
import pandas as pd
import time

def preprocess_data(x,y,time_window):
    '''
    Preprocess data so that train set becomes a list of time series data.
    Each time series data in this list will have time_len = time_window.
    Test set is generated as last time_window many data of the original time series.
    Train set and test set would be completely disjoint.
    '''
    train_x = []
    train_y = []
    test_x = x[1099-time_window:]
    test_y = y[1099-time_window:]
    for i in range(1100-2*time_window):
        train_x.append(x[i:i+time_window])
        train_y.append(y[i:i+time_window])

    return train_x,train_y,test_x,test_y

# a helper function to calculate accuracy
def get_accuracy(x,y,p=0.5):
    # flatten the data
    x = np.reshape(x,newshape=(-1))
    y = np.reshape(y,newshape=(-1))
    return ((x>=p)==y).mean()

def train_arc4(epochs,x,y,
               rnn_drop_prob = 0.5,batch_size=100,time_window = 100,num_units1=25,num_units2=25, lr = 0.01,
               optimizer = tf.train.AdamOptimizer):
    '''
    Given train set and test set (including time series and text data), we train a LSTM model that
    takes arbitrary time_window window data and predicts next day's price movement. We start with
    0 state each time. And finally, we evaluate on test set.
    Here, we also try the eager evaluation approach, with high level keras api.
    However, this model still overfits train set.
    '''
    if tf.executing_eagerly():
        pass
    else:
        raise ValueError('Must be in eager evaluation mode! Use tf.enable_eager_execution() .')

    tf.reset_default_graph()

    # prepare data
    train,train_label,test,test_label = preprocess_data(x,y,time_window)
    # train set is already a tensor of shape (batch_size,time_len,input_size)
    # but test set is a tensor of shape (time_len,input_size)
    # so we need to expand_dims at axis 0
    # note: keras.layers.CuDNNLSTM takes input of batch-major form
    test = tf.expand_dims(test,axis=0)
    test_label = tf.expand_dims(test_label,axis=0)

    # this time we can shuffle the data because we only care about the fixed window_size
    # and we do not care about the effect that is further away
    train_dataset = tf.data.Dataset.from_tensor_slices((train,train_label)).shuffle(1100-2*time_window)
    train_dataset = train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    # manually create 2 layers of LSTM
    cell1 = tf.keras.layers.CuDNNLSTM(num_units1,kernel_initializer='glorot_uniform',recurrent_initializer='glorot_uniform',
                                     return_sequences=True,
                                     return_state=False)
    dropout = tf.keras.layers.Dropout(rnn_drop_prob)
    cell2 = tf.keras.layers.CuDNNLSTM(num_units2,kernel_initializer='glorot_uniform',recurrent_initializer='glorot_uniform',
                                      return_sequences=True,
                                      return_state=False)
    dense_final = tf.layers.Dense(1)
    opt = optimizer(learning_rate=lr)

    accuracy_train = []
    accuracy_test = []

    for ep in range(epochs):
        # time the training time for each epoch
        start = time.time()

        # training
        # for example: our train set has 1100-2*window_size time stamps
        # say window_size = 100
        # then we have 990 training examples (each training example is itself a time series of length 100)
        # we take a batch of batch_size training examples and feed it to our LSTM network
        # say batch_size = 100
        # then each train_dataset would contain 9 batches
        # after each batch, we are going to update our weight once
        # note: each epoch will have 90 training examples dropped
        # but because we shuffle the dataset in the beginning, it does not matter in the long run
        for (batch, (X_train, y_train)) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # collect loss
                rnn_output1 = cell1(X_train)
                rnn_output1 = dropout(rnn_output1,training = True)
                rnn_output2 = cell2(rnn_output1)
                outputs = dense_final(rnn_output2)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train,
                                                               logits=outputs)
                loss = tf.reduce_mean(loss)

            pred = tf.sigmoid(outputs)
            # collect variables
            variables = cell1.variables + cell2.variables + dense_final.variables
            # auto diff gives gradients
            gradients = tape.gradient(loss, variables)
            # apply optimizer
            opt.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())

            train_accuracy = get_accuracy(pred,y_train)
            accuracy_train.append(train_accuracy)

        # evaluating:
        rnn_output_test1 = cell1(test)
        # note : during evaluation, should not apply dropout... but sometimes,we also need to rescale the output
        # to account for repetition
        rnn_output_test1 = dropout(rnn_output_test1,training = False)
        # rnn_output_test1 = tf.multiply(rnn_output_test1,tf.constant(1-rnn_drop_prob,shape = [1,time_window,num_units1]))
        rnn_output_test2 = cell2(rnn_output_test1)
        outputs_test = dense_final(rnn_output_test2)
        loss_test = tf.nn.sigmoid_cross_entropy_with_logits(labels=test_label,
                                                            logits=outputs_test)
        loss_test = tf.reduce_mean(loss_test)

        pred_test = tf.sigmoid(outputs_test)
        test_accuracy = get_accuracy(pred_test,test_label)
        accuracy_test.append(test_accuracy)

        print('Time taken for epoch {}: {} sec\n'.format(ep,time.time() - start))

    return accuracy_train,accuracy_test
