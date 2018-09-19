import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# this script deals with text data
# we used a pretrained model and NN to predict price movement
# we then extract the last hidden layer from NN to add to features


# a list of hub_module to try:
# "https://tfhub.dev/google/nnlm-en-dim128/1"
# "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"
# "https://tfhub.dev/google/nnlm-en-dim50/1"
# "https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1"
# a list of optimizer to try:
# tf.train.AdagradOptimizer
# tf.train.AdadeltaOptimizer
# tf.train.RMSPropOptimizer

def train_text_feature(epochs,train_text,train_label,test_text,test_label,
                       batch_size=100,
                       hub_module = "https://tfhub.dev/google/nnlm-en-dim128/1",
                       optimizer = tf.train.AdamOptimizer):
    '''
    Use a pretrained text embedding model followed by a 2 hidden layers NN to predict price movement.
    '''
    tf.reset_default_graph()

    train_flag = tf.placeholder(dtype = tf.bool)

    with tf.device('/cpu:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_text,train_label)).shuffle(1000).batch(batch_size).prefetch(10)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_text,test_label)).shuffle(100).batch(batch_size).prefetch(1)
        iter_train = train_dataset.make_initializable_iterator()
        sentences_train, y_train = iter_train.get_next()
        iter_test = test_dataset.make_initializable_iterator()
        sentences_test, y_test = iter_test.get_next()

    # load pretrained model
    pretrained_module = hub.Module(hub_module)

    # text embedding using pretrained model
    input = tf.cond(train_flag,lambda:pretrained_module(sentences_train),lambda:pretrained_module(sentences_test))

    # followed by several dense layers
    # note the dataset is small...so we don't want NN to be too large
    dense1 = tf.layers.Dense(64,activation=tf.nn.sigmoid)
    dense2 = tf.layers.Dense(16,activation=tf.nn.tanh)
    dense3 = tf.layers.Dense(1)

    dense1_output = dense1(input)
    dense2_output = dense2(dense1_output)
    outputs = dense3(dense2_output)

    loss = tf.cond(train_flag,lambda:tf.nn.sigmoid_cross_entropy_with_logits(labels=y_train,logits=outputs),
                              lambda:tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test,logits=outputs))

    loss = tf.reduce_mean(loss)
    opt = optimizer(learning_rate=0.03)
    training_op = opt.minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        for ep in range(1,epochs+2):
            sess.run(iter_test.initializer)
            if ep==1:
                sess.run([tf.global_variables_initializer(),tf.tables_initializer()])
            if ep%10==1:
                sess.run(iter_train.initializer)


            sess.run(training_op,feed_dict={train_flag:True})

            with tf.variable_scope('eval2', reuse=True):
                if ep%100 == 1:
                    sess.run(iter_train.initializer)
                    sess.run(iter_test.initializer)
                    print('Training Epoch : {}'.format(ep))
                    train_loss = sess.run(loss,feed_dict={train_flag:True})
                    sess.run(iter_test.initializer)
                    sess.run(iter_train.initializer)
                    test_loss = sess.run(loss,feed_dict={train_flag:False})
                    print('Training Loss at step {} : {}'.format(ep,train_loss))
                    print('Test Loss at step {} : {}'.format(ep,test_loss))

        save_path = saver.save(sess, "/training_text/model.ckpt")
        print("Model saved in path: %s" % save_path)

def get_text_feature(sentences,
                     hub_module = "https://tfhub.dev/google/nnlm-en-dim128/1",
                     save_path = "/training_text/model.ckpt"):
    '''
    Extract the features from the last hidden layer of our model.
    Note we need to redo this because we don't want to shuffle the train set.
    '''

    tf.reset_default_graph()
    # rebuild the 'same' graph
    pretrained_module = hub.Module(hub_module)

    input = pretrained_module(sentences)

    dense1 = tf.layers.Dense(64,activation=tf.nn.sigmoid)
    dense2 = tf.layers.Dense(16,activation=tf.nn.sigmoid)
    dense3 = tf.layers.Dense(1)

    dense1_output = dense1(input)
    dense2_output = dense2(dense1_output)
    outputs = dense3(dense2_output)

    with tf.Session() as sess:
        # load our trained model
        tf.train.Saver().restore(sess, save_path)
        # initialize table ! this step is necessary !
        sess.run(tf.tables_initializer())
        # extract last hidden layer
        return sess.run(dense2_output)
