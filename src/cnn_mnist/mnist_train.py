import os
import tensorflow as tf
import mnist_inference
import numpy as np
from tf_utils import random_mini_batches, convert_to_one_hot

MNIST_PATH = "/Users/liyuanzhao/Project/DeepLearning/mnist/dataset/mnist.npz"
MODEL_SAVE_PATH = "/Users/liyuanzhao/Project/DeepLearning/mnist/src/cnn_mnist/model/"
MODEL_NAME = "mnist_model.ckpt"

OUTPUT_NODE = 10  # 10 classes
REGULARIZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.8
BATCH_SIZE = 100
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 10000


def load_data():
    f = np.load(MNIST_PATH)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train / 255, y_train), (x_test / 255, y_test)


def preprocess_data(data_set):
    x, y = data_set
    print("Original data X shape: ", x.shape, " data Y shape: ", y.shape)

    # x = x.reshape(x.shape[0], -1)
    y_one_hot = convert_to_one_hot(y, 10)
    y = y_one_hot.transpose()
    print("----- Reshape Original Trains Dataset Shape ------")
    print("Reshape data X as: ", x.shape, " data Y shape: ", y.shape)

    return (x, y)


def train(x_train, y_train):
    input_img_size = x_train.shape[1]
    print("input_img_size: ", input_img_size)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32
                           , shape=(None, input_img_size, input_img_size, mnist_inference.INPUT_NUM_CHANNELS)
                           , name='x-input')

        y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE), name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True, regularizer)

    # Step of training number
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.name_scope('moving_average'):
        variabl_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variabl_averages_op = variabl_averages.apply(tf.trainable_variables())

    with tf.name_scope('loss_function'):
        # loss function
        # For sparse_softmax_cross_entropy_with_logits,
        #   labels must have the shape [batch_size] and the dtype int32 or int64.
        #   Each label is an int in range [0, num_classes-1].
        # For softmax_cross_entropy_with_logits,
        #   labels must have the shape [batch_size, num_classes] and dtype float32 or float64.
        # Labels used in softmax_cross_entropy_with_logits are the one hot version of labels
        # used in sparse_softmax_cross_entropy_with_logits.
        print("---debug y shape: ", y.shape, "y_ shape: ", y_.shape)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        # print("---------- Debug --------- ")
        # keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # for key in keys:
        #   print(key.name)
        # print("---------------------")
        # weights_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        reg_weights = tf.add_n(tf.get_collection('losses'))
        loss = cross_entropy_mean + reg_weights
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('reg_wights', reg_weights)

    input_x_number_examples = x_train.shape[0]
    print("input_x_number_examples : ", input_x_number_examples)
    with tf.name_scope('train_step'):
        # learning rate decay
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE
                                                   , global_step
                                                   , input_x_number_examples / BATCH_SIZE
                                                   , LEARNING_RATE_DECAY)
        # Note as from https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
        # global_step: Optional Variable to increment by one after the variables have been updated.
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variabl_averages_op]):
            train_op = tf.no_op(name='train')

    # saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        seed = 3
        # reshape x_train from [60000, 28, 28] to [60000, 28*28]
        x_train = x_train.reshape(x_train.shape[0], -1)
        print("---- before for loop: reshape x_train: ", x_train.shape)

        # random_mini_batches() function requires:  x as (input size, number of examples)
        # y label vector: as shape (1, number of examples)
        mini_batches = random_mini_batches(tf.transpose(x_train), tf.transpose(y_train), BATCH_SIZE, seed)

        # print("---- before for loop: mini_batches.shape: ", mini_batches.shape)
        for i in range(TRAINING_STEPS):
            k = i % len(mini_batches)
            if k == 0:
                seed = seed + 1
                mini_batches = random_mini_batches(tf.transpose(x_train), tf.transpose(y_train), BATCH_SIZE, seed)

            mini_x_batches, mini_y_batches = mini_batches[k]
            # print(" mini_x_batches shape: ", mini_x_batches.shape, "mini_y_batches shape: ", mini_y_batches.shape)
            reshaped_xs = np.reshape(mini_x_batches,
                                     (BATCH_SIZE, input_img_size, input_img_size, mnist_inference.INPUT_NUM_CHANNELS))
            # print(" reshaped_xs shape: ", reshaped_xs.shape, "mini_y_batches shape:", mini_y_batches.shape)
            # print("----- reshaped_xs : ", reshaped_xs)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: mini_y_batches})
            if (i % 1000) == 0:
                print("After %d training step(s), loss on training batch is %g" % (step, loss_value))
                print("----- global_step: ", sess.run(global_step))
            # if(step == 1 ):
            # 	saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)
            # else:
            # 	saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step, write_meta_graph = False)

    writer = tf.summary.FileWriter("./mnist_log", tf.get_default_graph())
    writer.close()

def main(argv=None):
    train_set, test_set = load_data()
    x_train, y_train = preprocess_data(train_set)
    train(x_train, y_train)


if __name__ == "__main__":
    tf.app.run()
