import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tf_utils import load_mnist_data, preprocess_mnist_data, cnn_random_mini_batches
import mnist_inference

# Cost after epoch 0: 0.115412
# Cost after epoch 5: 0.043399
# Cost after epoch 10: 0.037064
# Cost after epoch 15: 0.035943
# Cost after epoch 20: 0.053889
# Cost after epoch 25: 0.051561
# Cost after epoch 30: 0.054482
# Cost after epoch 35: 0.060633
# Cost after epoch 40: 0.077461
# Cost after epoch 45: 0.051994
# Cost after epoch 50: 0.062426
# Cost after epoch 55: 0.076253
# Cost after epoch 60: 0.095747
# Cost after epoch 65: 0.062591
# Cost after epoch 70: 0.113315
# Cost after epoch 75: 0.084294
# Cost after epoch 80: 0.094767
# Cost after epoch 85: 0.100090
# Cost after epoch 90: 0.070288
# Cost after epoch 95: 0.107087
#

MODEL_SAVE_PATH = "/Users/liyuanzhao/Project/DeepLearning/mnist/src/cnn_mnist/model/"
MODEL_NAME = "mnist_model.ckpt"

REGULARIZATION_RATE = 0.1
# input params
INPUT_NUM_CHANNELS = 1
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
# output params
OUTPUT_NODE = 10

# Define layer1-conv filter size
CONV1_SIZE = 1
CONV1_DEEP = 8

# Define pool1 size
POOL1_SIZE = 2
POOL1_STEP = 2

# Define later2-conv filter size
CONV2_SIZE = 8
CONV2_DEEP = 16

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    x_image = tf.reshape(X, [-1, n_H0, n_W0, n_C0])
    ### END CODE HERE ###

    return X, Y, x_image


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


def initialize_parameters(conv1_f_shape, conv2_f_shape):
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    # He initialization(variance_scaling_initializer() )
    #   works better for layers with ReLu activation.
    # Xavier initialization ( xavier_initializer() )
    #   works better for layers with sigmoid activation.

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.get_variable("W1", conv1_f_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))#xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", conv2_f_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))#tf.contrib.layers.xavier_initializer(seed=0))

    conv1_bias = tf.get_variable("conv1_bias", conv1_f_shape[3], initializer=tf.constant_initializer(0.0))
    conv2_bias = tf.get_variable("conv2_bias", conv2_f_shape[3], initializer=tf.constant_initializer(0.0))

    tf.summary.histogram("conv1_weights", W1)
    tf.summary.histogram("conv1_bias", conv1_bias)
    tf.summary.histogram("conv2_weights", W2)
    tf.summary.histogram("conv2_bias", conv2_bias)

    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "W2": W2,
                  "B1": conv1_bias,
                  "B2": conv2_bias}

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    B1 = parameters['B1']
    B2 = parameters['B2']
    ### START CODE HERE ###
    with tf.name_scope("conv_layer_1"):
        # CONV2D: stride of 1, padding 'SAME'
        Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
        # RELU
        A1 = tf.nn.relu(Z1+B1)
        tf.summary.histogram('conv_layer1_relu_activations', A1)

        # MAXPOOL: window 2x2, sride 2, padding 'SAME'
        P1 = tf.nn.max_pool(A1, ksize=[1, POOL1_SIZE, POOL1_SIZE, 1], strides=[1, POOL1_STEP, POOL1_STEP, 1], padding='SAME')
    with tf.name_scope("conv_layer_2"):
        # CONV2D: filters W2, stride 1, padding 'SAME'
        Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
        # RELU
        A2 = tf.nn.relu(Z2+B2)
        tf.summary.histogram('conv_layer2_relu_activations', A2)
        # MAXPOOL: window 4x4, stride 4, padding 'SAME'
        P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.name_scope("full_layer"):
        # FLATTEN
        P2 = tf.contrib.layers.flatten(P2)
        # FULLY-CONNECTED without non-linear activation function (not not call softmax).
        # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
        Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=10, activation_fn=None)
        tf.summary.histogram('final_output', Z3)

    ### END CODE HERE ###

    return Z3


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
    tf.summary.scalar('cost', cost)
    ### END CODE HERE ###

    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009,
          num_epochs=50, minibatch_size=128, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    X, Y, x_image = create_placeholders(n_H0, n_W0, n_C0, n_y)
    tf.summary.image('input', x_image)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    conv1_f_shape = [CONV1_SIZE, CONV1_SIZE, INPUT_NUM_CHANNELS, CONV1_DEEP]
    conv2_f_shape = [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP]

    parameters = initialize_parameters(conv1_f_shape, conv2_f_shape)
    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###

    # Step of training number
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.name_scope('moving_average'):
        variabl_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variabl_averages_op = variabl_averages.apply(tf.trainable_variables())

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    # decaystep means : every 100000 steps with a base of 0.96:
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE
                                               , global_step
                                               , m / minibatch_size  #decay steps
                                               , LEARNING_RATE_DECAY)
    # Note as from https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
    # global_step: Optional Variable to increment by one after the variables have been updated.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)


    # computing accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1))
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    with tf.control_dependencies([optimizer, variabl_averages_op]):
        train_op = tf.no_op(name='train')

    #optimizer = tf.train.AdamOptimizer(learning_rate=0.009).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    merged = tf.summary.merge_all()

    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)
        sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064')

        train_writer = tf.summary.FileWriter("./mnist_train_log", sess.graph)
        test_writer = tf.summary.FileWriter("./mnist_test_log")
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = cnn_random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _, summary, temp_cost, step = sess.run([train_op, merged, cost, global_step],
                                                       feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###

                minibatch_cost += temp_cost/num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f, %i step" % (epoch, minibatch_cost, step))
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_writer.add_run_metadata(run_metadata, 'step%03d' % step)

                test_accuracy, test_summary, step = sess.run([accuracy, merged, global_step],
                                                        feed_dict={X: X_test, Y: Y_test})
                print("test_accuracy: ", test_accuracy, "step: ", global_step)
                test_writer.add_summary(test_summary, step)

            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

            train_writer.add_summary(summary, step)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions

        train_writer.close()
        test_writer.close()

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters

train_data, test_data = load_mnist_data()
x_train, y_train = preprocess_mnist_data(train_data)
x_test, y_test = preprocess_mnist_data(test_data)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[1], 1))
_, _, parameters = model(x_train, y_train, x_test, y_test)



