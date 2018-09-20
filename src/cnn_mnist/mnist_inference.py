import tensorflow as tf

# input params
INPUT_NUM_CHANNELS = 1

# output params
OUTPUT_NODE = 10

# Define layer1-conv filter size
CONV1_SIZE = 5
CONV1_DEEP = 32

# Define pool1 size
POOL1_SIZE = 2
POOL1_STEP = 2

# Define later2-conv filter size
CONV2_SIZE = 5
CONV2_DEEP = 64

# Define FC layer number of nodes
FC_SIZE = 512


def inference(input_tensor, if_train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weights",
                                        [CONV1_SIZE, CONV1_SIZE, INPUT_NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        # conv1_print = tf.print(conv1,
        # [conv1, tf.shape(conv1), '---conv1 dump shape---'],
        # message='--Debug conv--', summarize=100)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, POOL1_SIZE, POOL1_SIZE, 1], strides=[1, POOL1_STEP, POOL1_STEP, 1],
                               padding='SAME')

    with tf.variable_scope('layer2-conv2'):
        conv2_weights = tf.get_variable("weights"
                                        , [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP]
                                        , initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))

        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # pool_shape = pool2.get_shape().as_list()
        # print("pool_shape: ", "pool_shape[0]: ", pool_shape[0], "pool_shape[1]: ", pool_shape[1], "pool_shape[2]: ", pool_shape[2], "pool_shape[3]: ", pool_shape[3])
        # nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
        # print("nodes: ", nodes)
        # reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

        # just verify P2 = tf.contrib.layers.flatten(P2) is same with reshaped
        flatten_ret = tf.contrib.layers.flatten(pool2)
        print("flatten: ", flatten_ret)


    # FC Layer
    with tf.variable_scope('layer3-fc1'):
        fc1_weights = tf.get_variable("weights", [flatten_ret.shape[1], FC_SIZE], \
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(flatten_ret, fc1_weights) + fc1_biases)

        # fc1_ret = tf.contrib.layers.fully_connected(flatten_ret, num_outputs=FC_SIZE\
        # 											, weights_regularizer = regularizer\
        # 											, biases_initializer=tf.constant_initializer(0.1))
        # if if_train:
        #     fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.variable_scope('layer4-fc2'):
        # fc2_ret = tf.contrib.layers.fully_connected(fc1_ret, num_outputs=OUTPUT_NODE\
        # 											, weights_regularizer = regularizer\
        #  											, biases_initializer=tf.constant_initializer(0.1)\
        #  											, activation_fn = None)
        fc2_weights = tf.get_variable("weights", [FC_SIZE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [OUTPUT_NODE]
                                     , initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logits

# def create_placeholder(n_H, n_W, n_C, n_Classes):
# 	x = tf.placeholder(tf.float32, shape=(None, n_H, n_W, n_C))
# 	#y = tf.placeholder(tf.float32, shape=(None, n_Classes))

# 	return x#, y

# tf.reset_default_graph()

# with tf.Session() as sess:
# 	x_input =create_placeholder(32, 32, 1, 6)
# 	logits = inference(x_input, False, None)

# 	init = tf.global_variables_initializer()
# 	sess.run(init)
# 	sess.run(logits, {x_input: np.random.randn(4, 32, 32, 1)})
