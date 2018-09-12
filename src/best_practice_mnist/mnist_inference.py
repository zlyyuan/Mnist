import tensorflow as tf

INPUT_NODE = 784
LAYER_1_NODE = 500
OUTPUT_NODE = 10

def get_weights_variable(shape, regularizer):
	weights = tf.get_variable("weights", shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))

	if regularizer != None:
		tf.add_to_collection('lossess', regularizer(weights))
	return weights


def inference( input_tensor, regularizer):

	with tf.variable_scope('layer1'):
		weights = get_weights_variable([INPUT_NODE, LAYER_1_NODE], regularizer)
		biases = tf.get_variable("biases", [LAYER_1_NODE], initializer=tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, average_wights(avg_class, weights)) + biases)
	with tf.variable_scope('layer2'):
		weights = get_weights_variable([LAYER_1_NODE, OUTPUT_NODE], regularizer)
		biases = tf.get_variable("biases", [OUTPUT_NODE], initializer = tf.constant_initializer(0.0))
		layer2=tf.matmul(layer1, average_wights(avg_class, weights))+biases

	return layer2