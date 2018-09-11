import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tf_utils import random_mini_batches, convert_to_one_hot

OUTPUT_NODE = 10 # 10 classes
LAYER1_NODE = 500
INPUT_NODE = 784
def load_data(path="dataset/data/mnist.npz"):
	f = np.load(path)
	x_train, y_train = f['x_train'], f['y_train']
	x_test, y_test = f['x_test'], f['y_test']
	f.close()
	return (x_train/255, y_train), (x_test/255, y_test)

def get_weights_variable(shape, regularizer):
	weights = tf.get_variable("weights", shape, initializer = tf.truncated_normal_initializer(stddev = 0.1))

	if regularizer != None:
		tf.add_to_collection('lossess', regularizer(weights))
	return weights

def average_wights(avg_class, weights):
	if avg_class == None :
		return weights
	else: 
		return avg_class.average(weights)


def inference( input_tensor, regularizer, avg_class, i_reuse=False):

	with tf.variable_scope('layer1', reuse=i_reuse):
		# shape = (input_size, LAYER1_NODE)
		# print("--- shape :", shape.dtype)
		weights = get_weights_variable([INPUT_NODE, LAYER1_NODE], regularizer)
		biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, average_wights(avg_class, weights)) + biases)
	with tf.variable_scope('layer2', reuse=i_reuse):
		weights = get_weights_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
		biases = tf.get_variable("biases", [OUTPUT_NODE], initializer = tf.constant_initializer(0.0))
		layer2=tf.nn.relu(tf.matmul(layer1, average_wights(avg_class, weights))+biases)

	return layer2

def train():

	BATCH_SIZE = 100

	LEARNING_RATE_BASE = 0.8
	LEARNING_RATE_DECAY = 0.99

	REGULARIZATION_RATE = 0.0001
	TRAINING_STEPS = 30000
	MOVING_AVERAGE_DECAY = 0.99


	(x_train, y_train_orig), (x_test, y_test_orig) = load_data()

	print("Original Train data X shape: ", x_train.shape, "Training data Y shape: ", y_train_orig.shape)
	print("Original Test data X shape: ", x_test.shape,"Test data Y shape: ", y_test_orig.shape)
	#print("before convert y_train_orig :", y_train_orig.shape, "y[0]: ", y_train_orig[0], "y[1]: ", y_train_orig[1],"y[2]: ", y_train_orig[2],)
	y_train = convert_to_one_hot(y_train_orig, 10)
	#print("after convert y_train_one_hot :",y_train.shape)
	y_test = convert_to_one_hot(y_test_orig, 10)

	print("----- Reshape Original Trains Dataset Shape ------")
	x_reshape_train = tf.reshape(x_train, [x_train.shape[0], -1])
	y_reshape_train = tf.transpose(y_train)
	
	print("Reshape Train data X as: ", x_reshape_train.shape, "Training data Y shape: ", y_reshape_train.shape)
	x_test_reshape = tf.reshape(x_test, [x_test.shape[0], -1])
	y_test_reshape = tf.transpose(y_test)
	print("Reshape Test data X as: ", x_test_reshape.shape, "Test data Y shape: ", y_test_reshape.shape)

	input_x_flatten_size = x_reshape_train.shape[1]
	input_x_size = tf.convert_to_tensor(input_x_flatten_size, dtype = tf.int32)
	input_x_number_examples = tf.convert_to_tensor(x_reshape_train.shape[0], dtype=tf.int32)

	x = tf.placeholder(tf.float32,  shape = (None, input_x_flatten_size), name = 'x-input')
	y_ = tf.placeholder(tf.float32, shape = (None, OUTPUT_NODE), name = 'y-input')


	# weights1 = tf.Variable(tf.truncated_normal([input_x_size, LAYER1_NODE], stddev = 0.1), name = "weights1")
	# biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]), name="biases1")

	# weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev = 0.1), name="weights2")
	# biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]), name = "biases2")

	# Forward propagation result
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	y = inference(x, regularizer, None)

	# Step of training number
	global_step = tf.Variable(0, trainable=False)

	variabl_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variabl_averages_op = variabl_averages.apply(tf.trainable_variables())
	#print(tf.trainable_variables())
	# Forward propagation using sliding average
	
	average_y = inference(x, regularizer, variabl_averages, True)

	# loss function
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	# regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	# regularization = regularizer(weights1) + regularizer(weights2)
	# Note using 'tf.add_n(tf.getcollection('lossess'))' replace original regularization method

	loss = cross_entropy_mean + tf.add_n(tf.get_collection('lossess'))
	# learning rate decay
	learning_rate = tf.train.exponential_decay( LEARNING_RATE_BASE, global_step, input_x_number_examples/BATCH_SIZE, LEARNING_RATE_DECAY)

	# Note as from https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
	# global_step: Optional Variable to increment by one after the variables have been updated.
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

	with tf.control_dependencies([train_step, variabl_averages_op]):
		train_op = tf.no_op(name = 'train')

	correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		


		# validation data sets
		validates_x = x_test_reshape.eval()
		validates_y = y_test_reshape.eval()
		validates_feed = {x: validates_x, y_: validates_y}
		
		# test data sets
		test_feed = {x: x_reshape_train.eval(), y_: y_reshape_train.eval()}
		seed = 3
		mini_batches = random_mini_batches(tf.transpose(x_reshape_train), tf.transpose(y_reshape_train), BATCH_SIZE, seed)
		
		for i in range(TRAINING_STEPS): 

			if i%1000 == 0:
				validate_acc = sess.run(accuracy, feed_dict = validates_feed)
				print("After %d training step(s), validation accuracy using average model is %g" % (i, validate_acc))
			
			k = i%len(mini_batches)
			if  k == 0:
				seed = seed +1
				mini_batches = random_mini_batches(tf.transpose(x_reshape_train), tf.transpose(y_reshape_train), BATCH_SIZE, seed)
			
			mini_x_batches, mini_y_batches = mini_batches[k]

			
	#print("----cross_mean shape: ", cross_entropy_mean.shape, " collection shape:", tf.add_n(tf.get_collection('losses')).shape )
	
			sess.run(train_op, feed_dict = {x: mini_x_batches, y_: mini_y_batches})

		test_acc = sess.run(accuracy, feed_dict = test_feed)
		print("After %d training step(s), testing accuracy using average model is %g" % (i, validate_acc))



if __name__ == "__main__":
	train()	