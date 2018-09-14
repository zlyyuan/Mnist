import os
import tensorflow as tf
import mnist_inference
import numpy as np
from tf_utils import random_mini_batches, convert_to_one_hot

MNIST_PATH = "/Users/liyuanzhao/Project/DeepLearning/mnist/dataset/mnist.npz"
MODEL_SAVE_PATH = "/Users/liyuanzhao/Project/DeepLearning/mnist/src/best_practice_mnist/model/"
MODEL_NAME = "mnist_model.ckpt"

OUTPUT_NODE = 10 # 10 classes
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
	return (x_train/255, y_train), (x_test/255, y_test)

def preprocess_data(data_set):
	x, y = data_set 
	print("Original data X shape: ", x.shape, " data Y shape: ", y.shape)

	x = x.reshape(x.shape[0], -1)
	y_one_hot = convert_to_one_hot(y, 10)
	y = y_one_hot.transpose()
	print("----- Reshape Original Trains Dataset Shape ------")
	print("Reshape data X as: ", x.shape, " data Y shape: ", y.shape)

	# x_reshape_train = tf.reshape(x_train, [x_train.shape[0], -1])
	# y_reshape_train = tf.transpose(y_train)
	
	
	# x_test_reshape = tf.reshape(x_test, [x_test.shape[0], -1])
	# y_test_reshape = tf.transpose(y_test)
	# x_test = x_test.reshape(x_test.shape[0], -1)
	# y_test = y_test.transpose()
	# print("Reshape Test data X as: ", x_test.shape, "Test data Y shape: ", y_test.shape)


	# input_x_size = tf.convert_to_tensor(input_x_flatten_size, dtype = tf.int32)
	# input_x_number_examples = tf.convert_to_tensor(x_train.shape[0], dtype=tf.int32)

	return (x, y)#, (x_test, y_test)

def train(x_train, y_train):

	input_x_flatten_size = x_train.shape[1]
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32,  shape = (None, input_x_flatten_size), name = 'x-input')
		y_ = tf.placeholder(tf.float32, shape = (None, OUTPUT_NODE), name = 'y-input')

	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE) 
	y = mnist_inference.inference(x, regularizer,None)

	# Step of training number
	global_step = tf.Variable(0, trainable=False)
	with tf.name_scope('moving_average'):
		variabl_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
		variabl_averages_op = variabl_averages.apply(tf.trainable_variables())

	with tf.name_scope('loss_function'):
		# loss function
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
		loss = cross_entropy_mean + tf.add_n(tf.get_collection('lossess'))

	input_x_number_examples = x_train.shape[0]
	with tf.name_scope('train_step'):
		# learning rate decay
		learning_rate = tf.train.exponential_decay( LEARNING_RATE_BASE, global_step, input_x_number_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
		# Note as from https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
		# global_step: Optional Variable to increment by one after the variables have been updated.
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

		with tf.control_dependencies([train_step, variabl_averages_op]):
			train_op = tf.no_op(name = 'train')

	saver = tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		seed = 3
		mini_batches = random_mini_batches(tf.transpose(x_train), tf.transpose(y_train), BATCH_SIZE, seed)
		for i in range(TRAINING_STEPS):
			k = i%len(mini_batches)
			if  k == 0:
				seed = seed +1
				mini_batches = random_mini_batches(tf.transpose(x_train), tf.transpose(y_train), BATCH_SIZE, seed)
			mini_x_batches, mini_y_batches = mini_batches[k]
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x: mini_x_batches, y_: mini_y_batches})
			if (i%1000) == 0:
				print("After %d training step(s), loss on training batch is %g" % (step, loss_value))
				print("----- global_step: ", sess.run(global_step))
				if(step == 1 ): 
					saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)
				else:
					saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step, write_meta_graph = False)

def main(argv=None):
	train_set, test_set = load_data()
	x_train, y_train = preprocess_data(train_set)
	train(x_train, y_train)

if __name__ == "__main__":
	tf.app.run()