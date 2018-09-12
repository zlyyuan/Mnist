import os
import tensorflow as tf
import mnist_inference

MNIST_PATH = "/Users/liyuanzhao/Project/DeepLearning/mnist/dataset/mnist.npz"
MODEL_SAVE_PATH = "/Users/liyuanzhao/Project/DeepLearning/mnist/src/best_practice_mnist/mdel/"
MODEL_NAME = "mnist_model.ckpt"

def load_data():
	f = np.load(MNIST_PATH)
	x_train, y_train = f['x_train'], f['y_train']
	x_test, y_test = f['x_test'], f['y_test']
	f.close()
	return (x_train/255, y_train), (x_test/255, y_test)

def preprocess_data():
	(x_train, y_train_orig), (x_test, y_test_orig) = load_data()

	print("Original Train data X shape: ", x_train.shape, "Training data Y shape: ", y_train_orig.shape)
	print("Original Test data X shape: ", x_test.shape,"Test data Y shape: ", y_test_orig.shape)
	#print("before convert y_train_orig :", y_train_orig.shape, "y[0]: ", y_train_orig[0], "y[1]: ", y_train_orig[1],"y[2]: ", y_train_orig[2],)
	y_train = convert_to_one_hot(y_train_orig, 10)
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


def train():

	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32,  shape = (None, input_x_flatten_size), name = 'x-input')
		y_ = tf.placeholder(tf.float32, shape = (None, OUTPUT_NODE), name = 'y-input')

	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE) 
	y = mnist_inference.inference(x, regularizer)

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

	with tf.name_scope('train_step'):
		# learning rate decay
		learning_rate = tf.train.exponential_decay( LEARNING_RATE_BASE, global_step, input_x_number_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
		# Note as from https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
		# global_step: Optional Variable to increment by one after the variables have been updated.
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)

		with tf.control_dependencies([train_step, variabl_averages_op]):
			train_op = tf.no_op(name = 'train')

	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		seed = 3
		mini_batches = random_mini_batches(tf.transpose(x_reshape_train), tf.transpose(y_reshape_train), BATCH_SIZE, seed)
		for i in range(TRAINING_STEPS): 

			if  (i%len(mini_batches)) == 0:
				seed = seed +1
				mini_batches = random_mini_batches(tf.transpose(x_reshape_train), tf.transpose(y_reshape_train), BATCH_SIZE, seed)
			
			mini_x_batches, mini_y_batches = mini_batches[k]
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x: mini_x_batches, y_: mini_y_batches})


			if (i%1000) == 0:
				print("After %d training step(s), loss on training batch is %g" % (step, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

def main(argv=None):
	load_data()
	preprocess_data()
	train()

if __name__ == "__main__":
	tf.app.run()