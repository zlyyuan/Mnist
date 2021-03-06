import time
import tensorflow as tf 
import mnist_train
import mnist_inference

EVAL_INTERVAL_SECS = 120


def evaluate(x_test, y_test):
	with tf.Graph().as_default() as g:
		input_x_flatten_size = x_test.shape[1]
		x = tf.placeholder(tf.float32,  shape = (None, input_x_flatten_size), name = 'x-input')
		y_ = tf.placeholder(tf.float32, shape = (None, mnist_inference.OUTPUT_NODE), name = 'y-input')

		validate_feed = {x: x_test, y_: y_test}
		y = mnist_inference.inference(x, None, None)

		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
		variables_to_restore =variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		while True:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path: 
					saver.restore(sess, ckpt.model_checkpoint_path)
					print("current ckpt path: ", ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					print("global_step: ", global_step)
					accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
					print( "After %s training steps, validation accuracy = %g " % (global_step, accuracy_score))
				else: 
					print("No checkpoint file found ")
					return
			time.sleep(EVAL_INTERVAL_SECS)	

def main(argv=None):
	train_set, test_set = mnist_train.load_data()
	x_test, y_test = mnist_train.preprocess_data(test_set)
	evaluate(x_test, y_test)
if __name__ == '__main__': 
	tf.app.run()