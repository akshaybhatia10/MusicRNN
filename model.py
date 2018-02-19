import tensorflow as tf
import numpy as np

def model_placeholders(input_size, output_size, steps=64):
	model_inputs = tf.placeholder(tf.float32, shape=(None, steps, input_size), name='input_placeholder')
	model_targets = tf.placeholder(tf.float32, shape=(None, output_size), name='output_placeholder')
	keep_prob = tf.placeholder(tf.float32, name="keep_prob")
	lr = tf.placeholder(tf.float32, name='learning_rate')

	return model_inputs, model_targets, keep_prob, lr


def model_parameters(output_size, hidden_size=128):
	w1 = tf.Variable(tf.random_normal([hidden_size, output_size]))
	b1 = tf.Variable(tf.random_normal([output_size]))

	return (w1, b1)


def rnn_layer(inputs, parameters, rnn_units, keep_prob, steps):
	inputs = tf.unstack(inputs, steps, 1)
	w1, b1 = parameters[0], parameters[1]
	basic_lstm = tf.contrib.rnn.BasicLSTMCell(rnn_units)
	#dropout = tf.contrib.rnn.DropoutWrapper(basic_lstm, keep_prob)
	outputs, state = tf.contrib.rnn.static_rnn(basic_lstm, inputs, dtype=tf.float32)

	final_outputs = tf.matmul(outputs[-1], w1) + b1
	prediction = tf.nn.softmax(final_outputs)

	return final_outputs, prediction


def get_loss(final_outputs, targets):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_outputs, labels=targets))		

	return loss


def get_optimizer(loss, learning_rate):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	return optimizer


def get_accuracy(targets, prediction):
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(targets, 1))
	acc = tf.reduce_mean(tf.cast(correct, tf.float32))

	return acc