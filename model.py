import tensorflow as tf
import numpy as np

def model_placeholders(input_size, output_size):
	inputs = tf.placeholder(tf.float32, shape=(None, timestamp, input_size), name='input_placeholder')
	targets = tf.placeholder(tf.float32, shape=(None, output_size), name='output_placeholder')
	keep_probability = tf.placeholder(tf.float32, name="keep_prob")
	learning_rate = tf.placeholder(tf.float32, name='learning_rate')

	return inputs, targets, keep_probability, learning_rate

def model_parameters(hidden_size):
	w1 = tf.Variable(tf.random_normal([hidden_size, output_size]))
	b1 = tf.Variable(tf.random_normal([hidden_size, 1]))

	return w1, b1



