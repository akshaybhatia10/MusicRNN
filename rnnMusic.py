#!/usr/bin/env python
"""
Generate Music using RNN
"""
import tensorflow as tf
import numpy as np
import midi
import sys

from utils.extractFeatures import get_songs
from utils.modelHelper import logger, generate_batches
from utils.preprocessMidi import sample_midi, convert_midi
from model import model_placeholders, model_parameters, rnn_layer, get_loss, get_optimizer, get_accuracy

from distutils.version import LooseVersion
import warnings

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.4.1'), 'Update to TF 1.4.1 or greater. Your version {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found.')
else:
    print('Your GPU Device: {}'.format(tf.test.gpu_device_name()))


def main():
	if len(sys.argv) != 7:
		print ("Usage: {0} <data directory> <hidden layer size> <min song length> <steps> <epochs> <batch_size>".format(sys.argv[0]))
		exit(2)

	path = sys.argv[1]
	hidden_size = int(sys.argv[2])
	min_len = int(sys.argv[3])
	steps = int(sys.argv[4])
	epochs = int(sys.argv[5])
	batch_size = int(sys.argv[6])

	all_songs = get_songs(path)
	print ('Preprocessed Songs')
	total_songs = len(all_songs)
	input_size = all_songs[0].shape[1]
	output_size = input_size
	rnn_units = hidden_size
	learning_rate = 0.001
	keep_probability = 0.6
	disp = 1
	print (total_songs, input_size)
	print (all_songs[0].shape)

	model_inputs, model_targets, keep_prob, lr = model_placeholders(input_size, output_size, steps)
	parameters = model_parameters(output_size, hidden_size)  #w1, b1
	final_outputs, prediction = rnn_layer(model_inputs, parameters, rnn_units, keep_prob, steps)
	loss = get_loss(final_outputs, model_targets)
	optimizer = get_optimizer(loss, lr)
	accuracy = get_accuracy(model_targets, prediction)

	init = tf.global_variables_initializer()
	session = tf.Session()

	print ('Start Training')
	with session as sess:
		sess.run(init)
		for epoch in range(epochs):
			inputs, targets = generate_batches(all_songs, batch_size, steps, input_size, output_size)
			feed_dict = {model_inputs: inputs, model_targets: targets, keep_prob: keep_probability, lr: learning_rate}
			sess.run(optimizer, feed_dict=feed_dict)

			if epoch % disp == 0 or epoch == 10:
				l, a = sess.run([loss, accuracy], feed_dict=feed_dict) 
				s = 'Epoch: {}, Loss: {:.4f}, Accuracy: {:.3f} \n'.format(epoch, l, a) 

				logger(epoch, epochs, s=s)

	# Generate new midi files
		get_random = False
		idx = 11 if get_random else np.random.randint(total_songs) 
		song = all_songs[idx][:steps].tolist()

		print ('Sampling new music')
		for i in range(100):

			initial = np.array([song[-steps]])
			sample = sess.run(prediction, feed_dict={model_inputs, initial})
			new_songs = sample_music(sample, output_size, song)			

		sample_midi(new_songs, name='gen_1')	
		sample_midi(all_songs[idx], name='base_1')


if __name__ == '__main__':
	main()