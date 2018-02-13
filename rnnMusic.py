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
from model import *

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
	if len(sys.argv) != 2:
		print ("Usage: {0} <data directory> <min_len>".format(sys.argv[0]))
		exit(2)

	path = sys.argv[1]

	#all_songs = get_songs(path)

if __name__ == '__main__':
	main()