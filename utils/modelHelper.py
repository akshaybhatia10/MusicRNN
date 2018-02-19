import numpy as np
import sys
from utils.extractFeatures import get_songs

def generate_batches(all_songs, batch_size, steps, input_dim, output_dim):
	r = np.random.randint(len(all_songs), size=batch_size)
	inputs = np.zeros((batch_size, steps, input_dim))
	targets = np.zeros((batch_size, output_dim))
	for i in range(batch_size):
		idx = r[i]
		start = np.random.randint(all_songs[idx].shape[0] - steps - 1)
		inputs[i] = all_songs[idx][start:start+steps]
		targets[i] = all_songs[idx][start+steps]
	return inputs, targets	


def logger(i, total, d=1, l=50, s=''):
	logs = '{0:.' + str(d) + 'f}'
	percent = logs.format(100 * (i/float(total)))
	completed = int(round(l*i/float(total)))
	saver = '=' * completed + '-' * (l - completed)
	sys.stdout.write('\r{} |{}| {} {} {}'.format('', saver, percent, '%', s))
	if i == total:
		sys.stdout.write('\n')
	sys.stdout.flush()

def sample_music(sample, output_size, new_songs):
	new = np.zeros(output_size)
	i = np.random.choice(range(output_size), p=sample[0])
	new[i] = 1
	new_songs.append(new)
	return new_songs
