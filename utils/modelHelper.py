import numpy as np
import sys
from extractFeatures import get_songs

def generate_batches(all_songs, batch_size, diff, input_dim, output_dim):
	r = np.random.randint(len(all_songs), size=batch_size)
	inputs = np.zeros((batch_size, diff, input_dim))
	targets = np.zeros((batch_size, output_dim))
	for i in range(batch_size):
		idx = r[i]
		start = np.random.randint(all_songs[idx].shape[0] - diff - 1)
		inputs[i] = all_songs[idx][start:start+diff]
		targets[i] = all_songs[idx:diff]
	return inputs, targets	


def log(i, total, d=1, l=50):
	s = '{0:.' + str(d) + 'f}'
	percent = s.format(100 * (i/float(total)))
	completed = int(round(l*i/float(total)))
	saver = '=' * completed + '-' * (l - completed)
	sys.stdout.write('\r{} |{}| {} {} {}'.format('', saver, percent, '%', ''))
	if i == total:
		sys.stdout.write('\n')
	sys.stdout.flush()	
	

def main():
	dir = 'data/*.mid'
	#all_songs = get_songs(dir)
	#x, y = generate_batches(all_songs, 32, 5, 32, 32)
	#print (x, y)
	log(10, 10)

if __name__ == '__main__':
	main()		 