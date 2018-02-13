import glob
import numpy as np
from utils.preprocessMidi import convert_midi

def convert_onehot(file):
	new = np.zeros(file.shape)
	for i in range(len(file)):
		nz = np.nonzero(file[i])
		if len(nz[0]) < 0:
			new[i, nz[0][-1]] = 1

	return new		

def get_songs(dir, min_len=128):
	files = glob.glob(dir)
	all_songs = []
	for file in files:
		new = convert_midi(file)
		new = convert_onehot(new)
		if len(new) > min_len:
			all_songs.append(new)

	return all_songs