import numpy as np
import midi

def start():
	pattern = midi.Pattern()
	track = midi.Track()
	pattern.append(track)
	on = midi.NoteOnEvent(tick=0, velocity=20, pitch=midi.G_3)
	track.append(on)
	off = midi.NoteOffEvent(tick=100, pitch=midi.G_3)
	track.append(off)
	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)
	print (pattern)
	midi.write_midifile("ex.mid", pattern)

def contents(file):
	return midi.read_midifile(file)

def convert_midi(file, low=24, high=102):
	states, time = [], 0
	test = True
	spectrum = high - low

	data = contents(file)
	p = [0 for note in data]
	total_time = [note[0].tick for note in data]

	state = [[0,0] for i in range(spectrum)]
	states.append(state)
	while test:
		if time % (data.resolution / 4) == (data.resolution / 8):
			previous = state
			state = [[previous[i][0],0] for i in range(spectrum)]
			states.append(state)
		for i in range(len(total_time)):
			if not test:
				break
			while total_time[i] == 0:
				current = data[i]
				pos = p[i]
				th = current[pos]

				if isinstance(th, midi.NoteEvent):
					if (th.pitch < low) or (th.pitch >= high):
						pass
					else:
						if isinstance(th, midi.NoteOffEvent) or (th.velocity == 0):
							state[th.pitch - low] = [0,0]
						else:
							state[th.pitch - low] = [1,1]	
				elif isinstance(th, midi.TimeSignatureEvent):
					if (th.numerator not in (2,4)):
						final = states
						test = False
						break

				try:
					total_time[i] = current[pos+1].tick
					p[i] += 1
				except IndexError:
					total_time[i] = None

			if (total_time[i] is not None):												 
				total_time[i] -= 1

		if all(t is None for t in total_time):
			break
		time += 1
	new = np.array(states)
	states = np.hstack((new[:,:,0], new[:,:,1]))
	states = np.asarray(states)
	ohe = np.array([1 if new[i][j].any() else 0 for i in range(len(new)) for j in range(len(new[i]))]).reshape((new.shape[0], new.shape[1]))				 
	return ohe

def sample_midi(states, low=24, high=102, filename='new_music'):
	scale, last_update = 55, 0
	data = midi.Pattern()
	current = midi.Track()
	data.append(current)

	spectrum = high - low

	states = np.array(states)
	previous = [0 for i in range(spectrum)]
	for t, state in enumerate(states + [previous[:]]):
		off, on = [], []
		for i in range(spectrum):
			n = state[i]
			p = previous[i]
			if (p == 1):
				if (n == 0):	 
					off.append(i)
			elif (n == 1):
				on.append(i)
		for note in off:
			current.append(midi.NoteOffEvent(tick=(t-last_update) * scale, pitch=note+low))	
			last_update = t
		for note in on:
			current.append(midi.NoteOnEvent(tick=(t-last_update)*scale, velocity=40, pitch=note+low))

		previous = state
		end = midi.EndOfTrackEvent(tick=1)
		current.append(end)
		midi.write_midifile('{}.mid'.format(filename), data)


def main():
	file = 'data/Taylor Swift - Shake It Off.mid'
	low = 24
	high = 102

	ohe = convert_midi(file, low, high)
	sample_midi(ohe, low, high, 'new_music')

if __name__ == '__main__':
	main()