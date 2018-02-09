import numpy as np
import midi


def start():
	pattern = midi.Pattern()
	# Instantiate a MIDI Track (contains a list of MIDI events)
	track = midi.Track()
	# Append the track to the pattern
	pattern.append(track)
	# Instantiate a MIDI note on event, append it to the track
	on = midi.NoteOnEvent(tick=0, velocity=20, pitch=midi.G_3)
	track.append(on)
	# Instantiate a MIDI note off event, append it to the track
	off = midi.NoteOffEvent(tick=100, pitch=midi.G_3)
	track.append(off)
	# Add the end of track event, append it to the track
	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)
	# Print out the pattern
	print (pattern)
	# Save the pattern to disk
	midi.write_midifile("ex.mid", pattern)

def contents(file):
	return midi.read_midifile(file)

def main():
	file = 'data/Taylor Swift - Shake It Off.mid'
	start()
	print (contents(file))


if __name__ == '__main__':
	main()