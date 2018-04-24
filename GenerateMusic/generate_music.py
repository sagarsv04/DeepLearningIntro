import sys
import os
import numpy as np

# from mido import MidiFile, MidiTrack, Message
# from keras.layers import LSTM, Dense, Activation, Dropout
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras import optimizers
# from sklearn.preprocessing import MinMaxScaler
# import mido

from music21 import *
from grammar import *
from preprocess import *
from quality_assurance import *
import lstm

# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\GenerateMusic')

# NOTE :: Song genrated method "generate_music" havs no sound, Hence going to siraj's files

midi_file_path = './midi/original_metheny.mid'
midi_out_base = './deep_music_with_'

# model settings
max_len = 30
batch_size = 300
epoch = 20
n_values = 3
max_tries = 1000
diversity = 0.5

# musical settings
bpm = 130


def CheckIfInt(str):
	try:
		int(str)
		return True
	except ValueError:
		return False


def play_midi_file(file_path):
	midi_obj = MidiFile(file_path)
	print("Playing file.")
	# create a port to send audio messages
	port = mido.open_output()
	# read messages and send it to port
	for msg in midi_obj.play():
		port.send(msg)

	print("Done Playing file.")
	return 0


def process_midi_file(file_path):
	# file_path = midi_file_path
	mid = MidiFile(file_path)

	notes = []
	time = float(0)
	prev = float(0)

	for msg in mid:
		# Mesages Eg: <message note_on channel=5 note=65 velocity=76 time=0.12276779166666667>,
		#			  <message note_on channel=9 note=44 velocity=66 time=0.005580354166666667>,
		# this time is in seconds, not ticks
		time += msg.time
		if not msg.is_meta:
			# only interested in piano channel
			if msg.channel == 0:
				if msg.type == 'note_on':
					# note in vector form to train on
					note = msg.bytes()
					# only interested in the note and velocity. note message is in the form of [type, note, velocity]
					note = note[1:3]
					note.append(time-prev)
					prev = time
					notes.append(note)

	# scale data to be between 0, 1
	t = []
	for note in notes:
		note[0] = (note[0]-24)/88
		note[1] = note[1]/127
		t.append(note[2])
	max_t = max(t) # scale based on the biggest time of any note
	# for note in notes:
		# note[2] = note[2]/max_t
	return notes, max_t


def get_data_label(n_prev, notes):

	X = []
	Y = []
	# n_prev notes to predict the (n_prev+1)th note
	for i in range(len(notes)-n_prev):
		x = notes[i:i+n_prev]
		y = notes[i+n_prev]
		X.append(x)
		Y.append(y)

	return X, Y


def save_predicted_track(prediction, save_path):

	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)

	for note in prediction:
		# note = prediction[0]
		note = np.insert(note, 0, 147) # 147 means note_on
		bytes = note.astype(int)
		msg = Message.from_bytes(bytes[0:3])
		time = int(note[3]/0.001025) # to rescale to midi's delta ticks. arbitrary value for now.
		msg.time = time
		track.append(msg)

	mid.save(save_path)

	return 0


def build_lstm_model(max_len, n_values):

	print('Build model...')
	model = Sequential()
	model.add(LSTM(128, input_shape=(max_len, n_values), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(128, input_shape=(max_len, n_values), return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(n_values))
	model.add(Activation('linear'))
	model.compile(loss='mse', optimizer=optimizers.RMSprop(lr=0.01))

	return model


def generate_music(arg):

	input_path = midi_file_path
	output_path	= midi_out_base + str(arg) + '_prediction_on_' + input_path.split('/')[-1]
	# play_midi_file(input_path)
	notes, max_t = process_midi_file(input_path)
	X, Y = get_data_label(max_len, notes)
	# save a seed to do prediction later
	seed = notes[0:max_len]
	model = build_lstm_model(max_len, n_values)
	model.fit(np.array(X), np.array(Y), batch_size=batch_size, epochs=epoch, verbose=1)

	prediction = []
	x = seed
	x = np.expand_dims(x, axis=0)

	for i in range(arg):
	   preds = model.predict(x)
	   preds = np.absolute(preds)
	   x = np.squeeze(x)
	   x = np.concatenate((x, preds))
	   x = x[1:]
	   x = np.expand_dims(x, axis=0)
	   preds = np.squeeze(preds)
	   prediction.append(preds)

	save_predicted_track(prediction, output_path)

	# play_midi_file(output_path)

	return 0


#----------------------------HELPER FUNCTIONS----------------------------------#

''' Helper function to sample an index from a probability array '''
def __sample(a, temperature=1.0):
	a = np.log(a) / temperature
	a = np.exp(a) / np.sum(np.exp(a))
	return np.argmax(np.random.multinomial(1, a, 1))

''' Helper function to generate a predicted value from a given matrix '''
def __predict(model, x, indices_val, diversity):
	preds = model.predict(x, verbose=0)[0]
	next_index = __sample(preds, diversity)
	next_val = indices_val[next_index]

	return next_val

''' Helper function which uses the given model to generate a grammar sequence
	from a given corpus, indices_val (mapping), abstract_grammars (list),
	and diversity floating point value. '''
def __generate_grammar(model, corpus, abstract_grammars, values, val_indices,
					   indices_val, max_len, max_tries, diversity):
	curr_grammar = ''
	# np.random.randint is exclusive to high
	start_index = np.random.randint(0, len(corpus) - max_len)
	sentence = corpus[start_index: start_index + max_len]    # seed
	running_length = 0.0
	while running_length <= 4.1:    # arbitrary, from avg in input file
		# transform sentence (previous sequence) to matrix
		x = np.zeros((1, max_len, len(values)))
		for t, val in enumerate(sentence):
			if (not val in val_indices): print(val)
			x[0, t, val_indices[val]] = 1.

		next_val = __predict(model, x, indices_val, diversity)

		# fix first note: must not have < > and not be a rest
		if (running_length < 0.00001):
			tries = 0
			while (next_val.split(',')[0] == 'R' or
				len(next_val.split(',')) != 2):
				# give up after 1000 tries; random from input's first notes
				if tries >= max_tries:
					print('Gave up on first note generation after', max_tries,
						'tries')
					# np.random is exclusive to high
					rand = np.random.randint(0, len(abstract_grammars))
					next_val = abstract_grammars[rand].split(' ')[0]
				else:
					next_val = __predict(model, x, indices_val, diversity)

				tries += 1

		# shift sentence over with new value
		sentence = sentence[1:]
		sentence.append(next_val)

		# except for first case, add a ' ' separator
		if (running_length > 0.00001): curr_grammar += ' '
		curr_grammar += next_val

		length = float(next_val.split(',')[1])
		running_length += length

	return curr_grammar


def music_generator(arg):

	data_fn = midi_file_path
	out_fn =  midi_out_base + str(arg) + '_prediction_on_' + data_fn.split('/')[-1]
	N_epochs = arg

	# get data
	chords, abstract_grammars = get_musical_data(data_fn)
	corpus, values, val_indices, indices_val = get_corpus_data(abstract_grammars)
	print('corpus length:', len(corpus))
	print('total # of values:', len(values))

	# build model
	model = lstm.build_model(corpus=corpus, val_indices=val_indices,
							 max_len=max_len, N_epochs=N_epochs)

	# set up audio stream
	out_stream = stream.Stream()

	# generation loop
	curr_offset = 0.0
	loopEnd = len(chords)
	for loopIndex in range(1, loopEnd):
		# get chords from file
		curr_chords = stream.Voice()
		for j in chords[loopIndex]:
			curr_chords.insert((j.offset % 4), j)

		# generate grammar
		curr_grammar = __generate_grammar(model=model, corpus=corpus,
										  abstract_grammars=abstract_grammars,
										  values=values, val_indices=val_indices,
										  indices_val=indices_val,
										  max_len=max_len, max_tries=max_tries,
										  diversity=diversity)

		curr_grammar = curr_grammar.replace(' A',' C').replace(' X',' C')

		# Pruning #1: smoothing measure
		curr_grammar = prune_grammar(curr_grammar)

		# Get notes from grammar and chords
		curr_notes = unparse_grammar(curr_grammar, curr_chords)

		# Pruning #2: removing repeated and too close together notes
		curr_notes = prune_notes(curr_notes)

		# quality assurance: clean up notes
		curr_notes = clean_up_notes(curr_notes)

		# print # of notes in curr_notes
		print('After pruning: %s notes' % (len([i for i in curr_notes
			if isinstance(i, note.Note)])))

		# insert into the output stream
		for m in curr_notes:
			out_stream.insert(curr_offset + m.offset, m)
		for mc in curr_chords:
			out_stream.insert(curr_offset + mc.offset, mc)

		curr_offset += 4.0

	out_stream.insert(0.0, tempo.MetronomeMark(number=bpm))

	# Play the final stream through output (see 'play' lambda function above)
	play = lambda x: midi.realtime.StreamPlayer(x).play()
	play(out_stream)

	# save stream
	mf = midi.translate.streamToMidiFile(out_stream)
	mf.open(out_fn, 'wb')
	mf.write()
	mf.close()

	return 0


def main():
	# arg = '3000'
	if len(sys.argv)>1:
		arg = sys.argv[1]
		if CheckIfInt(arg):
			arg = int(arg)
			# NOTE : Below method is WIP.
			# generate_music(arg)

			music_generator(arg) # The other method
		else:
			print("Invalid argumet type encountered")
	else:
		print("Please pass number of predictions.")


	return 0



if __name__ == '__main__':
	main()
