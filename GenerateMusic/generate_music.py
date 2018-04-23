import sys
import os
from mido import MidiFile, MidiTrack, Message
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mido

# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\GenerateMusic')


# NOTE :: Song genrated havs no sound, Hence going to siraj's files



midi_file_path = './midi/allegroconspirito.mid'
midi_out_base = './deep_music_with_'

# model settings
max_len = 30
batch_size = 300
epoch = 20
n_values = 3


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


def main():
	# arg = '3000'
	if len(sys.argv)>1:
		arg = sys.argv[1]
		if CheckIfInt(arg):
			arg = int(arg)
			generate_music(arg)
		else:
			print("Invalid argumet type encountered")
	else:
		print("Please pass number of predictions.")


	return 0



if __name__ == '__main__':
	main()
