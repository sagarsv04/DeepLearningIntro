
from mido import MidiFile, MidiTrack, Message
import mido
import os
import sys



def check_file_exist(file_path):
	try:
		if os.path.exists(file_path):
			return True
		else:
			return False
	except Exception as e:
		return False


def play_midi(file_path):
	midi_obj = MidiFile(file_path)
	print("Playing file.")
	# create a port to send audio messages
	port = mido.open_output()
	# read messages and send it to port
	for msg in midi_obj.play():
		port.send(msg)

	print("Done Playing file.")
	return 0


def main():

	if len(sys.argv)>1:
		arg = sys.argv[1]
		if check_file_exist(arg):
			play_midi(arg)
		else:
			print("Invalid path provided.")
	else:
		print("Please provide the file path.")

	return 0

if __name__ == '__main__':
	main()
