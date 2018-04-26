
import pickle
from tqdm import tqdm
from collections import Counter
# import keras
import os
import copy


# download the data from below link
# http://mlg.ucd.ie/datasets/bbc.html


# os.getcwd()
# os.chdir(r'E:\to_be_deleted\DeepLearningIntro\TextSummarizer')


data_dir_base = "./bbc/"
data_pickle_path = "./bbc/bbc_data.pickle"


def check_file_exist(path):
	if os.path.isfile(path):
		return True
	else :
		return False


def get_list_dir(path):
	list_dir = os.listdir(path)
	return list_dir


def get_files_path(data_dir_base):

	files_path = []
	folder_list = get_list_dir(data_dir_base)
	for folder in folder_list:
		# folder = folder_list[2]
		if len(folder.split('.')) == 1:
			file_list = get_list_dir(data_dir_base+folder)
			for file in file_list:
				files_path.append(data_dir_base+folder+'/'+file)

	return files_path


def get_title_body_generator(files_path):

	for file in files_path:
		with open(file, 'r', encoding="ISO-8859-1") as f:
			content = f.read()
		content = content.split('\n\n')
		title = content[0]
		body = ""
		for line in content[1:]:
			body = body + line + '\n'
		yield title, body


def create_pickle_file(data_pickle_path):

	if not check_file_exist(data_pickle_path):
		files_path = get_files_path(data_dir_base)
		data_generator = get_title_body_generator(files_path)
		print("Creating Pickle file at: {0}".format(data_pickle_path))
		is_empty = True
		for data in tqdm(data_generator):
			with open(data_pickle_path, 'ab') as handle:
				pickle.dump(data, handle)
			# print(data)
			# print('\n\n')
			is_empty = False

		if is_empty:
			print("Empty Generator Found!")

	else:
		print("Pickle file {0} already exist.".format(data_pickle_path))

	return data_pickle_path


def load_data_from_pickle(pickle_obj):
	try:
		while True:
			yield pickle.load(pickle_obj)
	except EOFError:
		print("Out of data in pickle file.")


def read_pickle_file(pickle_file_path):

	# f = open(pickle_file_path, 'rb')
	# heads, desc = pickle.load(f)
	# with open(pickle_file_path, 'rb') as f:
		# i = 0
		# for event in load_data_from_pickle(f):
			# i+=i+1
			# print(i)
			# print('\n')
	return 0


def create_pickle(data_dir_base):

	pickle_file_path = create_pickle_file(data_pickle_path)
	read_pickle_file(pickle_file_path)

	return 0


def main():

	create_pickle(data_dir_base)

	return 0


if __name__ == '__main__':
	main()
