
import pickle as pickle
from collections import Counter


# download dataset from
# http://opus.nlpl.eu/OpenSubtitles.php
# TMX/Moses Downloads the txt version of the files


language_one_path, language_one = "./OpenSubtitles.en-es.en", "English"
language_two_path, language_two = "./OpenSubtitles.en-es.es", "Spanish"

limit_vocab = True
limit_percentage = 30


def read_sentences(file_path, language):
	# file_path = language_one_path
	sentences = []
	with open(file_path, "r", encoding="utf-8") as reader:
		for s in reader:
			sentences.append(s.strip())
	print("Language: {0}, Length: {1}".format(language, len(sentences)))
	return sentences


def create_dataset(en_sentences, es_sentences):

	en_vocab_dict = Counter(word.strip(',." ;:)(][?!') for sentence in en_sentences for word in sentence.split())
	es_vocab_dict = Counter(word.strip(',." ;:)(][?!') for sentence in es_sentences for word in sentence.split())

	en_vocab = list(map(lambda x: x[0], sorted(en_vocab_dict.items(), key = lambda x: -x[1])))
	es_vocab = list(map(lambda x: x[0], sorted(es_vocab_dict.items(), key = lambda x: -x[1])))

	if limit_vocab:
		en_vocab = en_vocab[:int(limit_percentage/100*len(en_vocab))]
		es_vocab = es_vocab[:int(limit_percentage/100*len(es_vocab))]

	start_idx = 2
	en_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(en_vocab)])
	en_word2idx['<ukn>'] = 0
	en_word2idx['<pad>'] = 1

	en_idx2word = dict([(idx, word) for word, idx in en_word2idx.items()])

	start_idx = 4
	es_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(es_vocab)])
	es_word2idx['<ukn>'] = 0
	es_word2idx['<go>']  = 1
	es_word2idx['<eos>'] = 2
	es_word2idx['<pad>'] = 3

	es_idx2word = dict([(idx, word) for word, idx in es_word2idx.items()])

	x = [[en_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.split()] for sentence in en_sentences]
	y = [[es_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.split()] for sentence in es_sentences]

	X = []
	Y = []
	for i in range(len(x)):
		# i = 0
		n1 = len(x[i])
		n2 = len(y[i])
		n = n1 if n1 < n2 else n2
		if abs(n1 - n2) <= 0.3 * n:
			if n1 <= 15 and n2 <= 15:
				X.append(x[i])
				Y.append(y[i])

	return X, Y, en_word2idx, en_idx2word, en_vocab, es_word2idx, es_idx2word, es_vocab

def save_dataset(file_path, obj):
	with open(file_path, 'wb') as f:
		pickle.dump(obj, f)
	print("Saved data to path: {0}".format(file_path))

def read_dataset(file_path):
	with open(file_path, 'rb') as f:
		return pickle.load(f)

def main():
	en_sentences = read_sentences(language_one_path, language_one)
	es_sentences = read_sentences(language_two_path, language_two)

	save_dataset('./data_set.pkl', create_dataset(en_sentences, es_sentences))

if __name__ == '__main__':
	main()
