import librosa, numpy as np, librosa.display, helpers, vocal_separation, sys
from matplotlib import pyplot as plt

def generate_training_set():
	song_list = helpers.get_song_list('batch_main.csv')

	X = list()
	y = list()

	for song in song_list:
		f_words, non_f_words = separate_song(song['name'], song['timings'])
		non_f_words_sample = helpers.match_sizes(f_words, non_f_words)

		X.append(f_words)
		X.append(non_f_words_sample)

		y.append(np.ones((f_words.shape[0],)))
		y.append(np.zeros((non_f_words_sample.shape[0],)))

	X = np.concatenate(X, axis=0)
	y = np.concatenate(y, axis=0)
	y = y.reshape(y.shape[0], 1)

	y_bar = np.int32(y[:,0] == 0).reshape(y.shape[0], 1)

	y = np.append(y, y_bar, axis=1)

	return X, y

def separate_song(song_location, timings):
	chunks = load_chunks(helpers.MUSIC_DIR_CLEAN + song_location, 1)

	f_words = list()

	to_delete_obj = list()

	trim_percent = 0.2

	for timing in timings:
		chunk = chunks[timing:timing + 2]

		if trim_percent > 0:
			chunk = perform_trim(chunk, trim_percent)

		f_words.append(chunk)
		
		to_delete_obj.append(timing)
		to_delete_obj.append(timing + 1)

	chunks = np.delete(chunks, to_delete_obj, axis=0)

	if chunks.shape[0] % 2 != 0:
		chunks = np.delete(chunks, chunks.shape[0] - 1, axis=0)

	non_f_words = np.split(chunks, chunks.shape[0] / 2)

	non_f_words_list = list()

	if trim_percent > 0:
		for i in range(len(non_f_words)):
			non_f_words_list.append(perform_trim(non_f_words[i], trim_percent))

	return np.array(f_words), np.array(non_f_words_list)

def load_chunks(file_name, chunk_size_in_seconds):
	data, sr = librosa.load(file_name, sr=None, mono=True)

	chunk_size = chunk_size_in_seconds * sr
	to_delete = int(data.shape[0] % chunk_size)

	data = data[:data.shape[0] - to_delete]
	splits = data.shape[0] / chunk_size

	return np.array(np.split(data, splits))

def perform_trim(chunk, trim_percent):
	trim_amount = int(chunk.shape[1] * trim_percent)

	trim_indices = np.concatenate((np.arange(trim_amount), np.arange(chunk.shape[1] - trim_amount, chunk.shape[1])))

	return np.delete(chunk, trim_indices, 1)	

# X, y = generate_training_set()

# print(X.shape, y.shape)