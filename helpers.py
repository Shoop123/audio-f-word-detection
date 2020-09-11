import csv, librosa, vocal_separation
import os.path

BASE_DIR = 'song_downloader/'
MUSIC_DIR = BASE_DIR + 'songs/wav/'
MUSIC_DIR_CLEAN = BASE_DIR + 'songs/wav/clean/'
MTURK_DIR = BASE_DIR + 'mturk_files/'

def get_song_list(batch_name):
	song_timings = list()

	with open(MTURK_DIR + batch_name) as file:
		reader = csv.reader(file, delimiter=',')

		next(reader)

		for row in reader:
			song_name = row[0].replace('mp3', 'wav')

			song_location = MUSIC_DIR + song_name

			timings_str = row[1].split('\n')
			timings = list()

			for timing in timings_str:
				start = timing.split(',')[0]
				
				minute_seconds = start.split(':')

				minute = int(minute_seconds[0])
				seconds = int(minute_seconds[1])
				
				total_start = minute * 60 + seconds
				
				timings.append(total_start)

			song_timings.append({'name': song_name, 'timings': timings})

	return song_timings

def match_sizes(f_words, non_f_words):
	num = f_words.shape[0]

	mid = non_f_words.shape[0] // 2

	non_f_words_sample = non_f_words[mid:mid+num]

	return non_f_words_sample

def generate_foreground_audio():
	song_list = get_song_list('batch_main.csv')

	for song in song_list:
		clean_file_name = '{}clean/{}'.format(MUSIC_DIR, song['name'])

		if not os.path.isfile(clean_file_name):
			print('working on: {}'.format(song['name']))

			file_name = MUSIC_DIR + song['name']

			data, sr = librosa.load(file_name, sr=None, mono=True)

			foreground_data = vocal_separation.separate_vocals(data, sr)

			librosa.output.write_wav(clean_file_name, foreground_data, sr)

generate_foreground_audio()

# print(get_song_list('batch_8.csv', True))