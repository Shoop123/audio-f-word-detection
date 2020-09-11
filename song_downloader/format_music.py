from librosa.core import load, resample
from librosa.output import write_wav 
from os import listdir
from os.path import isfile, join

PATH = 'songs/'
NEW_PATH = 'songs/wav/'

SAMPLE_RATE = 16000

songs = [f for f in listdir(PATH) if isfile(join(PATH, f))]

for song in songs:
	song_location = PATH + song

	print(song_location)

	data, sample_rate = load(song_location, sr=None)
	resampled_data = resample(data, sample_rate, SAMPLE_RATE)

	new_song_location = NEW_PATH + song.replace('mp3', 'wav')
	
	write_wav(new_song_location, resampled_data, sr=SAMPLE_RATE)