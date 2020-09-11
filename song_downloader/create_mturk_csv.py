from os import listdir
from os.path import isfile, join
import csv

PATH = 'songs/mp3/'

songs = [f for f in listdir(PATH) if isfile(join(PATH, f))]

url = 'https://nocvi.com/songs/mp3/'

links = list()

for song in songs:
	dl_link = url + song
	links.append(dl_link)

with open('audio_links.csv', 'w') as file:
	file.write('name,url\n')

	for i in range(len(links)):
		file.write(songs[i] + ',' + links[i] + '\n')