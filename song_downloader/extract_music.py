import re

file = open('html_songs_list.txt')
content = file.read()

file.close()

songs_matches = re.findall('\/tracks">(?:.*)(?=<\/a>)', content)

songs = list()

for song_match in songs_matches:		
	song_name = song_match.split('>')[1] + ' - Lil Wayne'

	if '\'' in song_name:
		song_name = song_name.replace('\'', '')

	songs.append(song_name)
	# songs.append(song_name + ' Clean')

with open('songs.txt', 'w') as f:
	for item in songs:
		f.write("%s\n" % item)