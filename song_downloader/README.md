#How to get started

*The downloader.py file belong to rwithik, see the readme here: https://github.com/rwithik/song-downloader - i modified it ever so slightly so that it downloads songs as mp3 files.*

1. This folder has its own virtual env, with its own requirements
2. html_songs_list.txt is just some html that I copied off a website (sorry forgot the url), it just has a list of Lil Wayne songs
3. extract_music.py will parse html_songs_list.txt and output a file called songs.txt with each song on a new line
4. format_music.py will go through all the songs in songs/ directory, convert them from mp3 to wav, and give them a standard sample rate of 16000
5. you can use the downloader to donwload the songs in songs.txt
