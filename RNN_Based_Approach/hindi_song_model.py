import lyrics_model as lm

def main():
    lm.tensorflow_diagnostics()
    lyrics_model = lm.LyricsModel('config_files/hindi_song_config.json')
    lyrics_model.generate_model()


if __name__ == '__main__':
    main()
