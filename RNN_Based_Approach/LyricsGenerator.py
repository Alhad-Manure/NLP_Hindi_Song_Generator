import json
import os
import sys
import tensorflow as tf
from tensorflow import keras

# Import your custom modules
# Ensure these files are in the same directory or python path
from catalog import Catalog
from lyrics_formatter import LyricsFormatter

class LyricsGenerator:
    def __init__(self, config_file):
        """
        Initialize the generator:
        1. Load config
        2. Rebuild the Tokenizer (Catalog) exactly as it was during training
        3. Load the saved .h5 model
        """
        print(f"--- Loading configuration from {config_file} ---")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, encoding='utf-8') as json_file:
            self.config = json.load(json_file)

        # ---------------------------------------------------------
        # 1. REBUILD CATALOG (Crucial Step)
        # We must load the original text data to reconstruct the 
        # Word -> Integer mapping. Without this, the model output is useless.
        # ---------------------------------------------------------
        #print("Rebuilding Tokenizer/Catalog...")
        language = self.config.get('language', 'english')
        use_song_boundaries = self.config.get('use_song_boundaries', True)
        song_separator = self.config.get('song_separator', '\n--\n')

        self.catalog = Catalog(
            language=language,
            use_song_boundaries=use_song_boundaries,
            song_separator=song_separator
        )
        
        lyrics_path = self.config['lyrics_file_path']
        if not os.path.exists(lyrics_path):
            raise FileNotFoundError(f"Original lyrics file not found at: {lyrics_path}")

        self.catalog.add_file_to_catalog(lyrics_path)
        self.catalog.tokenize_catalog()
        print(f"Catalog loaded. Vocabulary size: {self.catalog.total_words}")

        # ---------------------------------------------------------
        # 2. LOAD MODEL
        # ---------------------------------------------------------
        model_path = self.config['saved_model_path']
        
        # Check if the path in config exists, otherwise check for _best version
        if not os.path.exists(model_path):
            best_path = model_path.replace('.h5', '_best.h5')
            if os.path.exists(best_path):
                model_path = best_path
        
        print(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR loading model: {e}")
            sys.exit(1)

    def generate(self, seed_text, word_count=None, temperature=1.0):
        """
        Generates lyrics based on the user provided seed_text
        """
        if word_count is None:
            word_count = self.config.get('word_count', 50)

        try:
            # Generate raw text using the catalog's method
            raw_lyrics = self.catalog.generate_lyrics_text(
                self.model,
                seed_text=seed_text,
                word_count=word_count,
                temperature=temperature,
                use_sampling=self.config.get('use_sampling', True)
            )

            # Cleanup
            cleaned_lyrics = self._remove_repetitions(raw_lyrics)

            # Format nicely
            formatted_lyrics = LyricsFormatter.format_lyrics(
                cleaned_lyrics, 
                self.config.get('word_group_count', 4)
            )
            
            return formatted_lyrics

        except Exception as e:
            print(f"Error during generation: {e}")
            return None

    def _remove_repetitions(self, text: str, max_repeat: int = 2) -> str:
        """Utility to prevent words looping like 'baby baby baby baby'"""
        words = text.split()
        result = []
        count = 1
        
        for i, word in enumerate(words):
            if i == 0 or word != words[i-1]:
                result.append(word)
                count = 1
            elif count < max_repeat:
                result.append(word)
                count += 1
        
        return ' '.join(result)

    def start_interactive_session(self):
        """
        Main loop that asks user for input
        """
        print(f"\n{'='*60}")
        print("LYRICS GENERATOR - INTERACTIVE MODE")
        print(f"{'='*60}")
        print("Instructions:")
        print("1. Type your seed text:")
        print("2. Type 'exit' to quit.")
        print("-" * 60)

        while True:
            # 1. Get Seed
            print("\nINPUT REQUIRED:")
            seed_text = input(">> Enter Seed Text: ").strip()

            if seed_text.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            
            if not seed_text:
                print("! Please enter at least one word.")
                continue

            # 2. Get Length (Optional)
            word_count = 50

            try:
                temperature = 1.0
            except ValueError:
                temperature = 1.0

            # 4. Generate
            lyrics = self.generate(seed_text, word_count, temperature)
            
            if lyrics:
                print(f"\n{'-'*20} GENERATED SONG {'-'*20}")
                print(lyrics)
                print(f"{'-'*56}")

if __name__ == "__main__":
    # You can pass the config path as an argument, or default to 'config.json'
    config_path = './config_files/hindi_song_config.json'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    try:
        generator = LyricsGenerator(config_path)
        generator.start_interactive_session()
    except Exception as e:
        print(f"Failed to start generator: {e}")