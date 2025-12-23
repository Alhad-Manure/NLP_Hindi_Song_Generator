import numpy as np
import csv
import re
from typing import List, Optional

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Catalog:
    """
    represents a catalog (aka corpus) of works, used in both model training and prediction
    Modified for Hindi/Devanagari text support with song separator handling
    """

    def __init__(self, padding: str = 'pre', oov_token='<OOV>', language='hindi', 
                 song_separator='\n--\n', use_song_boundaries=True):
        self.catalog_items: List[str] = []
        self.songs: List[List[str]] = []  # Store songs as separate entities
        self.tokenizer = Tokenizer(oov_token=oov_token, char_level=False)
        self.max_sequence_length = 0
        self.total_words = 0
        self.features: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self._padding = padding
        self.language = language
        self.song_separator = song_separator
        self.use_song_boundaries = use_song_boundaries

    def _preprocess_hindi_text(self, text: str) -> str:
        """
        Preprocess Hindi text for better tokenization
        
        :param text: raw Hindi text
        :return: preprocessed text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Unicode (important for Devanagari)
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # Remove English characters if desired (optional)
        # text = re.sub(r'[a-zA-Z]', '', text)
        
        # Remove special characters but keep Devanagari range
        # Devanagari Unicode range: \u0900-\u097F
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        
        return text.strip()

    def add_file_to_catalog(self, file_name: str, respect_song_separator: bool = True) -> None:
        """
        add a text file to the catalog
        Now supports Hindi text with UTF-8 encoding and song separator handling

        :param file_name: file name with lyrics/text
        :param respect_song_separator: if True, splits content by song_separator
        :return: None
        """
        with open(file_name, 'r', encoding='utf-8') as text_file:
            content = text_file.read()
            
            if respect_song_separator and self.song_separator in content:
                # Split by song separator
                songs = content.split(self.song_separator)
                #print(f"Found {len(songs)} songs in file '{file_name}'")
                
                for song_idx, song in enumerate(songs):
                    song_lines = []
                    for line in song.split('\n'):
                        if self.language == 'hindi':
                            processed_line = self._preprocess_hindi_text(line)
                        else:
                            processed_line = line.lower()
                        
                        if processed_line:  # Only add non-empty lines
                            song_lines.append(processed_line)
                            self.catalog_items.append(processed_line)
                    
                    if song_lines:
                        self.songs.append(song_lines)
                        #print(f"  Song {song_idx + 1}: {len(song_lines)} lines")
            else:
                # Process line by line (original behavior)
                song_lines = []
                for line in content.split('\n'):
                    if self.language == 'hindi':
                        processed_line = self._preprocess_hindi_text(line)
                    else:
                        processed_line = line.lower()
                    
                    if processed_line:
                        song_lines.append(processed_line)
                        self.catalog_items.append(processed_line)
                
                if song_lines:
                    self.songs.append(song_lines)

    def add_csv_file_to_catalog(self, file_name: str, text_column: int, skip_first_line: bool = True,
                                delimiter: str = ',', respect_song_separator: bool = True) -> None:
        """
        add a csv, tsv or other delimited file to the catalog
        Now supports Hindi text with UTF-8 encoding and song separator handling

        :param file_name: file name with lyrics/text
        :param text_column: column number to select, 0 based
        :param skip_first_line: skip first line of text
        :param delimiter: delimiter to use as separator
        :param respect_song_separator: if True, splits content by song_separator
        :return: None
        """
        with open(file_name, 'r', encoding='utf-8') as text_file:
            csv_reader = csv.reader(text_file, delimiter=delimiter)
            if skip_first_line:
                next(csv_reader)
            
            for row in csv_reader:
                if len(row) > text_column:
                    text_content = row[text_column]
                    
                    if respect_song_separator and self.song_separator in text_content:
                        # Split by song separator
                        songs = text_content.split(self.song_separator)
                        
                        for song in songs:
                            song_lines = []
                            for line in song.split('\n'):
                                if self.language == 'hindi':
                                    processed_line = self._preprocess_hindi_text(line)
                                else:
                                    processed_line = line.lower()
                                
                                if processed_line:
                                    song_lines.append(processed_line)
                                    self.catalog_items.append(processed_line)
                            
                            if song_lines:
                                self.songs.append(song_lines)
                    else:
                        if self.language == 'hindi':
                            processed_line = self._preprocess_hindi_text(text_content)
                        else:
                            processed_line = text_content.lower()
                        
                        if processed_line:
                            self.catalog_items.append(processed_line)

    def tokenize_catalog(self, respect_song_boundaries: bool = True) -> None:
        """
        tokenize the contents of the catalog, and set properties accordingly (ex: total_words, labels)
        Can optionally respect song boundaries to avoid creating sequences across different songs

        :param respect_song_boundaries: if True, doesn't create n-grams across song boundaries
        :return: None
        """
        # Filter out empty items
        self.catalog_items = [item for item in self.catalog_items if item.strip()]
        
        if not self.catalog_items:
            raise ValueError("No valid text in catalog after preprocessing")

        # tokenizer: fit, sequence, pad
        self.tokenizer.fit_on_texts(self.catalog_items)

        # create a list of n-gram sequences
        input_sequences = []

        if respect_song_boundaries and self.songs:
            #print(f"Tokenizing {len(self.songs)} songs with boundary respect...")
            # Process each song separately
            for song_idx, song_lines in enumerate(self.songs):
                song_sequences = 0
                for line in song_lines:
                    token_list = self.tokenizer.texts_to_sequences([line])[0]
                    for i in range(1, len(token_list)):
                        n_gram_sequence = token_list[:i + 1]
                        input_sequences.append(n_gram_sequence)
                        song_sequences += 1
                
                if (song_idx + 1) % 10 == 0:
                    #print(f"  Processed {song_idx + 1}/{len(self.songs)} songs...")
                    pass
        else:
            print("Tokenizing without song boundary respect...")
            # Original behavior - process all lines together
            for line in self.catalog_items:
                token_list = self.tokenizer.texts_to_sequences([line])[0]
                for i in range(1, len(token_list)):
                    n_gram_sequence = token_list[:i + 1]
                    input_sequences.append(n_gram_sequence)

        if not input_sequences:
            raise ValueError("No sequences generated. Check your input data.")

        # pad sequences
        self.max_sequence_length = max([len(item) for item in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_sequence_length,
                                                 padding=self._padding))

        self.features = input_sequences[:, :-1]
        labels_temp = input_sequences[:, -1]

        self.total_words = len(self.tokenizer.word_index) + 1
        self.labels = keras.utils.to_categorical(labels_temp, num_classes=self.total_words)


        print(f"\n{'='*50}")
        print("\n")
        '''
        print(f"Tokenization Complete:")
        print(f"  Total songs: {len(self.songs)}")
        print(f"  Total vocabulary size: {self.total_words}")
        print(f"  Maximum sequence length: {self.max_sequence_length}")
        print(f"  Total training sequences: {len(input_sequences)}")
        '''
        print(f"{'='*50}\n")

    def get_song_statistics(self) -> dict:
        """
        Get statistics about the songs in the catalog
        
        :return: dictionary with song statistics
        """
        if not self.songs:
            return {"total_songs": 0}
        
        song_lengths = [len(song) for song in self.songs]
        
        return {
            "total_songs": len(self.songs),
            "total_lines": sum(song_lengths),
            "avg_lines_per_song": np.mean(song_lengths),
            "min_lines": min(song_lengths),
            "max_lines": max(song_lengths),
            "median_lines": np.median(song_lengths)
        }

    def generate_lyrics_text(self, model: keras.Sequential, seed_text: str, word_count: int, 
                            temperature: float = 1.0, use_sampling: bool = True) -> str:
        """
        generate lyrics using the provided model and properties
        Now handles Hindi text properly with improved generation

        :param model: model used to generate text
        :param seed_text: starter text (in Hindi if language='hindi')
        :param word_count: total number of words to return
        :param temperature: sampling temperature (higher = more random)
        :param use_sampling: use probabilistic sampling instead of argmax
        :return: starter text + generated text
        """
        if self.language == 'hindi':
            seed_text = self._preprocess_hindi_text(seed_text)
        
        # Ensure seed text is in vocabulary
        seed_tokens = self.tokenizer.texts_to_sequences([seed_text])
        if not seed_tokens or not seed_tokens[0]:
            print(f"Warning: Seed text '{seed_text}' not in vocabulary. Using first word from vocabulary.")
            seed_text = list(self.tokenizer.word_index.keys())[0]
        
        print(f"Starting generation with seed: '{seed_text}'")
        seed_text_word_count = len(seed_text.split())
        words_to_generate = max(1, word_count - seed_text_word_count)

        for i in range(words_to_generate):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            
            if not token_list:
                print(f"Warning: Empty token list at iteration {i}")
                break
            
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_length - 1, padding=self._padding)
            
            # Get prediction probabilities
            predicted_probs = model.predict(token_list, verbose=0)[0]
            
            if use_sampling:
                # Apply temperature and sample
                predicted_probs = np.log(predicted_probs + 1e-10) / temperature
                predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))
                predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
            else:
                # Use argmax (deterministic)
                predicted_index = np.argmax(predicted_probs)
            
            output_word = self.tokenizer.index_word.get(predicted_index)
            
            if output_word is None or output_word == '<OOV>':
                print(f"Warning: Got invalid word at iteration {i} (index: {predicted_index})")
                # Try getting second best prediction
                predicted_probs[predicted_index] = 0
                predicted_index = np.argmax(predicted_probs)
                output_word = self.tokenizer.index_word.get(predicted_index)
            
            if output_word and output_word != '<OOV>':
                seed_text += ' ' + output_word
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{words_to_generate} words...")
            else:
                print(f"Stopping generation at iteration {i} - no valid word found")
                break

        print(f"Final generated text length: {len(seed_text.split())} words")
        return seed_text