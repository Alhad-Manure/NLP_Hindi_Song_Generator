import json
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split

from ptmlib.time import Stopwatch
from ptmlib import charts as pch
from catalog import Catalog
from lyrics_formatter import LyricsFormatter


def tensorflow_diagnostics():
    print('tf version:', tf.__version__)
    print('keras version:', keras.__version__)
    print('GPU available:', tf.config.list_physical_devices('GPU'))


class LyricsModel:

    def __init__(self, config_file):
        with open(config_file, encoding='utf-8') as json_file:
            self.config = json.load(json_file)
        
        language = self.config.get('language', 'english')
        use_song_boundaries = self.config.get('use_song_boundaries', True)
        song_separator = self.config.get('song_separator', '\n--\n')

        self.catalog = Catalog(language=language)
        self.catalog = Catalog(
            language=language,
            use_song_boundaries=use_song_boundaries,
            song_separator=song_separator
        )

        self.catalog.add_file_to_catalog(self.config['lyrics_file_path'])
        self.catalog.tokenize_catalog()
        self.is_interactive = self.config['is_interactive']

    def _get_compiled_model(self) -> keras.Sequential:

        total_words = self.catalog.total_words
        dimensions = self.config['hp_output_dimensions']
        input_length = self.catalog.max_sequence_length - 1
        units = self.config['hp_lstm_units']
        dropout_rate = self.config.get('hp_dropout_rate', 0.2)
        architecture = self.config.get('architecture', 'deep')  # 'simple', 'deep', 'very_deep'

        if architecture == 'simple':
            # Original architecture
            model = keras.Sequential([
                keras.layers.Embedding(total_words, dimensions, input_length=input_length),
                keras.layers.Bidirectional(keras.layers.LSTM(units)),
                keras.layers.Dense(total_words, activation='softmax')
            ])

        elif architecture == 'deep':
            # Improved architecture with 2 LSTM layers

            model = keras.Sequential([
                keras.layers.Embedding(total_words, dimensions, input_length=input_length),
                keras.layers.Bidirectional(keras.layers.LSTM(units, return_sequences=True)),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Bidirectional(keras.layers.LSTM(units)),
                keras.layers.Dense(units // 2, activation='relu'),
                keras.layers.Dense(total_words, activation='softmax')
            ])


        else:  # 'very_deep'
            # Even deeper architecture with 3 LSTM layers
            model = keras.Sequential([
                keras.layers.Embedding(total_words, dimensions, input_length=input_length),
                keras.layers.Bidirectional(keras.layers.LSTM(units, return_sequences=True)),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Bidirectional(keras.layers.LSTM(units, return_sequences=True)),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Bidirectional(keras.layers.LSTM(units // 2)),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Dense(units // 2, activation='relu'),
                keras.layers.Dropout(dropout_rate / 2),
                keras.layers.Dense(total_words, activation='softmax')
            ])

        # Configurable optimizer
        optimizer_name = self.config.get('optimizer', 'adam')
        learning_rate = self.config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        elif optimizer_name == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

        else:
            optimizer = 'adam'


        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', 'top_k_categorical_accuracy'])

        if self.is_interactive:
            model.summary()
            #print(f"\nTotal parameters: {model.count_params():,}")

        return model

    def _train_model(self, model: keras.Sequential):

        callbacks_list = []

        # Model Checkpoint - saves best model during training
        if self.config.get('save_best_model', True):
            checkpoint_path = self.config['saved_model_path'].replace('.h5', '_best.h5')
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            callbacks_list.append(model_checkpoint)

        # Reduce Learning Rate on Plateau - reduces LR when learning stagnates
        if self.config.get('use_reduce_lr', True):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.get('lr_patience', 3),
                min_lr=1e-7,
                verbose=1
            )
            callbacks_list.append(reduce_lr)

        # TensorBoard logging (optional)
        if self.config.get('use_tensorboard', False):
            tensorboard = keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                write_graph=True
            )
            callbacks_list.append(tensorboard)

        stopwatch = Stopwatch()
        stopwatch.start()

        x_train, x_valid, y_train, y_valid = train_test_split(
            self.catalog.features, 
            self.catalog.labels,
            test_size=self.config['hp_test_size'],
            random_state=self.config['random_state']
        )

        print(f"\nTraining samples: {len(x_train):,}")
        print(f"Validation samples: {len(x_valid):,}")
        print(f"Batch size: {self.config.get('batch_size', 32)}")

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=self.config['hp_epochs'],
            batch_size=self.config.get('batch_size', 32),
            verbose=1,
            callbacks=callbacks_list
        )

        stopwatch.stop(silent=not self.is_interactive)

        if self.config.get('save_history', True):
            history_path = self.config.get('history_path', 'training_history.json')
            history_dict = {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            }
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_dict, f, indent=2)

        if self.is_interactive:
            pch.show_history_chart(history, 'accuracy', save_fig_enabled=self.config['save_chart'])
            pch.show_history_chart(history, 'loss', save_fig_enabled=self.config['save_chart'])

            # Print final metrics
            print(f"\n{'='*50}")
            print("TRAINING COMPLETE")
            print(f"{'='*50}")
            print(f"Final training loss: {history.history['loss'][-1]:.4f}")
            print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
            print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
            print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
            print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
            print(f"{'='*50}\n")

    def _generate_sample_lyrics(self, model: keras.Sequential):

        print(f"\n{'='*50}")
        print("GENERATING LYRICS")
        print(f"{'='*50}")

        num_variations = self.config.get('num_variations', 1)

        for i in range(num_variations):
            print(f"\nGenerating variation {i+1}/{num_variations}...")
            
            # Use different parameters for each variation
            temperature = self.config.get('temperature', 1.0)
            if num_variations > 1:
                # Vary temperature for diversity
                temperature = temperature * (0.7 + i * 0.3 / num_variations)
            
            lyrics_text = self.catalog.generate_lyrics_text(
                model,
                seed_text=self.config['seed_text'],
                word_count=self.config['word_count'],
                temperature=temperature,
                use_sampling=self.config.get('use_sampling', False)
            )
            
            # IMPROVEMENT: Remove excessive repetitions
            if self.config.get('remove_repetitions', True):
                lyrics_text = self._remove_repetitions(lyrics_text, max_repeat=2)
            
            lyrics = LyricsFormatter.format_lyrics(lyrics_text, self.config['word_group_count'])
            
            print(f"\n--- Variation {i+1} (temp={temperature:.2f}) ---")
            print(lyrics)
            print("-" * 50)
            
            # Save each variation
            if num_variations > 1:
                output_path = self.config['saved_lyrics_path'].replace('.txt', f'_v{i+1}.txt')
            else:
                output_path = self.config['saved_lyrics_path']
            
            with open(output_path, 'w', encoding='utf-8') as lyrics_file:
                lyrics_file.write(f"# Generated Hindi Song (Variation {i+1})\n")
                lyrics_file.write(f"# Temperature: {temperature:.2f}\n")
                lyrics_file.write(f"# Seed: {self.config['seed_text']}\n\n")
                lyrics_file.write(lyrics)
            
            print(f"Saved to: {output_path}")

    def _remove_repetitions(self, text: str, max_repeat: int = 2) -> str:
        """Remove consecutive repeated words"""
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

    def generate_model(self):

        """
        train and save a model
        """
        try:
            print(f"\n{'='*60}")
            print("STARTING MODEL TRAINING")
            print(f"{'='*60}\n")

            model = self._get_compiled_model()
            self._train_model(model)

            model.save(self.config['saved_model_path'])
            print(f"\nModel saved to: {self.config['saved_model_path']}")

            # Generate sample lyrics
            self._generate_sample_lyrics(model)

            print(f"\n{'='*60}")
            print("ALL TASKS COMPLETED SUCCESSFULLY")
            print(f"{'='*60}\n")
        
        except Exception as e:
            print(f"\n{'='*60}")
            print("ERROR OCCURRED")
            print(f"{'='*60}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def evaluate_model(self, model: keras.Sequential):
        """
        IMPROVEMENT: New method to evaluate model performance
        """
        x_train, x_valid, y_train, y_valid = train_test_split(
            self.catalog.features,
            self.catalog.labels,
            test_size=self.config['hp_test_size'],
            random_state=self.config['random_state']
        )
        
        print("\nEvaluating model on validation set...")
        results = model.evaluate(x_valid, y_valid, verbose=0)
        
        print(f"Validation Loss: {results[0]:.4f}")
        print(f"Validation Accuracy: {results[1]:.4f}")
        
        return results