import os
import json
import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Import the enhanced hybrid model
from enhanced_hybrid_model import HybridLyricsModel

# ==================== MODIFIED CATALOG CLASS ====================

class HybridCatalog:
    """
    Modified Catalog class compatible with Hybrid Model
    Works with your existing dataset format (\n--\n separator)
    """
    
    def __init__(self, language='hindi', song_separator='\n--\n', 
                 use_song_boundaries=True):
        self.catalog_items = []
        self.songs = []
        self.tokenizer = HybridTokenizer(language=language)
        self.max_sequence_length = 0
        self.total_words = 0
        self.language = language
        self.song_separator = song_separator
        self.use_song_boundaries = use_song_boundaries
        
        # For compatibility with hybrid model
        self.features = None
        self.labels = None
    
    def _preprocess_hindi_text(self, text):
        """Preprocess Hindi text"""
        import unicodedata
        import re
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    
    def add_file_to_catalog(self, file_name):
        """Load file with \n--\n separator"""
        print(f"\nLoading lyrics from: {file_name}")
        
        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if self.song_separator in content:
            songs = content.split(self.song_separator)
            print(f"Found {len(songs)} songs")
            
            for song_idx, song in enumerate(songs):
                song_lines = []
                for line in song.split('\n'):
                    if self.language == 'hindi':
                        # cleans data and only keep hindi text
                        processed_line = self._preprocess_hindi_text(line)
                    else:
                        processed_line = line.lower()
                    
                    if processed_line:
                        song_lines.append(processed_line)
                        self.catalog_items.append(processed_line)
                
                if song_lines:
                    self.songs.append(song_lines)
                    if (song_idx + 1) % 50 == 0:
                        print(f"  Processed {song_idx + 1} songs...")
    
    def tokenize_catalog(self):
        """Tokenize for hybrid model"""
        print(f"\n{'='*60}")
        print("TOKENIZING CATALOG")
        print(f"{'='*60}\n")
        
        self.catalog_items = [item for item in self.catalog_items if item.strip()]
        
        if not self.catalog_items:
            raise ValueError("No valid text in catalog")
        
        # Build vocabulary
        self.tokenizer.fit(self.catalog_items)
        self.total_words = self.tokenizer.vocab_size
        
        # Create sequences
        input_sequences = []
        
        if self.use_song_boundaries and self.songs:
            print(f"Tokenizing {len(self.songs)} songs with boundary respect...")
            
            for song_idx, song_lines in enumerate(self.songs):
                for line in song_lines:
                    token_list = self.tokenizer.encode(line, add_special_tokens=False)
                    
                    # Create n-grams
                    for i in range(1, len(token_list)):
                        n_gram_sequence = token_list[:i + 1]
                        input_sequences.append(n_gram_sequence)
                
                if (song_idx + 1) % 50 == 0:
                    print(f"  Processed {song_idx + 1}/{len(self.songs)} songs...")
        else:
            print("Tokenizing without song boundaries...")
            for line in self.catalog_items:
                token_list = self.tokenizer.encode(line, add_special_tokens=False)
                for i in range(1, len(token_list)):
                    n_gram_sequence = token_list[:i + 1]
                    input_sequences.append(n_gram_sequence)
        
        if not input_sequences:
            raise ValueError("No sequences generated")

        # Get the max sequence length from the data, but DON'T pad here.
        # The Dataset class will handle padding/truncating based on config.
        self.max_sequence_length = max([len(seq) for seq in input_sequences])
        
        # For hybrid model, we'll use sequences directly in dataset
        # Store the UNPADDED list of lists
        self.sequences = input_sequences
        
        # These are now incompatible, so just set to None
        self.features = None
        self.labels = None
        
        print(f"\n✓ Tokenization Complete!")
        print(f"  Vocabulary size: {self.total_words:,}")
        print(f"  Max sequence length (in data): {self.max_sequence_length}")
        print(f"  Training sequences: {len(input_sequences):,}")
        print(f"{'='*60}\n")


class HybridTokenizer:
    """Tokenizer compatible with hybrid model"""
    
    def __init__(self, language='hindi'):
        # Match hybrid model's special tokens
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.vocab_size = 4
        self.language = language
    
    def _preprocess_hindi(self, text):
        import unicodedata
        import re
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def fit(self, texts):
        """Build vocabulary"""
        print(f"Building vocabulary from {len(texts)} lines...")
        
        for idx, text in enumerate(texts):
            if self.language == 'hindi':
                text = self._preprocess_hindi(text)
            
            words = text.split()
            for word in words:
                if word and word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1} lines... Vocab: {self.vocab_size}")
        
        print(f"✓ Final vocabulary size: {self.vocab_size}")
    
    def encode(self, text, add_special_tokens=True):
        """Convert text to indices"""
        if self.language == 'hindi':
            text = self._preprocess_hindi(text)
        
        words = text.split()
        indices = []
        
        if add_special_tokens:
            indices.append(self.word2idx['<START>'])
        
        for word in words:
            if word:
                indices.append(self.word2idx.get(word, self.word2idx['<UNK>']))
        
        if add_special_tokens:
            indices.append(self.word2idx['<END>'])
        
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Convert indices to text"""
        words = []
        special_tokens = {'<PAD>', '<UNK>', '<START>', '<END>'}
        
        for idx in indices:
            if idx >= self.vocab_size:
                continue
            word = self.idx2word.get(idx, '<UNK>')
            if skip_special_tokens and word in special_tokens:
                continue
            words.append(word)
        
        return ' '.join(words)


class HybridLyricsDataset(Dataset):
    """Dataset for hybrid model"""
    
    def __init__(self, sequences, max_len=128):
        self.sequences = sequences
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Pad or truncate
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [0] * (self.max_len - len(seq))
        
        # Input and target
        input_seq = torch.tensor(seq[:-1], dtype=torch.long)
        target_seq = torch.tensor(seq[1:], dtype=torch.long)
        
        return input_seq, target_seq


# ==================== INTEGRATED TRAINING CLASS ====================

class HybridLyricsTrainer:
    """
    Drop-in replacement for your existing trainer
    Compatible with your config files
    """
    
    def __init__(self, config_file):
        with open(config_file, encoding='utf-8') as json_file:
            self.config = json.load(json_file)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize catalog
        language = self.config.get('language', 'hindi')
        use_song_boundaries = self.config.get('use_song_boundaries', True)
        song_separator = self.config.get('song_separator', '\n--\n')
        
        self.catalog = HybridCatalog(
            language=language,
            use_song_boundaries=use_song_boundaries,
            song_separator=song_separator
        )
        
        # Load data
        self.catalog.add_file_to_catalog(self.config['lyrics_file_path'])
        self.catalog.tokenize_catalog()
        
        self.is_interactive = self.config.get('is_interactive', True)
        self.model = None
    
    def _get_compiled_model(self):
        """Create and compile hybrid model"""
        
        vocab_size = self.catalog.total_words
        d_model = self.config.get('d_model', 256)
        nhead = self.config.get('nhead', 8)
        num_transformer_layers = self.config.get('num_transformer_layers', 6)
        num_gru_layers = self.config.get('num_gru_layers', 2)
        dim_feedforward = self.config.get('dim_feedforward', 1024)
        dropout = self.config.get('dropout', 0.1)
        #max_len = self.catalog.max_sequence_length
        max_len = self.config.get('max_seq_length', 256)
        use_gru = self.config.get('use_gru', True)
        
        print(f"\n{'='*60}")
        print("CREATING HYBRID MODEL")
        print(f"{'='*60}")
        print(f"  Vocabulary size: {vocab_size:,}")
        print(f"  Model dimension: {d_model}")
        print(f"  Attention heads: {nhead}")
        print(f"  Transformer layers: {num_transformer_layers}")
        print(f"  GRU layers: {num_gru_layers}")
        print(f"  Use GRU: {use_gru}")
        print(f"  Max sequence length: {max_len}")
        print(f"{'='*60}\n")
        
        model = HybridLyricsModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            num_gru_layers=num_gru_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            use_gru=use_gru
        ).to(self.device)
        
        if self.is_interactive:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model parameters:")
            print(f"  Total: {total_params:,}")
            print(f"  Trainable: {trainable_params:,}")
            print(f"  Size: ~{total_params * 4 / 1024 / 1024:.2f} MB\n")
        
        return model
    
    def _train_model(self, model):
        """Train the hybrid model"""
        
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}\n")
        
        # Split dataset
        train_sequences, test_sequences = train_test_split(
            self.catalog.sequences,
            test_size=self.config.get('test_size', 0.15),
            random_state=self.config.get('random_state', 42),
            shuffle=True
        )
        
        print(f"Dataset split:")
        print(f"  Training: {len(train_sequences):,} sequences")
        print(f"  Testing: {len(test_sequences):,} sequences\n")
        
        # Create datasets
        max_seq_len = self.config.get('max_seq_length', 128)
        train_dataset = HybridLyricsDataset(train_sequences, max_len=max_seq_len)
        test_dataset = HybridLyricsDataset(test_sequences, max_len=max_seq_len)
        
        # Create dataloaders
        batch_size = self.config.get('batch_size', 16)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer
        learning_rate = self.config.get('learning_rate', 0.0001)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98)
        )
        
        # Learning rate scheduler
        warmup_steps = self.config.get('warmup_steps', 500)
        total_steps = len(train_loader) * self.config['epochs']
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Loss function
        label_smoothing = self.config.get('label_smoothing', 0.1)
        criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=label_smoothing
        )
        
        # Training history
        history = {
            'train_loss': [],
            'test_loss': [],
            'train_perplexity': [],
            'test_perplexity': [],
            'learning_rate': []
        }
        
        best_test_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('early_stopping_patience', 5)
        grad_clip = self.config.get('grad_clip', 1.0)
        
        # Training loop
        for epoch in range(self.config['epochs']):
            # TRAIN
            model.train()
            train_loss = 0
            train_batches = 0
            
            print(f"\nEpoch [{epoch+1}/{self.config['epochs']}]")
            print("-" * 60)
            
            for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                
                # Generate mask
                seq_len = input_seq.size(1)
                mask = model.generate_square_subsequent_mask(seq_len).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output, _ = model(input_seq, src_mask=mask)
                
                # Loss
                loss = criterion(
                    output.reshape(-1, self.catalog.total_words),
                    target_seq.reshape(-1)
                )
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = train_loss / train_batches
                    perplexity = math.exp(min(avg_loss, 20))
                    print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f} | "
                          f"Perplexity: {perplexity:.2f}")
            
            avg_train_loss = train_loss / train_batches
            train_perplexity = math.exp(min(avg_train_loss, 20))
            
            # EVALUATE
            model.eval()
            test_loss = 0
            test_batches = 0
            
            with torch.no_grad():
                for input_seq, target_seq in test_loader:
                    input_seq = input_seq.to(self.device)
                    target_seq = target_seq.to(self.device)
                    
                    seq_len = input_seq.size(1)
                    mask = model.generate_square_subsequent_mask(seq_len).to(self.device)
                    
                    output, _ = model(input_seq, src_mask=mask)
                    loss = criterion(
                        output.reshape(-1, self.catalog.total_words),
                        target_seq.reshape(-1)
                    )
                    
                    test_loss += loss.item()
                    test_batches += 1
            
            avg_test_loss = test_loss / test_batches
            test_perplexity = math.exp(min(avg_test_loss, 20))
            
            # Save history
            history['train_loss'].append(avg_train_loss)
            history['test_loss'].append(avg_test_loss)
            history['train_perplexity'].append(train_perplexity)
            history['test_perplexity'].append(test_perplexity)
            history['learning_rate'].append(scheduler.get_last_lr()[0])
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1} SUMMARY")
            print(f"{'='*60}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
            print(f"  Test Loss:  {avg_test_loss:.4f} | Perplexity: {test_perplexity:.2f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Check improvement
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
                print(f"  ✓ New best! Saving checkpoint...")
                self.save_model(model, f"checkpoints/best_hybrid_epoch_{epoch+1}.pt")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")
            
            print(f"{'='*60}\n")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save history
        history_file = self.config.get('history_file', 'hybrid_training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✓ Training complete!")
        print(f"  Best test loss: {best_test_loss:.4f}")
        print(f"  History saved to: {history_file}")
        
        return history
    
    def generate_sample_lyrics(self, model):
        """Generate sample lyrics"""
        
        print(f"\n{'='*60}")
        print("GENERATING SAMPLE LYRICS")
        print(f"{'='*60}\n")
        
        seed_text = self.config.get('seed_text', 'मेरा')
        word_count = self.config.get('word_count', 100)
        temperature = self.config.get('temperature', 0.9)
        top_k = self.config.get('top_k', 50)
        top_p = self.config.get('top_p', 0.9)
        num_variations = self.config.get('num_variations', 3)
        
        model.eval()
        
        for i in range(num_variations):
            print(f"\n--- Variation {i+1} ---")
            
            # Encode seed
            tokens = self.catalog.tokenizer.encode(seed_text, add_special_tokens=True)
            
            # Generate
            generated_tokens = model.generate(
                start_tokens=tokens,
                max_length=word_count,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=self.device
            )
            
            # Decode
            generated_text = self.catalog.tokenizer.decode(
                generated_tokens.cpu().tolist(),
                skip_special_tokens=True
            )
            
            print(generated_text)
            print("-" * 60)
            
            # Save
            output_path = self.config.get('saved_lyrics_path', 'generated_lyrics.txt')
            if num_variations > 1:
                output_path = output_path.replace('.txt', f'_v{i+1}.txt')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            
            print(f"Saved to: {output_path}\n")
    
    def save_model(self, model, path):
        """Save model"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': self.catalog.tokenizer,
            'config': self.config,
            'vocab_size': self.catalog.total_words
        }, path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.catalog.tokenizer = checkpoint['tokenizer']
        self.config = checkpoint['config']
        
        self.model = self._get_compiled_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {path}")
        
        return self.model
    
    def generate_model(self):
        """Main training function - compatible with your existing code"""
        try:
            print(f"\n{'='*70}")
            print("HYBRID LYRICS MODEL TRAINING")
            print(f"{'='*70}\n")
            
            # Create model
            model = self._get_compiled_model()
            
            # Train
            history = self._train_model(model)
            
            # Save final model
            model_path = self.config.get('saved_model_path', 'hybrid_model_final.pt')
            self.save_model(model, model_path)
            
            # Generate samples
            self.generate_sample_lyrics(model)
            
            print(f"\n{'='*70}")
            print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n{'='*70}")
            print("ERROR OCCURRED")
            print(f"{'='*70}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


# ==================== USAGE ====================

if __name__ == "__main__":
    # Your existing config format works!
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    trainer = HybridLyricsTrainer('./config/config.json')
    trainer.generate_model()