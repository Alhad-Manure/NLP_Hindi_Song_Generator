import torch
import torch.nn.functional as F
import argparse
import os
import sys

# ============================================================================
#  IMPORT FROM MODEL.PY
# ============================================================================
try:
    import model
    from model import TransformerLyricsModel, HindiTokenizer
except ImportError:
    print("Error: 'model.py' not found.")
    print("Please ensure your training script is renamed to 'model.py' and is in this folder.")
    sys.exit(1)

# ----------------------------------------------------------------------------
# CRITICAL FIX FOR PICKLE LOADING
# If you trained your model using a script run as __main__, PyTorch saved the 
# classes as '__main__.TransformerLyricsModel'. When loading here, it will fail
# unless we map '__main__' to the 'model' module.
sys.modules['__main__'] = model
# ----------------------------------------------------------------------------

class LyricsGenerator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the checkpoint
        # map_location ensures it loads on CPU if CUDA is not available
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 1. Load Configuration
        self.config = checkpoint['config']
        
        # 2. Load Tokenizer 
        # (This is a pickled object, relies on HindiTokenizer class being available)
        self.tokenizer = checkpoint['tokenizer']
        
        # 3. Initialize Model Architecture (using imported class)
        self.model = TransformerLyricsModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout'],
            max_len=self.config['max_seq_length']
        )
        
        # 4. Load Weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model and Tokenizer loaded successfully!")

    def generate(self, seed_text, max_length=100, temperature=0.9, top_k=50):
        """
        Generate lyrics using the loaded model.
        """
        # Preprocess and encode the seed text
        tokens = self.tokenizer.encode(seed_text, add_special_tokens=True)
        
        # If the tokenizer adds an <END> token at the end, remove it so we can continue generating
        end_token_idx = self.tokenizer.word2idx.get('<END>', 3)
        if tokens[-1] == end_token_idx:
            tokens = tokens[:-1]

        input_ids = tokens
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare batch: (1, seq_len)
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                
                # Generate mask using the model's internal method
                seq_len = input_tensor.size(1)
                mask = self.model.generate_square_subsequent_mask(seq_len).to(self.device)

                # Forward pass
                output = self.model(input_tensor, mask)
                
                # Get logits for the last token only
                next_token_logits = output[0, -1, :] / temperature
                
                # Top-K Sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample from the distribution
                next_token_index = torch.multinomial(probs, 1).item()
                real_token = top_k_indices[next_token_index].item()
                
                # Stop if <END> token is generated
                if real_token == end_token_idx:
                    break
                    
                input_ids.append(real_token)

        # Decode the generated indices back to text
        generated_text = self.tokenizer.decode(input_ids)
        return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hindi Lyrics using trained Transformer")
    
    parser.add_argument('--model', type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument('--seed', type=str, default="मेरा दिल", help="Start of the song (Hindi text)")
    parser.add_argument('--length', type=int, default=100, help="Max words to generate")
    parser.add_argument('--temp', type=float, default=0.9, help="Creativity (0.7-1.2)")
    parser.add_argument('--top_k', type=int, default=40, help="Top-K sampling")

    args = parser.parse_args()

    try:
        gen = LyricsGenerator(args.model)
        
        print(f"\n{'='*60}")
        print(f"SEED: {args.seed}")
        print(f"{'='*60}\n")
        
        lyrics = gen.generate(
            seed_text=args.seed,
            max_length=args.length,
            temperature=args.temp,
            top_k=args.top_k
        )
        
        print(lyrics)
        print(f"\n{'='*60}")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()