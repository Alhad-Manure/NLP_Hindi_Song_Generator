import torch
import json
import sys
import traceback
import argparse  # Added for command line arguments
from enhanced_hybrid_model import HybridLyricsModel
from hybrid_model_train import HybridTokenizer


def load_model_and_generate(
    model_path='fixed_hybrid_final.pt',
    seed_text='तेरी याद में',
    max_length=100,
    temperature=0.9,
    top_k=50,
    top_p=0.9,
    num_variations=3
):
    """
    Load model and generate lyrics
    """
    
    print("=" * 70)
    print("HINDI LYRICS GENERATOR")
    print("=" * 70)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load checkpoint
    print(f"Loading model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        print("Make sure you have the correct path to your .pt file")
        return
    
    # Get tokenizer and config
    tokenizer = checkpoint['tokenizer']
    config = checkpoint.get('config', {})
    vocab_size = checkpoint.get('vocab_size', tokenizer.vocab_size)
    
    print(f"✓ Model loaded successfully!")
    print(f"  Vocabulary size: {vocab_size:,}")
    
    # Create model architecture
    print("\nRecreating model architecture...")
    
    # Import the model class
    try:
        
        model = HybridLyricsModel(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 4),
            num_transformer_layers=config.get('num_transformer_layers', 3),
            num_gru_layers=config.get('num_gru_layers', 1),
            dim_feedforward=config.get('dim_feedforward', 512),
            dropout=config.get('dropout', 0.15),
            max_len=config.get('max_seq_length', 512),
            use_gru=config.get('use_gru', True)
        )
    except ImportError:
        print("Error: Cannot import HybridLyricsModel")
        print("Make sure fixed_hybrid_model_final.py is in the same directory")
        return
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✓ Model weights loaded!")
    
    # Generate lyrics
    print("\n" + "=" * 70)
    print("GENERATING LYRICS")
    print("=" * 70)
    print(f"\nSeed text: '{seed_text}'")
    print(f"Generating {num_variations} variation(s)...")
    #print(f"Settings: temperature={temperature}, top_k={top_k}, top_p={top_p}")
    print()
    
    generated_songs = []
    
    for i in range(num_variations):
        print(f"\n{'='*70}")
        print(f"VARIATION {i+1}/{num_variations}")
        print(f"{'='*70}\n")
        
        try:
            # Encode seed text
            seed_tokens = tokenizer.encode(seed_text, add_special_tokens=True)
            
            if not seed_tokens:
                print(f"Warning: Seed text '{seed_text}' produced no tokens")
                print("Using default seed")
                seed_tokens = [tokenizer.word2idx.get('मेरा', 2)]
            
            # Generate
            generated_tokens = model.generate(
                start_tokens=seed_tokens,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device
            )
            
            # Decode
            generated_text = tokenizer.decode(
                generated_tokens.cpu().tolist(),
                skip_special_tokens=True
            )
            
            # Format output
            print(generated_text)
            print()
            
            # Save
            generated_songs.append({
                'variation': i + 1,
                'seed': seed_text,
                'text': generated_text,
                'length': len(generated_text.split())
            })
            
        except Exception as e:
            print(f"Error generating variation {i+1}: {e}")
            traceback.print_exc()
    
    # Save to file
    # Clean filename to avoid OS errors with special characters
    safe_seed = "".join([c for c in seed_text[:10] if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
    output_file = f"generated_lyrics_{safe_seed}.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("GENERATED HINDI LYRICS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Seed: {seed_text}\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"Generated: {len(generated_songs)} variations\n\n")
            
            for song in generated_songs:
                f.write("=" * 70 + "\n")
                f.write(f"VARIATION {song['variation']}\n")
                f.write("=" * 70 + "\n\n")
                f.write(song['text'] + "\n\n")
        
        print("=" * 70)
        print(f"✓ Saved to: {output_file}")
        print("=" * 70)
    except Exception as e:
        print(f"Warning: Could not save to file: {e}")
    
    return generated_songs


if __name__ == "__main__":
    # Create the Argument Parser
    parser = argparse.ArgumentParser(description="Generate Hindi Lyrics using Hybrid Model")

    # Add arguments
    parser.add_argument('--model', type=str, default='./checkpoints/best_hybrid_epoch_120.pt', 
                        help='Path to the model .pt file')
    
    parser.add_argument('--seed', type=str, default='तेरी याद में', 
                        help='Starting text (seed phrase) in Hindi')
    
    parser.add_argument('--length', type=int, default=100, 
                        help='Maximum length of generated lyrics')
    
    parser.add_argument('--temp', type=float, default=0.9, 
                        help='Temperature (creativity): 0.7=safe, 1.2=creative')
    
    parser.add_argument('--variations', type=int, default=3, 
                        help='Number of different versions to generate')

    # Parse the arguments
    args = parser.parse_args()

    print("-" * 70)
    
    # Call the function using arguments from command line
    load_model_and_generate(
        model_path=args.model,
        seed_text=args.seed,
        max_length=args.length,
        temperature=args.temp,
        num_variations=args.variations
    )
    
    print("-" * 70)