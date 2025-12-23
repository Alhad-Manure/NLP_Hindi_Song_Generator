"""
Enhanced Hybrid Model for Hindi Lyrics Generation - FIXED
Fixes: Positional Encoding buffer mismatch error during generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class EnhancedAttention(nn.Module):
    """Multi-head attention with additional context-aware mechanisms"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Linear layers
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k) * 2.0
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize with small weights"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k)

        Q = Q.permute(0, 2, 1, 3).contiguous()
        K = K.permute(0, 2, 1, 3).contiguous()
        V = V.permute(0, 2, 1, 3).contiguous()
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        scores = torch.clamp(scores, min=-30, max=30)
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            
            # Ensure dimensions match before applying
            if scores.shape[-2:] == mask.shape[-2:]:
                scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        if torch.isnan(attn_weights).any():
            attn_weights = torch.ones_like(attn_weights) / attn_weights.size(-1)
        
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights


# ==================== FIXED CLASS ====================
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with Dynamic Resizing.
    Fixes the 'tensor a (21) must match tensor b (20)' error.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model  # Save d_model for dynamic resizing
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe * 0.1
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # DYNAMIC RESIZING FIX
        # If the input sequence (x) is longer than the cached encoding (self.pe),
        # we generate a new encoding matrix on the fly.
        if x.size(1) > self.pe.size(1):
            device = x.device
            new_max_len = x.size(1) + 50 # Add a buffer
            
            pe = torch.zeros(new_max_len, self.d_model, device=device)
            position = torch.arange(0, new_max_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float().to(device) * (-math.log(10000.0) / self.d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe * 0.1
            pe = pe.unsqueeze(0)
            
            # Update the buffer so we don't recalculate every step
            self.pe = pe.to(device)
            
        # Standard addition
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
# =====================================================


class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.self_attn = EnhancedAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.5)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x, mask=None):
        normed = self.norm1(x)
        attn_output, attn_weights = self.self_attn(normed, normed, normed, mask)
        x = x + self.dropout1(attn_output)
        
        normed = self.norm2(x)
        ffn_output = self.linear2(F.gelu(self.linear1(normed)))
        x = x + self.dropout2(ffn_output)
        
        return x, attn_weights


class HybridLyricsModel(nn.Module):
    """Hybrid model combining Transformer + GRU"""
    def __init__(self, vocab_size, d_model=256, nhead=8, num_transformer_layers=4,
                 num_gru_layers=1, dim_feedforward=1024, dropout=0.1, 
                 max_len=512, use_gru=True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_gru = use_gru
        
        # Special tokens
        self.pad_id = 0
        self.unk_id = 1
        self.start_id = 2
        self.end_id = 3
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_transformer_layers)
        ])
        
        self.transformer_norm = nn.LayerNorm(d_model, eps=1e-5)
        
        if use_gru:
            self.gru = nn.GRU(
                d_model, d_model, num_gru_layers,
                batch_first=True,
                dropout=dropout if num_gru_layers > 1 else 0
            )
            self.gru_norm = nn.LayerNorm(d_model, eps=1e-5)
            
            for name, param in self.gru.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param, gain=0.5)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param, gain=0.5)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        self.output_projection = nn.Linear(d_model, vocab_size, bias=True)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.5)
        nn.init.zeros_(self.output_projection.bias)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return (mask == 0)
    
    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = src * math.sqrt(self.d_model) * 0.1
        src = self.pos_encoder(src)
        
        attention_weights = []
        for layer in self.transformer_layers:
            src, attn_weights = layer(src, src_mask)
            attention_weights.append(attn_weights)
            
        src = self.transformer_norm(src)
        
        if self.use_gru:
            gru_out, _ = self.gru(src)
            
            # Shape protection for GRU
            if gru_out.shape != src.shape:
                if gru_out.size(1) > src.size(1):
                    gru_out = gru_out[:, :src.size(1), :]
                elif gru_out.size(1) < src.size(1):
                    padding = torch.zeros(
                        src.size(0), 
                        src.size(1) - gru_out.size(1), 
                        gru_out.size(2),
                        device=src.device,
                        dtype=src.dtype
                    )
                    gru_out = torch.cat([gru_out, padding], dim=1)
            
            src = self.gru_norm(src + gru_out * 0.5)
        
        output = self.output_projection(src)
        output = torch.clamp(output, min=-30, max=30)
        
        return output, attention_weights
    
    def generate(self, start_tokens, max_length=100, temperature=1.0,
                 top_k=50, top_p=0.9, device='cpu'):
        self.eval()
        
        if isinstance(start_tokens, list):
            current_tokens = torch.LongTensor([start_tokens]).to(device)
        else:
            if len(start_tokens.shape) == 1:
                current_tokens = start_tokens.unsqueeze(0).to(device)
            else:
                current_tokens = start_tokens.to(device)
        
        with torch.no_grad():
            for _ in range(max_length):
                seq_len = current_tokens.size(1)
                
                if seq_len >= 512:
                    break
                
                mask = self.generate_square_subsequent_mask(seq_len).to(device)
                
                output, _ = self.forward(current_tokens, src_mask=mask)
                
                logits = output[0, -1, :] / max(temperature, 0.1)
                logits = torch.clamp(logits, min=-30, max=30)
                
                # Top-k
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits_filtered = torch.full_like(logits, -30.0)
                    logits_filtered.scatter_(0, top_k_indices, top_k_values)
                    logits = logits_filtered
                
                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = -30.0
                
                probs = F.softmax(logits, dim=-1)
                
                if torch.isnan(probs).any() or probs.sum() == 0:
                    break
                
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == self.end_id:
                    break
                
                next_token = next_token.view(1, 1)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        return current_tokens[0]