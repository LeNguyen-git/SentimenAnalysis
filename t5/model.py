import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math

class ManualEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        return F.embedding(input_ids, self.weight)
    
class PositionEmbedding(nn.Module):
    def __init__(self, max_positions, d_model):
        super().__init__()
        self.max_positions = max_positions
        self.d_model = d_model

        self.weight = nn.Parameter(torch.empty(max_positions, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, dtype=torch.long , device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        return F.embedding(position_ids, self.weight)
    

class T5Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_positions, dropout=0.1):
        super().__init__()
        self.token_embedding = ManualEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(max_positions, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, input_ids):
        tok_emb = self.token_embedding(input_ids)         
        pos_emb = self.position_embedding(input_ids)      
        x = tok_emb + pos_emb                             
        x = self.layernorm(x)                             
        x = self.dropout(x)                               
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, query, key, value, mask=None):
    #     batch_size, seq_len, _ = query.size()

    #     Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    #     K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    #     V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    #     scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

    #     if mask is not None:
    #         scores = scores.masked_fill(mask == 0, float('-inf'))

    #     attn_weights = F.softmax(scores, dim=-1)
    #     attn_weights = self.dropout(attn_weights)

    #     context = torch.matmul(attn_weights, V)

    #     context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    #     output = self.o_proj(context)

    #     return output

    def forward(self, query, key, value, mask=None):
        batch_size, target_len, _ = query.size()
        src_len = key.size(1)

        Q = self.q_proj(query).view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            assert mask.shape[-1] == scores.shape[-1], f"mask shape {mask.shape}, scores shape {scores.shape}"
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, target_len, self.d_model)
        output = self.o_proj(context)

        return output, attn_weights
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model,hidden_dim)
        self.dropout = nn.Dropout(dropout)   
        self.linear2 = nn.Linear(hidden_dim,d_model)

    def forward(self, x: torch.Tensor):

        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
    
class EncodeLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, hidden_dim, dropout)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.layernorm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.layernorm2(x)

        return x
    
class DecodeLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, hidden_dim, dropout)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = x + self.dropout(attn_output)
        x = self.layernorm1(x)

        cross_output, _ = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
        x = x + self.dropout(cross_output)
        x = self.layernorm2(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.layernorm3(x)
        
        return x
    

class T5Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, hidden_dim, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embedding = T5Embedding(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncodeLayer(d_model, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_ids, mask=None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.layernorm(x)

        return x
    
class T5Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, hidden_dim, max_len, dropout=0.1):
        super().__init__()

        self.embedding = T5Embedding(vocab_size, d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecodeLayer(d_model, num_heads, hidden_dim, dropout)
            for _ in range (num_layers)
        ])

        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input_ids, encoder_output, self_mask=None, cross_mask=None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
        
        x = self.layernorm(x)

        return x
    
class T5Model(nn.Module):
    def __init__(self, args): 
        super().__init__()

        self.encoder = T5Encoder(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            max_len=args.max_len,
            dropout=args.dropout
        )

        self.decoder = T5Decoder(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            max_len=args.max_len,
            dropout=args.dropout
        )

        self.output_proj = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.pad_idx = args.pad_idx
        self.init_weight()

    def init_weight(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def create_padding_mask(self, seq, pad_idx=0):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool().to(torch.bool)
        return ~mask
    
    def forward(self, src_input_ids, target_input_ids, pad_idx=0):
          
        pad_idx = self.pad_idx
        
        src_mask = self.create_padding_mask(src_input_ids,pad_idx)

        target_padding_mask = self.create_padding_mask(target_input_ids, pad_idx)
        look_ahead_mask = self.create_look_ahead_mask(target_input_ids.size(1)).to(target_input_ids.device)
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1)

        target_mask = target_padding_mask & look_ahead_mask

        encoder_output = self.encoder(src_input_ids, src_mask)

        decoder_output = self.decoder(target_input_ids, encoder_output, self_mask=target_mask, cross_mask=src_mask)

        logits = self.output_proj(decoder_output)

        return logits

class ModelArgs:
    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        num_heads=8,
        num_layers=6,
        hidden_dim=512,
        max_len=512,
        dropout=0.1,
        pad_idx=0
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.dropout = dropout
        self.pad_idx = pad_idx








        
