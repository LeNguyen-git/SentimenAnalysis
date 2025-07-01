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
        possition_ids = torch.arange(seq_len, dtype=torch.long , device=input_ids.device)
        possition_ids = possition_ids.unsqueeze(0).expand(batch_size, seq_len)

        return F.embedding(possition_ids, self.weight)


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, d_model):
        super().__init__()
        self.type_vocab_size = type_vocab_size
        self.d_model = d_model

        self.weight = nn.Parameter(torch.empty(type_vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_type_ids):
        return F.embedding(token_type_ids, self.weight)
    
class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_positions=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.token_embedding = ManualEmbedding(vocab_size, d_model)
        self.posstion_embedding = PositionEmbedding(max_positions, d_model)
        self.segment_embedding = SegmentEmbedding(type_vocab_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.posstion_embedding(input_ids)
        segment_embeddings = self.segment_embedding(token_type_ids)

        embeddings = token_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_linear = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask=None):
        batch_size = x.size(0)

        q = self.q_proj(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) 
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))


        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out_linear(context)
        
        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        hidden_dim = hidden_dim or 4 * d_model

        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor):

        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class BertLayer(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask=None):
        attn_output = self.attention(x, attention_mask)
        x = x + self.dropout1(attn_output)
        x = self.layer_norm1(x)

        ffn_output = self.feed_forward(x)
        x = x + self.dropout2(ffn_output)
        x = self.layer_norm2(x)

        return x

class BertEncoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BertLayer(d_model, n_heads, hidden_dim, dropout) for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor, attention_mask=None):

        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class BertPooler(nn.Module):
    def __init__(self, d_model):

        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):

        first_token_tensor = hidden_states[:, 0]  # [CLS]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return pooled_output

class ModelArgs:
    def __init__(self, 
                 vocab_size, 
                 d_model=256, 
                 n_layers=6, 
                 n_heads=8, 
                 hidden_dim=512, 
                 max_positions=512, 
                 type_vocab_size=2,
                 dropout=0.1
    ):
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.max_positions = max_positions
        self.type_vocab_size = type_vocab_size
        self.dropout = dropout

class BertModel(nn.Module):
    def  __init__(self, args: ModelArgs, num_labels=None):
        super().__init__()
        self.args = args

        self.embedding = BertEmbedding(
            vocab_size= args.vocab_size,
            d_model= args.d_model, 
            max_positions= args.max_positions, 
            type_vocab_size=args.type_vocab_size, 
            dropout=args.dropout
        )

        self.encoder = BertEncoder(
            d_model=args.d_model, 
            n_layers=args.n_layers, 
            n_heads=args.n_heads, 
            hidden_dim=args.hidden_dim, 
            dropout=args.dropout
        )

        self.pooler = BertPooler(d_model=args.d_model)
        self.num_labels = num_labels

        if num_labels is not None:
            self.classifier = nn.Linear(args.d_model, num_labels)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        
        embeddings = self.embedding(input_ids, token_type_ids)
        encoder_output = self.encoder(embeddings, attention_mask)
        pooled_output = self.pooler(encoder_output)

        if self.num_classes is not None:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits
        return encoder_output, pooled_output

