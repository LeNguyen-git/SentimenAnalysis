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



