import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math

with open("../data/UIT-VSFC/llama_vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

vocab_size = len(vocab)

# Embedding Layer thư viện nn
# torch_embedding = nn.Embedding(vocab_size, d_model)

class ManualEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.weight = nn.Parameter(torch.randn(vocab_size, d_model) * 0.1)

    def forward(self, input_ids):
        return F.embedding(input_ids, self.weight)
    

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for Rotary Position Embedding"
        self.d_model = d_model
        self.max_len = max_len

        self.sin_cache = None
        self.cos_cache = None

    def build_cache(self, x: torch.Tensor):
        if self.cos_cache is not None and x.shape[0] <= self.cos_cache.shape[0]:
            return
        
        seq_len = x.shape[0]

        #self.base = 10000
        theta = 1. / (10000 ** (torch.arange(0, self.d_model, 2, device=x.device).float() / self.d_model))

        seq_idx = torch.arange(seq_len, device=x.device).float()

        idx_theta = torch.einsum('i,j->ij', seq_idx, theta)

        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.sin_cache = idx_theta2.sin()[:, None, None, :]
        self.cos_cache = idx_theta2.cos()[:, None, None, :]

    def neg_half(self, x: torch.Tensor):
        d_model2 = self.d_model // 2
        return torch.cat([x[:, :, :, :d_model2], -x[:, :, :, d_model2:]], dim=-1)
    
    def forward(self, x: torch.Tensor, seq_len: int = None):
        self.build_cache(x)
        neg_half_x = self.neg_half(x)

        x_rope = (x * self.cos_cache[:x.shape[0]]) + (neg_half_x * self.sin_cache[:x.shape[0]])

        return x_rope


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, d_model, n_heads, max_len=1024):
#         super().__init__()
#         assert d_model % n_heads == 0

#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.head_dim = d_model // n_heads

#         self.q_proj = nn.Linear(d_model, d_model, bias=False)
#         self.k_proj = nn.Linear(d_model, d_model, bias=False)
#         self.v_proj = nn.Linear(d_model, d_model, bias=False)

#         self.rope = RotaryPositionEmbedding(self.head_dim, max_len)

#     def forward(self, x: torch.Tensor, attention_mask=None):
#         batch_size, seq_len, d_model = x.shape

#         q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
#         k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
#         v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

#         q = self.rope(q, seq_len)
#         k = self.rope(k, seq_len)

#         q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)

#         attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         # causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
#         # attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))

#         if attention_mask is not None:
#             attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
#             attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))

#         attention_weights = torch.softmax(attention_scores, dim=-1)

#         attention_output = torch.matmul(attention_weights, v)
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

#         return attention_output

class GroupedSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, num_groups, max_len=1024):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % num_groups == 0, "n_heads phải chia hết cho num_groups"

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_groups = num_groups
        self.heads_per_group = n_heads // num_groups
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionEmbedding(self.head_dim, max_len)

    def forward(self, x: torch.Tensor, attention_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        outputs = []
        for g in range(self.num_groups):
            start = g * self.heads_per_group
            end = start + self.heads_per_group

            q_g = q[:, start:end]  # (batch, group_heads, seq_len, dim)
            k_g = k[:, start:end]
            v_g = v[:, start:end]

            scores = torch.matmul(q_g, k_g.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_g)
            outputs.append(out)

        output = torch.cat(outputs, dim=1)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return output



class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        output = self.weight * self._norm(x.float()).type_as(x)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        hidden_dim = int (2 * hidden_dim / 3)

        # gate_proj --> bộ lọc
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)

         # up_proj --> nội dung
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)

        # down_proj --> đầu ra
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        # gate * up --> kết hợp nội dung và bộ lọc
        down = self.down_proj(gate * up)
        return down
        

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim, num_groups, max_len=1024):
        super().__init__()

        self.attention_norm = RMSNorm(d_model)
        # self.self_attention = MultiHeadSelfAttention(d_model, n_heads, max_len)
        self.self_attention = GroupedSelfAttention(d_model, n_heads, num_groups, max_len=max_len)
        self.feed_forward = FeedForward(d_model, hidden_dim)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, attention_mask=None):

        # Attention Block
        residual = x
        x = self.attention_norm(x)
        x = self.self_attention(x, attention_mask)
        x = x + residual

        # Feed Forward Block
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = x + residual

        return x


class ModelArgs:
    def __init__(
            self,
            vocab_size,
            d_model: int = 128,
            n_heads: int = 8,
            n_layers: int = 6,
            hidden_dim: int = 512,
            max_seq_len: int = 256,
            num_labels: int = None,
            num_groups: int = 4,
            num_topics: int = None
            
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels
        self.num_groups = num_groups
        self.num_topics = num_topics
        

class MiniLlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Token Embedding
        self.embedding = ManualEmbedding(args.vocab_size, args.d_model)

        #Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=args.d_model,
                n_heads=args.n_heads,
                hidden_dim=args.hidden_dim,
                num_groups=args.num_groups,
                max_len=args.max_seq_len,
            )for _ in range(args.n_layers)
        ])

        #Layer Norm
        self.norm = RMSNorm(args.d_model)

        self.num_labels = args.num_labels
        self.num_topics = args.num_topics

        # Classifier head
        if args.num_labels is not None: 
            self.classifier = nn.Linear(args.d_model, args.num_labels)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
        
        if args.num_topics is not None:
            self.topic_classifier = nn.Linear(args.d_model, args.num_topics)
            nn.init.xavier_uniform_(self.topic_classifier.weight)
            nn.init.zeros_(self.topic_classifier.bias)

        #Dropout
        self.dropout = nn.Dropout(0.1)

        #Khởi tao trọng số
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # if module.bias is not None:
            #     torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.xavier_uniform_(module)
            # torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None, topics=None):
        # Token Embedding
        x = self.embedding(input_ids)
        #dropout
        x = self.dropout(x)

        # Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)

        # Layer Norm
        x = self.norm(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            x_masked = x * mask
            pooled = x_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        else:
            pooled = x.mean(dim=1)
                
        output = {}
        
        # Calculate loss if labels are provided
        if self.num_labels is not None:
            logits = self.classifier(pooled)
            output['logits'] = logits
            if labels is not None:
                loss_function = nn.CrossEntropyLoss()
                output['loss'] = loss_function(logits, labels)
        if self.num_topics is not None:
            topic_logits = self.topic_classifier(pooled)
            output['topic_logits'] = topic_logits
            if topics is not None:
                loss_function = nn.CrossEntropyLoss()
                output['topics_loss'] = loss_function(topic_logits, topics)
        return output

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    


def main():
    args = ModelArgs(
        vocab_size=vocab_size,
        hidden_dim=512,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=128,
        num_labels=3,
        num_groups=4
    )

    model = MiniLlamaModel(args)  

    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")

if __name__ == "__main__":
    main()       