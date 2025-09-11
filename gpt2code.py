import torch
from torch import nn


GPT_2_CONFIG = {
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qvk_bias': False
}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_head, context_length, drop_rate, qvk_bias=False):
        super().__init__()

        assert d_out%num_head == 0, "d_out should be divisible by number of heads"

        self.head_dim = int(d_out/num_head) 
        self.d_out = d_out
        self.num_heads = num_head

        self.W_query = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qvk_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qvk_bias)

        self.drop_out = nn.Dropout(p=drop_rate)

        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))

        attn_weigths = torch.softmax(attn_scores/keys.shape[-1]**(0.5), dim=-1)
        attn_weigths = self.drop_out(attn_weigths)

        context_vec = (attn_weigths @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        return context_vec
        

class Layernorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * x + self.shift 
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.norm_1 = Layernorm(cfg['emb_dim'])
        self.norm_2 = Layernorm(cfg['emb_dim'])

        self.attn = MultiHeadAttention(cfg['emb_dim'], cfg['emb_dim'], cfg['n_heads'], cfg['context_length'], cfg['drop_rate'])

        self.drop_out = nn.Dropout(cfg['drop_rate'])

        self.ff = FeedForward(cfg)

    def forward(self, x):

        norm_1_outputs = self.norm_1(x)
        attn_outputs = self.attn(norm_1_outputs)
        dropout_outputs = self.drop_out(attn_outputs)
        x += dropout_outputs

        norm_2_outputs = self.norm_2(x)
        feed_forward_outputs = self.ff(norm_2_outputs)
        dropout_outputs = self.drop_out(feed_forward_outputs)
        x += dropout_outputs

        return x 
    

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_embs = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_embs = nn.Embedding(cfg['context_length'], cfg['emb_dim'])

        self.trf_block = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = Layernorm(cfg['emb_dim'])

        self.out_proj = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

        self.drop_out = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        b, num_tokens = x.shape

        token_embs = self.token_embs(x)
        pos_embs = self.pos_embs(torch.arange(num_tokens, device=x.device))
        input_embs = token_embs + pos_embs

        dropout_outputs = self.drop_out(input_embs)

        block_outputs = self.trf_block(dropout_outputs)

        norm_outputs = self.final_norm(block_outputs)

        logits = self.out_proj(norm_outputs)

        return logits
    

model = GPTModel(cfg=GPT_2_CONFIG)
    