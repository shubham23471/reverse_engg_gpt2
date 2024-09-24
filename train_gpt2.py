from dataclasses import dataclass
import torch 
import torch.nn as nn 
from torch.nn import functional as F 

# variable name are same as the implementation 
# transformer from hugging face so we can exactly port over all the weights
# this is our implementation of GPT-2 and this allows us not to use the 
# transforms/src/transformer/models/gpt2/modeling_gpt2.py file from their repo
@dataclass
class GPTConfig:
    block_size: int = 256 
    vocab_size: int = 65 
    n_layer: int = 6 
    n_head: int = 6 
    n_embd: int = 384
    # # SAME A AS GPT-2
    # block_size: int = 1024 
    # vocab_size: int = 50257 # num of token: 50K BPE mergers + 256 bytes tokens + 1 <|endoftext|>
    # n_layer: int = 12 # num of layers
    # n_head: int = 12 # num of heads 
    # n_embd: int = 768 # embedding dimension 



#4
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # att @ v : like the weighted sum of values of the tokens that we found intresting at every 
        # single token  
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

# for 4 
# in the previous video we had that multi-headed attention module MultiHeadAttention(nn.Module)
# and the implementation of this class made it obivios that these heads are not complicated 
# they are basically in parallel inside every attention block and they are multiple head 
# and they function in parallel and their output are just being concat in in the forward pass 
# of this call (which we did in the previous video)

#3 
class MLP():
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 

#2 
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # communication b/w all the tokens
        x = x + self.attn(self.ln_1(x))
        # thinking about the collected information
        x = x + self.mlp(self.ln_2(x))
        return x 
    

############################################################################################################
# for 2
# residual
# You should have a clean residual stream: all the way from 
# supervision to all the way down to the inputs (tokens)
# This is desiable, because the gradients that flows from the top 
# flows straight to the inputs (the tokens) throught the residual pathways
# unchanged and in addition to that the gradient also throught the blocks 
# and the blocks  

# attention 
# remember attention is the communication operation. It is where all the tokens, where 
# 1024 lined up a sequence, and this is where the token communicate and exchange information. 
# So attention is a aggregation function, it's a pooling function, it's a weighted sum function
# it is a reduced operation. 
# where as MLP, happens to every single token individually. There is no information being collected 
# or exchanged b/w the tokens. This is where they think indivially, about the information that they 
# gathered 
# so, attention is the reduce and MLP is the map. So, what end up is the transformer end up being the 
# repeated application of map-reduce. 
############################################################################################################
# 1 
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # weights of token emb
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd), # additional final layer norm from GPT-2 paper
        ))
        # final classifier which projects n_embd to vocab_size without bias 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)