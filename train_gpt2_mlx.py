from dataclasses import dataclass
from collections import OrderedDict
import mlx
import mlx.core as mx
import mlx.nn as nn
import math

#-------------------------------------------------
def get_state_dict(model):
    # Assuming your model's parameters are stored in an OrderedDict as shown
    state_dict = {}
    for name, module in model.transformer.items():
        if isinstance(module, list):
            for idx, sub_module in enumerate(module):
                state_dict[f'{name}.{idx}.weight'] = sub_module.weight
                if hasattr(sub_module, 'bias') and sub_module.bias is not None:
                    state_dict[f'{name}.{idx}.bias'] = sub_module.bias
        else:
            state_dict[f'{name}.weight'] = module.weight
            if hasattr(module, 'bias') and module.bias is not None:
                state_dict[f'{name}.bias'] = module.bias
    state_dict['lm_head.weight'] = model.lm_head.weight
    return state_dict

@dataclass
class GPTConfig:
    block_size: int = 1024  # Length of input sequence (context length)
    vocab_size: int = 50257 # Number of possible input tokens (50,000 BPE tokens + 256 bytes tokens + 1 <|endoftext|>)
    n_layer: int = 12       # Number of transformer blocks
    n_head: int = 12        # Number of heads for multi-head attention
    n_embd: int = 768       # Dimension of embeddings


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, Query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
                # Bias buffer (mask) - manually handled
        self.bias = mx.tril(mx.ones((config.block_size, config.block_size))).reshape(1, 1, config.block_size, config.block_size)
        # self.register_buffer("bias", mx.tril(mx.ones(config.block_size, config.block_size))
        #                      .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all head in batch and move head forward to be the batch dim
        # nh is "number of heads",
        # hs is "head size",
        # C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for all the queries and keys)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = mx.fast.scaled_dot_product_attention(q, k, v) #, is_causal=True in pytorch

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approx='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # self.transformer = OrderedDict(
        #     # Token Embedding Layer
        #     wte=nn.Embedding(config.vocab_size, config.n_embd),
        #     # Position Embedding Layer
        #     wpe=nn.Embedding(config.block_size, config.n_embd),
        #     # The actual Transformer blocks
        #     h=[Block(config) for _ in range(config.n_layer)],
        #     # Final LayerNorm before the output
        #     lm_f=nn.LayerNorm(config.n_embd),
        # )
        self.transformer = OrderedDict([
            ('wte', nn.Embedding(config.vocab_size, config.n_embd)),  # Token Embedding Layer
            ('wpe', nn.Embedding(config.block_size, config.n_embd)),  # Position Embedding Layer
            ('h', [Block(config) for _ in range(config.n_layer)]),     # Transformer blocks
            ('lm_f', nn.LayerNorm(config.n_embd))                     # Final LayerNorm before the output
        ])
        
        # The Precition Head, Linear Layer but w/o Bias.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer['wte'].weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights) # iterates over every sub module and applies the init weights on them

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                mx.zeros_like(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal(module.weight, mean=0.0, std=std)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        
        pos = mx.arange(0, T, dtype=mx.int32)  # Positional indices
        pos_emb = self.transformer['wpe'](pos)  # Positional embeddings
        tok_emb = self.transformer['wte'](idx)  # Token embeddings
        x = tok_emb + pos_emb
        
        for block in self.transformer['h']:
            x = block(x)
        
        x = self.transformer['lm_f'](x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = get_state_dict(model)
        # sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                sd[k] = sd_hf[k].transpose() 
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                sd[k] = sd_hf[k]

        return model

model = GPT.from_pretrained("gpt2")
print("didnt crash, yay!")