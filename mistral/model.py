import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import *
import traceback

import torch
from torch import nn
import torch.nn.functional as F
from simple_parsing.helpers import Serializable

#from mistral.rope import precompute_freqs_cis, apply_rotary_emb
#from mistral.cache import CacheView, RotatingBufferCache
from mistral.moe import MoeArgs, MoeLayer
import time

import torch_xla.core.xla_model as xm
from .xla_model_parallel import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

logging.basicConfig(level=logging.DEBUG)

@dataclass
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int

    # xla
    rank: int
    world_size: int 
    groups: Optional[List[int]]
    quant: bool
    max_batch_size: int = 0

    # For rotary embeddings. If not set, will be infered from sliding window.
    rope_theta: Optional[float] = None
    sliding_window: Optional[int] = None
    moe: Optional[MoeArgs] = None

    max_batch_sie: int = 32
    max_seqlen: int = 4096


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(
        xq.transpose(1, 2).reshape(-1, xq.shape[1], int(xq.shape[-1] / 2),
                                   2).float())
    xk_ = torch.view_as_complex(
        xk.transpose(1, 2).reshape(-1, xq.shape[1], int(xq.shape[-1] / 2),
                                   2).float())
    xq_out = torch.view_as_real(xq_ * freqs_cis)
    xk_out = torch.view_as_real(xk_ * freqs_cis)
    xq_out = xq_out.reshape(xq.shape[0], xq.shape[2], xq.shape[1],
                            xq.shape[3]).transpose(1, 2)
    xk_out = xk_out.reshape(xk.shape[0], xk.shape[2], xk.shape[1],
                            xk.shape[3]).transpose(1, 2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

    
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads # 32
        self.head_dim: int = args.head_dim # :15828
        self.n_kv_heads: int = args.n_kv_heads # 8

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5
        self.wq = ColumnParallelLinear( # shape (B, args.dim // world_size, args.n_heads*args.head_dim)
            args.dim,
            args.n_heads * args.head_dim, # split
            bias=False,
            gather_output=False,
            init_method=lambda x:x,
            world_size=args.world_size,
            rank=args.rank,
            groups=args.groups,
            quant=args.quant,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_kv_heads * args.head_dim, # split
            bias=False,
            gather_output=False,
            init_method=lambda x:x,
            world_size=args.world_size,
            rank=args.rank,
            groups=args.groups,
            quant=args.quant,
        )

        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_kv_heads * args.head_dim, # split
            bias=False,
            gather_output=False,
            init_method=lambda x:x,
            world_size=args.world_size,
            rank=args.rank,
            groups=args.groups,
            quant=args.quant,)

        self.wo = RowParallelLinear( 
            args.n_heads * args.head_dim, # split
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x:x,
            world_size=args.world_size,
            rank=args.rank,
            groups=None,
            quant=args.quant,
        )
        assert self.wq.weight.shape == (args.dim // args.world_size, args.n_heads*args.head_dim)
        assert self.wo.weight.shape == (args.n_heads*args.head_dim, args.dim//args.world_size,)

    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor], input_idexes: torch.Tensor,
                cache_kv: Tuple[torch.Tensor, torch.Tensor]):
        bsz, seqlen, _ = x.shape
        cache_k, cache_v = cache_kv
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        n_group_queries = self.n_heads // self.args.world_size 
        n_group_keys = self.n_kv_heads // self.args.world_size
        n_groups = self.n_heads // self.n_kv_heads

        # what is non local heads
        xq = xq.view(bsz, seqlen, n_group_queries, self.head_dim) # last 2 dims has args.dim // world_size # of params
        xk = xk.view(bsz, seqlen, n_group_keys, self.head_dim)
        xv = xv.view(bsz, seqlen, n_group_keys, self.head_dim)

        xm.master_print(f'applying rotarary embs')

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        cache_k = cache_k.index_copy(1, input_idexes, xk)
        cache_v = cache_v.index_copy(1, input_idexes, xv)

        keys = cache_k[:, :]
        values = cache_v[:, :]

        xm.master_print(f'repeating kvs')

        key,values = repeat_kv(keys,values,repeats=n_groups,dim=2)

        xm.master_print(f'calc attn')

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        xm.master_print(f'xq {xq.shape} keys {keys.shape} vals {values.shape}')
        scores = torch.einsum('bhsd,bhnd->bhsn') / self.scale
        xm.master_print(f'scores {scores.shape}') 
        xm.master_print(f'mask {mask.shape}') # mask shape (1,1,1,512)
        scores = scores + mask  # (bs, n_local_heads, slen, max_slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.einsum('bhsn,bhnd->bhsd')
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        xm.master_print(f'out {output.shape}') # (32,1,512)
        return self.wo(output), (cache_k, cache_v)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
    
        self.w1 = ColumnParallelLinear(args.dim,
                                       args.hidden_dim,
                                       bias=False,
                                       gather_output=False,
                                       init_method=lambda x:x,
                                       world_size=args.world_size,
                                       rank=args.rank,
                                       groups=args.groups,
                                       quant=args.quant)

        self.w2 = RowParallelLinear(args.hidden_dim,
                                    args.dim,
                                    bias=False,
                                    input_is_parallel=True,
                                    init_method=lambda x:x,
                                    world_size=args.world_size,
                                    rank=args.rank,
                                    groups=args.groups,
                                    quant=args.quant)

        self.w3 = ColumnParallelLinear(args.dim,
                                       args.hidden_dim,
                                       bias=False,
                                       gather_output=False,
                                       init_method=lambda x:x,
                                       world_size=args.world_size,
                                       rank=args.rank,
                                       groups=args.groups,
                                       quant=args.quant)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args
        self.feed_forward = FeedForward(args=args)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor], input_idxs: torch.Tensor,
                cache_kv: Tuple[torch.Tensor, torch.Tensor]):
        h, new_cache_kv = self.attention.forward(self.attention_norm(x), freqs_cis, mask, input_idxs, cache_kv)
        h = x + h
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None
        assert self.vocab_size > 0
        # Modules specific to some ranks:
        self.tok_embeddings: Optional[nn.Embedding] = None
        self.norm: Optional[RMSNorm] = None
        self.output: Optional[nn.Linear] = None
        self.tok_embeddings = ParallelEmbedding(args.vocab_size,
                                                args.dim,
                                                init_method=lambda x:x,
                                                world_size=args.world_size,
                                                rank=args.rank,
                                                groups=args.groups)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = ColumnParallelLinear(args.dim,
                                        args.vocab_size,
                                        bias=False,
                                        init_method=lambda x:x,
                                        world_size=args.world_size,
                                        rank=args.rank,
                                        groups=args.groups,
                                        quant=args.quant)

        # Initialize all layers but slice off those not of this rank.
        self.cache_kvs = []
        n_group_queries = args.n_heads // args.world_size
        for _ in range(args.n_layers):
            cache_k = torch.zeros((args.max_batch_size, args.max_seqlen, n_group_queries, args.head_dim)) 
            cache_v = torch.zeros((args.max_batch_size, args.max_seqlen, n_group_queries, args.head_dim)) 
            self.cache_kvs.append((cache_k,cache_v))

        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.n_local_layers = len(self.layers)
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, input_idxs: torch.Tensor,
        output_index: torch.Tensor, 
        cache_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
        freqs_cis, mask
        ):
        try:

            (bsz, seqlen) = tokens.shape

            h = self.tok_embeddings(tokens)
            freqs_cis = freqs_cis.index_select(0, input_idxs)
            mask = mask.index_select(2, input_idxs)

            xm.master_print(f'h device: {h.device}')
            
            new_cache_kvs = []
            for layer, cache_kv in zip(self.layers, cache_kvs):
                h, new_cache_kv = layer(h, freqs_cis, mask, input_idxs, cache_kv)
                new_cache_kvs.append(new_cache_kv)

            h = self.norm(h)
            out = h.index_select(1, output_index-input_idxs[0]).squeeze(dim=1)
            outs = self.output(h)
            return outs.float(), new_cache_kvs
        except Exception as e:
            tb = traceback.format_exc()
            xm.master_print(f'Raised Exception {e}')
            xm.master_print(tb)

    '''
    def load_state_dict(self, state_dict, *args, **kwargs):
        state_to_load = {}
        skipped = set([])
        for k, v in state_dict.items():
            if k.startswith("tok_embeddings"):
                state_to_load[k] = v
            elif k.startswith("norm") or k.startswith("output"):
                state_to_load[k] = v
            elif k.startswith("layers"):
                state_to_load[k] = v
        assert set(state_dict.keys()) == skipped.union(set(state_to_load.keys()))
        super().load_state_dict(state_to_load, *args, **kwargs)
    '''

    def from_folder(
        folder: Path,
        rank: int,
        world_size: int,
        max_batch_size: int = 1,
        device=None,
        dtype=torch.float16,
        quant=False
    ) -> "Transformer":
        with open(folder / "params.json", "r") as f:
            args = json.load(f)
            args = {**args, 
            'rank': rank, 'world_size': world_size, 'quant': quant,'groups': None}
            model_args = ModelArgs.from_dict(args)
        model_args.max_batch_size = max_batch_size
        xm.master_print(f'loading model... rank: {rank} world sz: {world_size}')
        torch.set_default_tensor_type(torch.BFloat16Tensor)
        model = Transformer(model_args)
        weights = load_split_weights(folder, world_size)
        model.load_state_dict(weights[rank], strict=False)
        xm.master_print(f'wq dev type {model.layers[0].attention.wq.weight.device}')
        model = model.to(device)
        for i in range(len(model.cache_kvs)):
            model.cache_kvs[i] = tuple(t.to(device) for t in model.cache_kvs[i])
        xm.master_print(f'wq dev type {model.layers[0].attention.wq.weight.device}')
        #torch.set_default_tensor_type(torch.FloatTensor)
        return model
