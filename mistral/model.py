import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable

import torch
from torch import nn
import torch.nn.functional as F
from simple_parsing.helpers import Serializable

from mistral.rope import precompute_freqs_cis, apply_rotary_emb
from mistral.cache import CacheView, RotatingBufferCache
from mistral.moe import MoeArgs, MoeLayer
import time

import torch_xla.core.xla_model as xm
from .xla_model_parallel import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)

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
    # If this is set, use sliding window attention rotating cache.
    sliding_window: Optional[int] = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe: Optional[MoeArgs] = None

 


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads # 32
        self.head_dim: int = args.head_dim # 128
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

    # TODO follow llama impl 
    # causal mask
    # - in Seq x Seq would be upper triangle filled with -inf
    # - in Seq

    # in mqa
    # Q => (seqlen, n_queries, k_dim)
    # K => (seqlen, k_dim)
    # V => (seqlen, v_dim)
    # scores = Q @ K.T => (seqlen, n_queries, seqlen)
    # scores += mask 
    # out = scores @ V => (seqlen, n_queries, v_dim)
    # out = out.view(seqlen, n_queries*v_dim) @ W => (seqlen, model_dim)

    # in gqa
    # Q => (seqlen, n_queries, k_dim)
    # K => (seqlen, n_queries_per_group, k_dim)
    # V => (seqlen, n_queries_per_group, v_dim)
    # scores = Q @ K.T => (seqlen, n_queries, n_queries_per_group)
    # scores += mask
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask,
        cache_kv: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # at enc, seqlen_sum == num_toks (all combined)
        seqlen_sum, dim = x.shape
        assert dim == self.args.dim

        world_sz = self.args.world_size
        W = cache.sliding_window
        s = seqlen_sum
        qh = self.n_heads // world_sz
        h = self.n_kv_heads // world_sz
        g = self.n_heads // self.n_kv_heads
        d = self.head_dim

        s0 = time.perf_counter()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(s, qh, d)
        xk = xk.view(s, h, d)
        xv = xv.view(s, h, d)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        e = time.perf_counter()
        xm.master_print(f'wq, wk, wv time {(e-s0)*1000:7.2f}')

        # enc
        if cache.prefill:
            # when not chunking, key, val == xk, xv
            s0 = time.perf_counter()
            key, val = cache.interleave_kv(xk, xv)
            e = time.perf_counter()
            xm.master_print(f'interleave tm {(e-s0)*1000:7.2f}')
            assert key.shape[0] == val.shape[0] == seqlen_sum
            cache.update(xk, xv)
        # dec
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            # each new seq of each batch has context of 4096 toks
            # assert seqlen_sum == B == len(prompts)
            key = key.view(
                seqlen_sum * W, self.n_kv_heads // world_sz, self.head_dim
            )
            val = val.view(
                seqlen_sum * W, self.n_kv_heads // world_sz, self.head_dim
            )
        # Repeat keys and values to match number of query heads
        s0 = time.perf_counter()
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        #xm.master_print(f'key shape {key.shape}')
        #xm.master_print(f'val shape {val.shape}')

        e = time.perf_counter()
        #xm.master_print(f'repeat tm {(e-s0)*1000:7.2f}')

        xq = xq.view(g,h,s,d)
        key = key.view(g*h, s if cache.prefill else s*W, d)
        val = val.view(g*h, s if cache.prefill else s*W, d)  # (4,1,28,128),(1,28,128) -> (4,28,128)
        s0 = time.perf_counter()
                              #
        scores = torch.einsum('ghnd,hsd->gns',xq,key) / self.scale # (1,28,28)
        e = time.perf_counter()
        #xm.master_print(f'score calc tm {(e-s0)*1000:7.2f}')

        # the mask is (4,28,28), but should be equivalent across heads, so
        # can just pick the first? 
        xm.master_print(f'score shape {scores.shape}')
        xm.master_print(f'cache shape {cache.mask.shape}')
        scores += cache.mask
        scores = scores.softmax(-1)
                            # (4,28,128),(1,28,128) 
        s0 = time.perf_counter()
        output = torch.einsum('gns,hsd->gnd',scores,val)
        e = time.perf_counter()
        #xm.master_print(f'weighted val calc tm {(e-s0)*1000:7.2f}')
        output = output.view(s,qh*d)

        # (28,128) @ (512,4096) 
        s0 = time.perf_counter()
        out = self.wo(output)
        e = time.perf_counter()
        #xm.master_print(f'out lin proj time {(e-s0)*1000:7.2f}')
        #xm.master_print(out.shape)
        return out 


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

        self.feed_forward: nn.Module
        if args.moe is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
                gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
                moe_args=args.moe,
            )
        else:
            self.feed_forward = FeedForward(args=args)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        s = time.perf_counter()
        x = self.attention_norm(x)
        e = time.perf_counter()
        #xm.master_print(f'norm tm {(e-s)*1000:7.2f}')

        s = time.perf_counter()
        r = self.attention.forward(x, freqs_cis, cache)
        e = time.perf_counter()
        #xm.master_print(f'attn tm {(e-s)*1000:7.2f}')

        h = x + r
        s = time.perf_counter()
        r = self.feed_forward.forward(self.ffn_norm(h))
        e = time.perf_counter()
        #xm.master_print(f'attn tm {(e-s)*1000:7.2f}')
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None
        assert self.vocab_size > 0
        assert pipeline_rank < num_pipeline_ranks, (pipeline_rank, num_pipeline_ranks)
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        # Modules specific to some ranks:
        self.tok_embeddings: Optional[nn.Embedding] = None
        self.norm: Optional[RMSNorm] = None
        self.output: Optional[nn.Linear] = None
        if pipeline_rank == 0:
            # NOTE xla
            #self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
            self.tok_embeddings = ParallelEmbedding(args.vocab_size,
                                                args.dim,
                                                init_method=lambda x:x,
                                                world_size=args.world_size,
                                                rank=args.rank,
                                                groups=args.groups)
        if pipeline_rank == num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            #self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
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
        for _ in range(args.n_layers):
            cache_k = torch.zeros((args.max_batch_size, args.max_seqlen, n_local_heads, args.head_dim)) 
            cache_v = torch.zeros((args.max_batch_size, args.max_seqlen, n_local_heads, args.head_dim)) 
            self.cache_kvs.append((cache_k,cache_v))
        
        freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2)
        self.register_buffer("freqs_cis", freqs_cis) 

        mask = torch.full(
        (1, 1, args.max_seq_len, args.max_seq_len),
        float("-inf")).to(torch.float)
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)
            
        layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(self.n_layers, offset + num_layers_per_rank)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
        self.n_local_layers = len(self.layers)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device



    def forward(
        self,
        tokens: torch.Tensor,
        input_indexes: torch.Tensor,
        output_index: torch.Tensor,
        cache_kvs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:

        (bsz, seqlen) = tokens.shape

        h = self.tok_embeddings(input_ids)
        xm.master_print(f'h shape {tokens.shape}')

        freqs_cis = self.freqs_cis.index_select(0, input_indexes)

        new_cache_kvs = []
        for local_layer_id, layer in zip(self.layers, cache_kvs):
            h, new_cache_kv = layer(h, freqs_cis, mask, input_indexes, cache_kv)
            new_cache_kvs.append(new_cache_kv)

        out = self.norm(h)
        out = out.index_select(1, output_index - input_indexes[0]).squeeze(dim=1)
        outs = self.output(h)
        return outs.float(), new_cache_kvs

    def load_state_dict(self, state_dict, *args, **kwargs):
        state_to_load = {}
        skipped = set([])
        for k, v in state_dict.items():
            if k.startswith("tok_embeddings"):
                if self.pipeline_rank == 0:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("norm") or k.startswith("output"):
                if self.pipeline_rank == self.num_pipeline_ranks - 1:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            elif k.startswith("layers"):
                layer_id = k.split(".")[1]
                if layer_id in self.layers:
                    state_to_load[k] = v
                else:
                    logging.debug(
                        "Skipping parameter %s at pipeline rank %d",
                        k,
                        self.pipeline_rank,
                    )
                    skipped.add(k)
            else:
                raise ValueError(f"Unexpected key {k}")
        assert set(state_dict.keys()) == skipped.union(set(state_to_load.keys()))
        super().load_state_dict(state_to_load, *args, **kwargs)
    
    # mistral parallelism
    '''
        each model shard gets total_layers // world_size layers based on rank
        rank = 0 shard embeds toks
            1) passes output of its layers to rank+1
            2) iterate until last rank
        last rank normalizes, and outputs as prob over vocab

        total time would be (time_per_shard) + (transmission_time_between_shard * (world_size-1))
    '''

    # how does llama prallelism differ?


    @staticmethod
    def from_folder(
        weight,
        folder: Path,
        rank: int,
        world_size: int,
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device="cuda",
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
        model = Transformer(model_args) #pipeline_rank=rank, num_pipeline_ranks=world_size)
        # mmap = True removed, now throwing err
        #loaded = torch.load(str(folder / "consolidated.00.pth"), map_location='cpu')
        # assign = True removed, now throwing err
        #loaded = torch.load(str(folder / "consolidated.00.pth"))
        model.load_state_dict(weight)
        return model.to(device=device, dtype=dtype)
