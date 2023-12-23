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


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads # 32
        self.head_dim: int = args.head_dim # 128
        self.n_kv_heads: int = args.n_kv_heads # 8

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5
        # args.dim = dim for each token
        #self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wq = ColumnParallelLinear( # shape (B, args.dim // world_size, args.n_heads*args.head_dim)
            args.dim,
            args.n_heads * args.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x:x,
            world_size=args.world_size,
            rank=args.rank,
            groups=args.groups,
            quant=args.quant,
        )
        #self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x:x,
            world_size=args.world_size,
            rank=args.rank,
            groups=args.groups,
            quant=args.quant,
        )

        #self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x:x,
            world_size=args.world_size,
            rank=args.rank,
            groups=args.groups,
            quant=args.quant,)

        #self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        # equiv to all attention from n_heads concatenated together to predict final logits
        self.wo = RowParallelLinear(
            args.n_heads * args.head_dim, # shape (B,args.dim//world_size,args.n_heads*args.head_dim)
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x:x,
            world_size=args.world_size,
            rank=args.rank,
            groups=None,
            quant=args.quant,
        )
        xm.master_print('wq wo shape')
        xm.master_print(self.wq.weight.shape)
        xm.master_print(self.wo.weight.shape)
        assert self.wq.weight.shape == (args.dim // world_size, args.n_heads*args.head_dim)
        assert self.wo.weight.shape == (args.n_heads*args.head_dim, args.dim//args.world_size,)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        # at enc, seqlen_sum == num_toks (all combined)
        seqlen_sum, dim = x.shape
        assert dim == self.args.dim
        
        # this lin step is performed seperately, by each core
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # NOTE 
        # due to gather_all sync step in the forward of parallel linear layers,
        # i would expect xq, xk, xv's shape to be equal to non-parallel shapes

        xm.master_print('q k v shape')
        xm.master_print(xq.shape) # (seqlen_sum, args.head_dim//world_size * args.n_heads)
        xm.master_print(xk.shape) # (seqlen_sum, args.head_dim//world_size * args.n_kv_heads)
        xm.master_print(xv.shape) # (seqlen_sum, args.head_dim//world_size * args.n_kv_heads)

        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        # enc
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        # dec
        else:
            assert seqlen_sum == 1
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim
            )

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        scores = torch.matmul(xq, xk.transpose(1,2)) / self.scale

        # add mask
        mask = cache.mask.transpose(2,0,1)
        xm.master_print(f'cache_mask shape {mask.shape}')
        scores += mask 
        # mask should be zeros and neg infs
        scores = scores.softmax(-1)
        output = torch.matmul(scores, values)

        '''
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        # TODO torch_xla incompile this, implement fast attention? 
        output = memory_efficient_attention(
            xq, key, val, None if cache is None else cache.mask
        )
        '''
        assert self.n_heads * self.heads_dim == self.args.dim # 4096
        out = self.wo(output.view(seqlen_sum, self.n_heads * self.head_dim))
        return out # (num_toks, self.args.dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
    
        #self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w1 = ColumnParallelLinear(args.dim,
                                       args.hidden_dim,
                                       bias=False,
                                       gather_output=False,
                                       init_method=lambda x:x,
                                       world_size=args.world_size,
                                       rank=args.rank,
                                       groups=args.groups,
                                       quant=args.quant)

        #self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w2 = RowParallelLinear(args.hidden_dim,
                                    args.dim,
                                    bias=False,
                                    input_is_parallel=True,
                                    init_method=lambda x:x,
                                    world_size=args.world_size,
                                    rank=args.rank,
                                    groups=args.groups,
                                    quant=args.quant)

        #self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)
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
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        # out.shape == [num_toks, args.dim]
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

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's  dtype means we cannot register it as a buffer
        if self._precomputed_freqs_cis is None:
            # If no sliding window, assume a larger seqlen
            theta = self.args.rope_theta
            if theta is None:
                theta = 1000000.0 if self.args.sliding_window is None else 10000.0
            # theta = 10000.
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta
            )
        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        return self._precomputed_freqs_cis

    def forward_partial(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        assert (
            len(seqlens) <= self.args.max_batch_size
        ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)
        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)

        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)
            xm.master_print(f'h shape {h.shape}')
        else:
            h = torch.empty(
                num_toks, self.args.dim, device=self.device, dtype=self.dtype
            )
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        freqs_cis = self.freqs_cis[input_metadata.positions]

        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            return h
        else:
            # Last rank has a final normalization step.
            assert self.norm is not None
            out = self.norm(h)
            return out

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        h = self.forward_partial(input_ids, seqlens, cache=cache)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # ignore the intermediate activations as we'll get the final output from
            # the last stage
            outs = torch.empty(
                h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            assert self.output is not None
            outs = self.output(h)
            #assert out.shape == [num_toks, self.args.vocab_size]
        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
        return outs.float()

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
        #weight,
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
        model = Transformer(model_args, pipeline_rank=rank, num_pipeline_ranks=world_size)
        # mmap = True removed, now throwing err
        #loaded = torch.load(str(folder / "consolidated.00.pth"), map_location='cpu')
        # assign = True removed, now throwing err
        loaded = torch.load(str(folder / "consolidated.00.pth"))
        model.load_state_dict(weight)
        return model.to(device=device, dtype=dtype)
