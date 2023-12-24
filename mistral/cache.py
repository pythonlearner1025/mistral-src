import torch
from typing import List, Tuple
from dataclasses import dataclass
import torch_xla.core.xla_model as xm

from .mask_utils import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)

@dataclass
class RotatingCacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor # unused
    # which elements in the sequences need to be cached
    to_cache_mask: torch.Tensor
    # how many elements are cached per sequence
    cached_elements: torch.Tensor # unused
    # where tokens should go in the cache
    cache_positions: torch.Tensor 

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]



class CacheView:
    def __init__(self, cache_k: torch.Tensor, cache_v: torch.Tensor, metadata: RotatingCacheInputMetadata, kv_seqlens: torch.Tensor):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata
        xm.master_print(f'in cacheview')
        xm.master_print(f'cache_kv shape {cache_k.shape}')
    
    def update(self, xk: torch.Tensor, xv: torch.Tensor):
        """
        to_cache_mask masks the last [sliding_window] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        # flatten by max_batch_size * sliding_window in first dim
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

        # at prefill
        #xm.master_print(f'cache pos {self.metadata.cache_positions.shape}')
        #xm.master_print(f'cache mask {self.metadata.cache_mask.shape}') # to mask or not mask each seq. 
                                                                    # only copy items to be masked??
        assert self.metadata.cache_positions.shape[0] == self.metadata.cache_mask.shape[0]
        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk[self.metadata.to_cache_mask])
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv[self.metadata.to_cache_mask])

    def interleave_kv(self, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3 # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)] respective to each prompt sequence
        xk = torch.split(xk, self.metadata.seqlens)
        xv = torch.split(xv, self.metadata.seqlens)
        assert len(xk) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Order elements in cache by position by unrotating
        xm.master_print(f'cache shape')
        xm.master_print(self.cache_k.shape)
        xm.master_print(f'kv_seqlens')
        xm.master_print(self.kv_seqlens) # if first prefill [0] * seqlens
        # NOTE only if we don't set prompt chunk_size in main
        assert sum(self.kv_seqlens) == 0

        assert self.cache_k.ndim == 4 and len(self.cache_k) == len(xk) == len(self.kv_seqlens)
        batch_size = self.cache_k.shape[0]

        def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
            assert cache.ndim == 3  # (W, H, D)
            W = cache.shape[0]
            position = seqlen % W
            if seqlen < W: # if in W, return cached tokens till :seqlen
                return cache[:seqlen]
            elif position == 0: #seqlen == 0, or seqlen == W
                return cache # empty or full, either way return all toks
            else: # seqlen > W 
                # suppose seqlen = 16 and W = 10
                # then we return cache of size 10, where cache[6:] (4) + cache[:6] (6)
                # then in interleave_list we interleave thi
                return torch.cat([cache[position:], cache[:position]], dim=0)

        # at prefill, cache_k is just empty lists since cache[:0]
        # right after attn computation, kv_seqlens incremented with seqlens_0 used for attn
        cache_k = [unrotate(t, s) for t, s in zip(self.cache_k, self.kv_seqlens)]
        cache_v = [unrotate(t, s) for t, s in zip(self.cache_v, self.kv_seqlens)]

        assert len(cache_k) == len(cache_v) == batch_size

        interleaved_k = [v for pair in zip(cache_k, xk) for v in pair]
        interleaved_v = [v for pair in zip(cache_v, xv) for v in pair]
        
        # but their dims are not the same?
        xm.master_print(f'sanity check')
        assert sum([v.shape[0] for v in interleaved_k]) == sum(self.metadata.seqlens)
        assert len(interleaved_k) == 2*len(cache_k) 

        out_k, out_v = torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)
        xm.master_print(f'interleaved cache shapes')
        xm.master_print(f'k {out_k.shape}, v {out_v.shape}')
        return out_k, out_v

    @property
    def sliding_window(self):
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[:len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[:len(self.kv_seqlens)]

    @property
    def prefill(self):
        return self.metadata.prefill

    @property
    def mask(self):
        return self.metadata.mask

class RotatingBufferCache:
    """
    This is an example that implements a less naive rotating buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """
    def __init__(self, n_layers: int, max_batch_size: int, sliding_window: int, n_kv_heads: int, n_heads: int, head_dim: int):

        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        self.cache_v = torch.empty((
            n_layers,
            max_batch_size,
            sliding_window,
            n_kv_heads,
            head_dim
        ))
        # holds the valid length for each batch element in the cache
        self.kv_seqlens = None
        self.n_heads = n_heads
        self.head_dim = head_dim

    # cache_view for when parallel by # layers
    # what about parallel by 
    def get_view(self, layer_id: int, metadata: RotatingCacheInputMetadata) -> CacheView:
        return CacheView(self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens)

    def reset(self):
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int):
        self.kv_seqlens = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

    @property
    def device(self):
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype):
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    # called in forward
    def update_seqlens(self, seqlens: List[int]):
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    # when embedding prompt for first time, 
    # first prefill = True
    # with each subseqent forward() call in decoder self.kv_seqlens inc by 1 
    def get_input_metadata(self, seqlens: List[int]) -> RotatingCacheInputMetadata:
        """
            inpput = seqlens [5,7,2] // seqpos [0, 1, 3] // sliding_window 3
            --> only cache last 3 tokens in each sequence
            - to_cache_mask = [0 0 1 1 1 | 0 0 0 0 1 1 1 | 1 1]
            - cached_elements = [3 | 3 | 2]
            --> absolute positions are used for rope
            - positions = [0 1 2 3 4 | 1 2 3 4 5 6 7 | 3 4]
            --> cache positions are positions cache_masked, modulo sliding_window + batch_idx * sliding_window
            - cache_positions = [2 0 1 | 5 3 4 | 6 7]
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))
        assert len(seqlens) == len(self.kv_seqlens), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist()

        assert len(seqlens) > 0, seqlens
        # if seqlen > self.sliding_window, then don't add to cache (context)
        # the first seqlen-self.sliding_window tokens. they are omitted. 
        masks = [
            [x >= seqlen - self.sliding_window for x in range(seqlen)]
            for seqlen in seqlens
        ]
        
        # [True] * sum(seqlens)
        to_cache_mask = torch.tensor(sum(masks, []), device=self.device, dtype=torch.bool)
        xm.master_print(f'to_cache_mask {to_cache_mask}')
        # at encoding, [n_masked_0, ... ,n_masked_seqlens]
        # at decoding, [1,...,1]
        cached_elements = torch.tensor([sum(mask) for mask in masks], device=self.device, dtype=torch.long)
        # at prefill, each pos = [0, ... ,seqlen-1]
        # during decode, each pos = [n_total_seqs]
        positions = torch.cat([torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]).to(device=self.device, dtype=torch.long)
        batch_idx = torch.tensor(sum([[i]*seqlen for i, seqlen in enumerate(seqlens)], []), device=self.device, dtype=torch.long)
        # len(cache[0]) == len(seqlens) * self.sliding_window
        cache_positions = positions % self.sliding_window + batch_idx * self.sliding_window

        xm.master_print(f'cache_positions {cache_positions}')

        xm.master_print(f'cache_positions indexed {cache_positions[to_cache_mask]}')

        first_prefill = all([pos == 0 for pos in seqpos])
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)

        # TODO how slow is mask.materialize and can it be optimized? 
        if first_prefill:
            assert all([pos == 0 for pos in seqpos]), (seqpos)
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(self.sliding_window)
            shape = (self.n_heads, sum(seqlens), sum(seqlens))
            xm.master_print(f'seqlens {sum(seqlens)}')
            mask = mask.materialize(shape, device=self.device, dtype=torch.bfloat16)
            xm.master_print(f'made mask shape {mask.shape}')

        # subsequent encodings
        # unless we set chunk_size, this is never called
        elif subsequent_prefill:
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[s + cached_s.clamp(max=self.sliding_window).item() for (s, cached_s) in zip(seqlens, self.kv_seqlens)]
            ).make_local_attention_from_bottomright(self.sliding_window)
            shape = (self.n_heads, self.n_kv_heads, sum(seqlens))
            mask = mask.materialize(shape, device=self.device, dtype=torch.bfloat16)
        # decoding
        else:
            assert all([n_seq == 1 for n_seq in seqlens]), (seqlens)
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                # upper bound for each individual key len
                kv_padding=self.sliding_window,
                                            # inc by 1              # max seq size == W (4096)
                kv_seqlen=(self.kv_seqlens + cached_elements).clamp(max=self.sliding_window).tolist()
            )
            shape = (self.n_heads, self.n_kv_heads, sum(seqlens))
            mask = mask.materialize(shape, device=self.device, dtype=torch.bfloat16)

        return RotatingCacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions[to_cache_mask],
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )
