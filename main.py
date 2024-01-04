from mistral.cache import RotatingBufferCache
import logging
import torch
import fire
from typing import List
from pathlib import Path
import json

from mistral.model import Transformer, ModelArgs
from mistral.tokenizer import Tokenizer

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from mistral.xla_model_parallel import get_model_parallel_rank, get_model_parallel_world_size
from typing import *
import re, time, os

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_sum - probs_sort) > p
    probs_sort = torch.where(mask, 0.0, probs_sort)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

class Generator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_tok = self._gen_tok

    def _gen_tok(self, toks, input_toks, 
            input_text_mask, cur_pos, input_pos,
            output_pos, cache_kvs, temp,
            top_p, with_temp, freqs_cis, mask):
        logits, cache_kvs = self.model(input_toks, input_pos,
                                        output_pos, cache_kvs, freqs_cis, mask)
        # sample
        if with_temp:
            probs = torch.softmax(logits / temp, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        input_text_mask_tmp = input_text_mask.index_select(1, cur_pos).squeeze(dim=1)
        toks_tmp = toks.index_select(1, cur_pos).squeeze(dim=1)
        next_token = torch.where(input_text_mask_tmp, toks_tmp, next_token)
        next_token = next_token.unsqueeze(1)
        toks = toks.index_copy(1, cur_pos, next_token)
        # prepare for the next iteration
        input_pos = cur_pos.unsqueeze(0)
        cur_pos = cur_pos + 1
        output_pos =  output_pos - 1 
        input_toks = toks.index_select(1, input_pos)

        return toks, input_toks, cur_pos, input_pos, output_pos, cache_kvs

    @torch.no_grad() 
    def generate(self, prompts: List[str], model: Transformer, tokenizer: Tokenizer, *, max_toks: int,  temperature: float, chunk_size: int = None, top_p: float = 0.8):
        xm.master_print(f'in generate...')
        args = model.args
        device = xm.xla_device()
        model = model.eval()
        B, V = len(prompts), model.args.vocab_size

        # Tokenize
        encoded_prompts = [self.tokenizer.encode(prompt, bos=True) for prompt in prompts]
        seqlens = [len(x) for x in encoded_prompts]

        xm.mark_step()

        # One chunk if size not specified
        max_prompt_size = max(seqlens)
        min_prompt_size = min(seqlens)

        # fix token size
        toks = torch.full((args.max_batch_size, args.max_seqlen), self.tokenizer.pad_id).long()

        # fill in 
        for i,prompt in enumerate(encoded_prompts):
            toks[i, :len(prompt)] = torch.tensor(prompt).long()
        toks = toks.to(device)
        input_text_mask = toks != self.tokenizer.pad_id

        # Passing tensors instead of floats gen_tok 
        # so that different values would not trigger compilations of new graphs 
        temp = torch.tensor(float(temperature)).to(device)
        top_p = torch.tensor(float(top_p)).to(device)
        with_temp = top_p > 0

        cache_kvs = model.cache_kvs
        xm.mark_step()
        # Encode prompt by chunks
        prev_pos = 0
        buckets = [128,256,384,512]

        freqs_cis = precompute_freqs_cis(
                args.dim // args.n_heads,
                args.max_seqlen * 2).to(device)
        mask = torch.full(
            (1, 1, args.max_seqlen, args.max_seqlen),
            float("-inf")).to(torch.float)
        mask = torch.triu(mask, diagonal=1).to(device)

        xm.master_print(f'encoding prompts')
        while prev_pos < min_prompt_size:
            remaining = min_prompt_size - prev_pos
            section_len = 0
            for bucket in buckets:
                if bucket >= remaining:
                    section_len = bucket
                    break
            if section_len == 0:
                section_len = buckets[-1]

            cur_pos = min(min_prompt_size, prev_pos + section_len)
            cur_pos = torch.tensor(cur_pos).to(device)
            input_pos = torch.arange(prev_pos, prev_pos + section_len).to(device)
            output_pos = cur_pos - 1
            input_toks = toks.index_select(1, input_pos)
            xm.mark_step()
            toks, input_toks, cur_pos, input_pos, output_pos, cache_kvs \
                = self.gen_tok(
                    toks, input_toks, input_text_mask, cur_pos,
                    input_pos, output_pos, cache_kvs,
                    temp, top_p, with_temp, freqs_cis, mask
                )
            xm.mark_step()

            prev_pos = cur_pos

        xm.master_print(f'done encoding prompts')

        assert cur_pos.item() == prev_pos + 1 and prev_pos == min_prompt_size
        for i in tqdm(t:=range(prev_pos + 1, total_len)):
            xm.master_print(f'gen tok {i}')
            toks, input_toks, cur_pos, input_pos, output_pos, cache_kvs \
                = self.gen_tok(
                    toks, input_toks, input_text_mask, cur_pos,
                    input_pos, output_pos, cache_kvs,
                    temp, top_p, with_temp, freqs_cis, mask
                )
            xm.mark_step()
        self.model.cache_kvs = cache_kvs

        decoded = []
        for i, t in enumerate(toks.tolist()):
            if i >= len(prompt_toks):
                break
            # cut to max gen len
            t = t[:len(prompt_toks[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[:t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            try:
                sentence = self.tokenizer.decode(t)
            except IndexError:
                sentence = self.tokenizer.decode(t[1:])
            decoded.append(sentence)
        return decoded

def setup_model_parallel() -> Tuple[int, int]:
    # assuming model parallelism over the whole world size
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    # seed must be the same in all processes
    generator = torch.Generator()
    generator.manual_seed(1)
    device = xm.xla_device()
    xm.set_rng_state(1, device=device)
    return rank, world_size

def load_split_weights(folder, world_size, device):
    loaded = torch.load(str(Path(folder) / "consolidated.00.pth"), map_location='cpu')
    split_dims = {
        'tok_embeddings.weight': -1,
        'output.weight': -2,
        '(wq|wk|wv).weight': -2,
        'wo.weight': -1,
        'w1.weight': -2,
        'w2.weight': -1,
        'w3.weight': -2,
    }
    res = [dict() for _ in range(world_size)]
    for key, value in loaded.items():
        split_dim = None
        for pattern, dim in split_dims.items():
            if re.search(pattern, key):
                split_dim = dim
                break
        if split_dim is not None:
            split_size = value.size(split_dim) // world_size
            split_tensors = value.split(split_size, dim=split_dim)
            for i, split_tensor in enumerate(split_tensors):
                assert split_tensor.shape[split_dim] == split_size
                res[i][key] = split_tensor
        else:
            for i in range(world_size):
                res[i][key] = value
    return res

def load(
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
        xm.master_print(f'loading checkpoint at:  ckpt_rank_{rank}')
        weight = torch.load(str(Path(folder) / f"ckpt_rank_{rank}.pth"))
        model.load_state_dict(weight, strict=False)
        xm.master_print(f'wq dev type {model.layers[0].attention.wq.weight.device}')
        xm.master_print(f'moving to device {device}')
        model = model.to(device)
        for i in range(len(model.cache_kvs)):
            model.cache_kvs[i] = tuple(t.to(device) for t in model.cache_kvs[i])
        xm.master_print(f'wq dev type {model.layers[0].attention.wq.weight.device}')
        #torch.set_default_tensor_type(torch.FloatTensor)
        return model

def _accelerate(idx, model_path: str = '/home/minjunes/mistral-src/mistral-7B-v0.1', max_toks: int = 35, temperature: float = 0, num_pipeline_ranks : int =1):
    rank, world_size = setup_model_parallel()
    device = xm.xla_device()
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = load(
        Path(model_path), rank, world_size, max_batch_size=3, device=device, dtype=torch.bfloat16
    )
    generator = Generator(transformer, tokenizer)

    with torch.no_grad():
        res = generator.generate(
            ["the meaning of life is"],
            transformer,
            tokenizer,
            max_toks=max_toks,
            temperature=temperature,
            top_p=0.8
        )

    for x in res:
        xm.master_print(x)
        xm.master_print("=====================")


def mp_main(model_path: str = '/home/minjunes/mistral-src/mistral-7B-v0.1', max_toks: int = 4096, temperature: float = 0, num_pipeline_ranks=1):
    xmp.spawn(_accelerate, args=(model_path, max_toks,temperature,num_pipeline_ranks))

if __name__ == "__main__":
    os.environ['PJRT_DEVICE'] = 'TPU'
    logging.basicConfig(level=logging.INFO)
    fire.Fire(mp_main)
