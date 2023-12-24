from mistral.cache import RotatingBufferCache
import logging
import torch
import fire
from typing import List
from pathlib import Path

from mistral.model import Transformer
from mistral.tokenizer import Tokenizer

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from mistral.xla_model_parallel import get_model_parallel_rank, get_model_parallel_world_size
from typing import *
import re

def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


@torch.no_grad() # TODO set to no_grad
def generate(prompts: List[str], model: Transformer, tokenizer: Tokenizer, *, max_tokens: int,  temperature: float, chunk_size: int = None):
    model = model.eval()
    B, V = len(prompts), model.args.vocab_size

    # Tokenize
    encoded_prompts = [tokenizer.encode(prompt, bos=True) for prompt in prompts]
    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    # TODO what is cache window
    # TODO how does cache work
    cache_window = max(seqlens) + max_tokens
    if model.args.sliding_window is not None and cache_window > model.args.sliding_window:
        cache_window = model.args.sliding_window

    cache = RotatingBufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window, #4096 
        # parallel changes
        model.args.n_kv_heads // model.args.world_size,
        model.args.n_heads // model.args.world_size,
        model.args.head_dim,
    )

    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()
    xm.mark_step()
    
    # Bookkeeping
    logprobs = [[] for _ in range(B)]
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s:s+chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        assert B == len(seqlens) == len(prompts)
        # x will be reshaped to [B, chunk_size] in model
        x = torch.tensor(sum(prompt_chunks,[]), device=model.device, dtype=torch.long)
        (seqlens_sum,) = x.shape
        prelogits = model.forward(
            x,
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache
        )
        logits = torch.log_softmax(prelogits, dim=-1)
        xm.mark_step()

        if last_token_prelogits is not None:
            # Pass > 1
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(B):
                logprobs[i_seq].append(last_token_logits[i_seq, prompt_chunks[i_seq][0]].item())

        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            # get the probs of each seq
            logprobs[i_seq].extend([logits[offset + i, sequence[i + 1]].item() for i in range(len(sequence) - 1)])
            offset += len(sequence)

        # for all batches
        assert prelogits.shape == (seqlens_sum,V)
        # select last tok for all prompts
        last_token_prelogits = prelogits.index_select(0, torch.tensor([len(p) for p in prompt_chunks], device=prelogits.device).cumsum(dim=0) - 1)
        assert last_token_prelogits.shape == (B, V)
    
    # TODO implement streaming 
    # decode
    generated_tokens = []
    assert last_token_prelogits is not None
    for i_token in range(max_tokens):
        xm.master_print(f'generating {i_token}')
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())

        generated_tokens.append(next_token[:, None])
        # if i_token == 0, next_token == last tok of prompt
        # cache contains K,V context of all prompts 
        # next_tok is used as Q
        last_token_prelogits = model.forward(next_token, seqlens=[1] * len(prompts), cache=cache)
        xm.mark_step()
        assert last_token_prelogits.shape == (B, V)

    generated_words = []
    if generated_tokens:
        generated_tokens = torch.cat(generated_tokens, 1)
        for i, x in enumerate(encoded_prompts):
            generated_words.append(tokenizer.decode(x + generated_tokens[i].tolist()))

    return generated_words, logprobs

def setup_model_parallel() -> Tuple[int, int]:
    # assuming model parallelism over the whole world size
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()

    # seed must be the same in all processes
    torch.manual_seed(1)
    device = xm.xla_device()
    xm.set_rng_state(1, device=device)
    return rank, world_size

def interactive(model_path: str = '/home/minjunes/mistral-src/mistral-7B-v0.1', max_tokens: int = 35, temperature: float = 0.7, instruct: bool = False):
    rank, world_size = setup_model_parallel()
    device = xm.xla_device()
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), rank, world_size, max_batch_size=3, device=device)

    while True:
        prompt = 'what is the meaning of life?'#input("Prompt: ")
        if instruct:
            prompt = f"[INST] {prompt} [/INST]"
        res, _logprobs = generate(
            [prompt],
            transformer,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(res[0])
        print("=====================")

def _accelerate(idx,  model_path: str = '/home/minjunes/mistral-src/mistral-7B-v0.1', max_tokens: int = 35, temperature: float = 0, num_pipeline_ranks : int =1):
    rank, world_size = setup_model_parallel()
    weights = load_split_weights(model_path, world_size)
    device = xm.xla_device()
    xm.master_print("tokenizing")
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    xm.master_print("loading transformer")
    transformer = Transformer.from_folder(
        weights[rank], Path(model_path), rank, world_size, max_batch_size=3, device=device, dtype=torch.bfloat16
        )
    xm.master_print("generating")
    res, _logprobs = generate(
        [
            "This is a test",
            "This is another great test",
            "This is a third test, mistral AI is very good at testing. ",
        ],
        transformer,
        tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    for x,l in zip(res, _logprobs):
        xm.master_print(x)
        logging.debug('Logprobs: %s',l)
        xm.master_print("=====================")

def load_split_weights(folder, world_size):
    loaded = torch.load(str(Path(folder) / "consolidated.00.pth"), 
                map_location='cpu')
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
                #xm.master_print(f'{key} {split_tensor.shape}')
                assert split_tensor.shape[split_dim] == split_size
                res[i][key] = split_tensor
        else:
            for i in range(world_size):
                res[i][key] = value
    return res

def mp_main(model_path: str = '/home/minjunes/mistral-src/mistral-7B-v0.1', max_tokens: int = 4096, temperature: float = 0, num_pipeline_ranks=1):
    xmp.spawn(_accelerate, args=(model_path, max_tokens,temperature,num_pipeline_ranks))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(mp_main)
    exit(-1)
    fire.Fire({
        "interactive": interactive,
        "demo": demo,
    })
