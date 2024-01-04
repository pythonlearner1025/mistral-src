import torch
from pathlib import Path
import re

def load_split_weights(folder, world_size):
    print('loading weights...')
    loaded = torch.load(str(Path(folder) / "consolidated.00.pth"), map_location='cpu')
    print('done!')
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
            print('splitting...')
            split_size = value.size(split_dim) // world_size
            split_tensors = value.split(split_size, dim=split_dim)
            for i, split_tensor in enumerate(split_tensors):
                assert split_tensor.shape[split_dim] == split_size
                res[i][key] = split_tensor
        else:
            for i in range(world_size):
                res[i][key] = value
    return res

def save_split_weights(weights, folder, world_size):
    for i in range(world_size):
        weight = weights[i]
        checkpoint_path = Path(folder) / f'ckpt_rank_{i}.pth'
        torch.save(weight, checkpoint_path)
        print(f'Saved checkpoint for rank {i} at {checkpoint_path}')

if __name__ == '__main__':
    folder = '/home/minjunes/mistral-src/mistral-7B-v0.1'
    world_size = 8
    weights = load_split_weights(folder, world_size)
    print(len(weights))
    save_split_weights(weights, folder, world_size)
