import numpy as np
import torch
from torch import nn


def _recursive_to(xs, device):
    if torch.is_tensor(xs):
        return xs.to(device)
    elif isinstance(xs, tuple):
        return tuple(_recursive_to(x, device) for x in xs)
    elif isinstance(xs, dict):
        return {k: _recursive_to(v, device) for k, v in xs.items()}
    elif isinstance(xs, list):
        return [_recursive_to(x, device) for x in xs]
    elif type(xs) == np.ndarray:
        return xs
    else:
        raise TypeError(f"Valid types: tensor, tuple, dict, list. Got {type(xs)}")


def load_pretrained_embedding_from_file(embed_path, vocab, freeze=True):
    vocab = vocab.copy()
    num_embeddings = len(vocab)
    embed_dict = parse_embedding(embed_path)
    embed_dim = embed_dict[list(embed_dict)[0]].size(0)
    embed_tokens = nn.Embedding(num_embeddings, embed_dim)
    embed_tokens.weight.requires_grad = not freeze
    return load_embedding(embed_dict, vocab, embed_tokens)


def load_embedding(embed_dict, vocab, embedding):
    """[From fairseq]"""
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding


def parse_embedding(embed_path):
    """[From fairseq] Parse embedding text file into a dictionary of word and embedding tensors.
    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.
    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path, encoding='utf-8') as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict
