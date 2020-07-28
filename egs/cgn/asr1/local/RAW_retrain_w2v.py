# encoding=utf-8
# Project: learn-pytorch
# Author: xingjunjie    github: @gavinxing
# Create Time: 29/07/2017 11:58 AM on PyCharm
# Basic template from http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F


class CBOW(nn.Module):

    CONTEXT_SIZE = 10
    EMBEDDING_SIZE = 100

    def __init__(self, embedding_size=EMBEDDING_SIZE, 
                 vocab_size=None, 
                 pretrained_embeddings=None):

        super(CBOW, self).__init__()
        assert vocab_size or pretrained_embeddings, "Vocabulary size was not given"

        if pretrained_embeddings is not None:
            if isinstance(pretrained_embeddings, str):
                _, pretrained_embeddings = self.load_embeddings(pretrained_embeddings)
                vocab_size, embedding_size = pretrained_embeddings.size()
            self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_size)

        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.output_layer(embeds)
        out = F.log_softmax(out)
        return out

    @classmethod
    def parse_args(cls, parser=None):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument_group("CBOW")
        parser.add_argument("--pretrained-embeddings", type=Path)
        parser.add_argument("--context-size", type=int, default=cls.CONTEXT_SIZE)
        parser.add_argument("--embedding-size", type=int, default=cls.EMBEDDING_SIZE)
        return parser

    @staticmethod
    def load_embeddings(embedding_path):

        def parse_line(line):
            word, *vector = line.strip().split()
            return word, np.array(vector, dtype=np.float32)

        with open(embedding_path) as f:
            expected_shape = tuple(map(int, f.readline().strip().split()))
            vocab, vectors = map(list, zip(*map(parse_line, f.readlines())))
        
        vectors = torch.tensor(vectors)
        shape = tuple(vectors.size())
        assert shape == expected_shape, f"Shape mismatch: Expected {expected_shape}, got {shape}"
        return vocab, vectors


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# print(make_context_vector(data[0][0], word_to_ix))  # example


if __name__ == '__main__':
    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
    EMBEDDING_SIZE = 10
    raw_text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules
    called a program. People create programs to direct processes. In effect,
    we conjure the spirits of the computer with our spells.""".split()

    parser = argparse.ArgumentParser(description="Train Word2Vec model")
    parser.add_argument_group("Data")
    parser.add_argument("texts", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser = CBOW.parse_args(parser)
    parser, options = parser.parse_known_args()

    with open(parser.texts) as f:
        texts = list(map(str.strip, f.readlines()))

    if options.pretrained_embeddings:
        vocab, embeddings = CBOW.load_embeddings(options.pretrained_embeddings)
        net = CBOW(embeddings)
    else:
        vocab = set(sum(texts, []))
        vocab_size = len(vocab)
        net = CBOW(options.embedding_size, len(vocab))

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    data = []
    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))

    loss_func = nn.CrossEntropyLoss()
    net = CBOW(CONTEXT_SIZE, embedding_size=EMBEDDING_SIZE, vocab_size=vocab_size)
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(100):
        total_loss = 0
        for context, target in data:
            context_var = make_context_vector(context, word_to_ix)
            net.zero_grad()
            log_probs = net(context_var)

            loss = loss_func(log_probs, autograd.Variable(
                torch.LongTensor([word_to_ix[target]])
            ))

            loss.backward()
            optimizer.step()

            total_loss += loss.data
        print(total_loss)