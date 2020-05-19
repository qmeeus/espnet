import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VGGLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_size, pool_stride):

        super(VGGLayer, self).__init__()

        self.conv1, self.conv2 = (
            nn.Conv2d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ) for i in range(2)
        )

        self.pool = nn.MaxPool2d(
            kernel_size=pool_size,
            stride=pool_stride,
            ceil_mode=True
        )

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        return self.pool(X)

    def output_length(self, input_length):
        raise NotImplementedError


class VGG2L(nn.Module):

    # TODO: test

    def __init__(self, in_channels=1, **options):
        super(VGG2L, self).__init__()

        # HACK (compat)

        filters = options.get('filters', [64, 128])
        kernel_sizes = options.get("kernel_sizes", [3, 3])
        strides = options.get("strides", [1, 1])
        paddings = options.get("paddings", [1, 1])

        self.layers = nn.Sequential(*[
            VGGLayer(
                in_channels=in_channels if i == 0 else filters[i-1],
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                pool_size=2,
                pool_stride=2
            ) for i in range(len(filters))
        ])

        self.in_channels = in_channels

    def forward(self, xs_pad, ilens, **options):
        """VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128 * D // 4)
        :rtype: torch.Tensor
        """
        assert torch.is_tensor(ilens)
        logging.debug(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.view(*xs_pad.size()[:2], self.in_channels, -1).transpose(1, 2)
        xs_pad = self.layers(xs_pad)

        def compute_lengths(lengths, subsampling=2):
            return (lengths.float() / 2).ceil().type_as(lengths)

        ilens = compute_lengths(compute_lengths(ilens))

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2).contiguous()
        xs_pad = xs_pad.view(*xs_pad.size()[:2], -1)
        return xs_pad, ilens  # no state in this layer

    @staticmethod
    def get_output_dim(idim, in_channel=1, out_channel=128):
        """Return the output size of the VGG frontend.
        :param in_channel: input channel size
        :param out_channel: output channel size
        :return: output size
        :rtype int
        """
        idim = idim / in_channel
        idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
        idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
        return int(idim) * out_channel  # number of channels


class DilatedCNN(nn.Module):

    @classmethod
    def default_arch(cls, input_dim):
        return cls(input_dim, [64, 128], [3, 3], [1, 1], [1, 1], [2, 2])

    def __init__(self, input_dim, filters, kernel_sizes, strides, paddings, dilations):
        super(DilatedCNN, self).__init__()

        self.layers = nn.Sequential(*[
            VGGLayer(
                in_channels=in_channels if i == 0 else filters[i-1],
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                pool_size=2,
                pool_stride=2
            ) for i in range(len(filters))
        ])

        self.in_channels = in_channels

    def forward(self, xs_pad, ilens):
        xs_pad = xs_pad.view(*xs_pad.size()[:2], self.in_channels, -1).transpose(1, 2)
        xs_pad = self.layers(xs_pad)

        def compute_lengths(lengths, subsampling=2):
            return (lengths.float() / 2).ceil().type_as(lengths)

        ilens = compute_lengths(compute_lengths(ilens))

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2).contiguous()
        xs_pad = xs_pad.view(*xs_pad.size()[:2], -1)
        return xs_pad, ilens  # no state in this layer
