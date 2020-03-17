import logging
import numpy as np
from torch import nn
from torch.nn import functional as F


class VGG2L(nn.Module):

    # TODO: test

    def __init__(self, in_channels=1, **kwargs):
        super(VGG2L, self).__init__()

        # HACK (compat)

        filters = kwargs.get('filters', [64, 128])
        kernel_sizes = kwargs.get("kernel_sizes", [3, 3])
        strides = kwargs.get("strides", [1, 1])
        paddings = kwargs.get("paddings", [1, 1])

        self.layers = nn.ModuleList([
            self.conv_layer(
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

    def forward(self, xs_pad, ilens, **kwargs):
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
        xs_pad = xs_pad.view(*xs_pad.size()[:2], self.in_channel, -1).transpose(1, 2)
        xs_pad = self.layers(xs_pad)
        
        def compute_lengths(lengths, subsampling=2):
            return (lengths.float() / 2).ceil().type_as(lengths)
                
        ilens = compute_lengths(compute_lengths(ilens))

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2).contiguous().view(*xs_pad.size()[:2], -1)
        return xs_pad, ilens  # no state in this layer

    def conv_layer(self, in_channels, 
                   out_channels, 
                   kernel_size, 
                   stride=1,
                   padding=1,
                   pool_size=2, 
                   pool_stride=2):

        layer = nn.ModuleList()
        
        for _ in range(2):
            layer.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )

            layer.append(nn.ReLU())
            in_channels = out_channels

        layer.append(
            nn.MaxPool2d(
                kernel_size=pool_size, 
                stride=pool_stride, 
                ceil_mode=True
            )
        )

        return layer

    @staticmethod
    def get_output_dim(idim, in_channel=3, out_channel=128):
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