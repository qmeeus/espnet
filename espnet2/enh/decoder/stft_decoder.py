from collections import OrderedDict
from typing import Tuple

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.layers.stft import Stft

from espnet2.enh.decoder.abs_decoder import AbsDecoder


class STFTDecoder(AbsDecoder):
    """ STFT decoder for speech enhancement and separation """

    def __init__(
        self,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window="hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
    ):
        super().__init__()
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

    def forward(self, input: torch.Tensor, ilens: torch.Tensor):
        """
        Args:
            input (torch.Tensor or ComplexTensor): spectrum [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        wav, wav_lens =  self.stft.inverse(input, ilens)

        return wav, wav_lens

