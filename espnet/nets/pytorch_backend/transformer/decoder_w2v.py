import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.decoder import Decoder as BaseDecoder


class Decoder(BaseDecoder):

    def get_input_layer(self, input_layer,
                        dropout_rate=None,
                        positional_dropout_rate=.1,
                        pos_enc_class=PositionalEncoding):

        if False:
            return self.get_input_layer_compat(
                input_layer,
                positional_dropout_rate=positional_dropout_rate,
                pos_enc_class=pos_enc_class
            )

        layer = nn.Sequential()

        if isinstance(input_layer, nn.Module):
            layer.add_module("input_layer", input_layer)

        if self.output_dim != self.attention_dim or input_layer == "linear":
            layer.add_module("linear", nn.Linear(self.output_dim, self.attention_dim))
            layer.add_module("layer_norm", nn.LayerNorm(self.attention_dim))
            layer.add_module("dropout", nn.Dropout(dropout_rate))
            layer.add_module("relu", nn.ReLU())

        layer.add_module("positional_encoding", pos_enc_class(self.attention_dim, positional_dropout_rate))

        return layer

    def get_input_layer_compat(self, input_layer,
                        dropout_rate=None,
                        positional_dropout_rate=.1,
                        pos_enc_class=PositionalEncoding):

        return nn.Sequential(
            input_layer,
            nn.Linear(self.output_dim, self.attention_dim),
            pos_enc_class(self.attention_dim, positional_dropout_rate)
        )
