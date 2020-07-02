import torch
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.decoder import Decoder as BaseDecoder


class Decoder(BaseDecoder):

    def get_input_layer(self, input_layer,
                        dropout_rate=None,
                        positional_dropout_rate=.1, 
                        pos_enc_class=PositionalEncoding):
        
        if self.output_dim != self.attention_dim:
            return torch.nn.Sequential(
                input_layer,
                torch.nn.Linear(self.output_dim, self.attention_dim),
                pos_enc_class(self.attention_dim, positional_dropout_rate)
            )
        else:
            return torch.nn.Sequential(
                input_layer,
                pos_enc_class(self.attention_dim, positional_dropout_rate)
            )
