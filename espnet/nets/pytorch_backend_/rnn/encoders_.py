import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from espnet.nets.pytorch_backend_.cnn import VGG2L
from espnet.nets.pytorch_backend_.nets_utils import lengths_to_padding_mask


class RNNEncoder(nn.Module):

    """
    RNN encoder with input dropout. 
    Input shape: [batch size, timesteps, features]
    Output shape: [batch size, timesteps, hidden dim]
    Hidden shape: [num layers * num directions, batch size, hidden dim]
    """

    def __init__(self, input_dim, 
                 hidden_units,
                 output_dim=None,
                 recurrent_unit_type="lstm",
                 num_layers=1,
                 bidirectional=True, 
                 f_combine_bidir="mean", 
                 dropout=.1):

        super(RNNEncoder, self).__init__()
        
        RNN = nn.__dict__[recurrent_unit_type.upper()]

        self.rnn = RNN(
            input_dim, hidden_units, 
            num_layers=num_layers, 
            bidirectional=bidirectional
        )
        
        self.hidden_units = self.output_dim = hidden_units
        self.num_directions = int(bidirectional) + 1
        assert f_combine_bidir in ("mean", "sum")
        self.f_combine_bidir = getattr(torch.Tensor, f_combine_bidir)
        if output_dim is not None:
            self.output_dim = output_dim
            self.output_layer = nn.Linear(
                hidden_units * self.num_directions, 
                output_dim
            )
                
    def forward(self, inputs, input_lengths, prev_state=None):
        # inputs [ batch size, seq len, channels ]
        batch_size, seqlen, _ = inputs.size()

        self.rnn.flatten_parameters()

        if prev_state is not None and self.num_directions > 1:
            prev_state = reset_backward_rnn_state(prev_state)

        packed_inputs = pack_padded_sequence(
            inputs, input_lengths, batch_first=True)
        packed_outputs, hidden = self.rnn(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs)
        # outputs [ seq len, batch size, num directions * hidden size ]
        # hidden [ num layers * num directions, batch size, hidden size ]
        
        if hasattr(self, 'output_layer'):
            outputs = self.output_layer(outputs)

        elif self.num_directions > 1:
            outputs = self.f_combine_bidir(
                outputs.view(seqlen, batch_size, 2, -1),
                dim=2
            )
    
            # outputs [ seq len, batch size, hidden size ]

        outputs = torch.tanh(outputs)
        return outputs, input_lengths, hidden


class PyramidalRNNLayer(nn.Module):

    def __init__(self, input_dim, 
                 hidden_size, 
                 output_dim, 
                 subsampling, 
                 dropout=.1, 
                 recurrent_unit_type="lstm", 
                 bidirectional=True, 
                 output_activation=None):

        super(PyramidalRNNLayer, self).__init__()

        self.recurrent_unit_type = recurrent_unit_type.upper()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.subsampling = subsampling
        self.num_directions = int(self.bidirectional) + 1
        self.output_activation = torch.__dict__[output_activation] if output_activation else None

        RNN = nn.__dict__[self.recurrent_unit_type]

        self.rnn = RNN(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size * self.num_directions, output_dim)

        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, input_lengths, prev_state=None):
        batch_size = inputs.size(0)
            
        packed_inputs = pack_padded_sequence(inputs, input_lengths, batch_first=True)
        self.rnn.flatten_parameters()

        if prev_state is not None and self.num_directions > 1:
            prev_state = reset_backward_rnn_state(prev_state)

        packed_outputs, states = self.rnn(packed_inputs, hx=prev_state)
        outputs, output_lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        output_lengths = output_lengths.type_as(input_lengths)
        
        if self.subsampling > 1:
            outputs = outputs[:, ::self.subsampling]
            output_lengths = (
                ((output_lengths.float() + 1) / self.subsampling)
                .floor().type_as(input_lengths)
            )

        timesteps = outputs.size(1)
        outputs = self.fc(outputs.contiguous().view(-1, outputs.size(2)))

        if hasattr(self, "dropout"):
            outputs = self.dropout(outputs)

        if self.output_activation:
            outputs = self.output_activation(outputs)

        outputs = outputs.view(batch_size, timesteps, self.output_dim)
        return outputs, output_lengths, states


class PyramidalRNN(nn.Module):
    """RNN with projection layer module
    :param int input_dim: dimension of inputs
    :param int num_layers: number of encoder layers
    :param int hidden_units: number of rnn units (resulted in hidden_units * 2 if bidirectional)
    :param int output_dim: number of projection units
    :param np.ndarray subsampling: list of subsampling numbers
    :param float dropout: dropout rate
    :param str recurrent_unit_type: The RNN unit type
    """

    def __init__(self, input_dim, 
                 hidden_units, 
                 output_dim, 
                 num_layers, 
                 subsampling,
                 recurrent_unit_type="lstm", 
                 bidirectional=True, 
                 dropout=.1):

        super(PyramidalRNN, self).__init__()

        assert len(subsampling) == num_layers + 1

        self.layers = nn.ModuleList([
            PyramidalRNNLayer(
                input_dim=input_dim if i == 1 else output_dim,
                hidden_size=hidden_units,
                output_dim=output_dim,
                subsampling=subsampling[i],
                dropout=dropout if i < num_layers else None,
                bidirectional=bidirectional,
                recurrent_unit_type=recurrent_unit_type,
                output_activation="tanh" if i < num_layers else None
            ) for i in range(1, num_layers + 1)  
            # HACK: Ignore subsampling[0] since already done in forward
        ])

        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.subsampling = subsampling

    def forward(self, inputs, input_lengths, prev_state=None):
        """RNNP forward
        :param torch.Tensor inputs: batch of padded input sequences (B, Tmax, input_dim)
        :param torch.Tensor input_lengths: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, output_dim)
        :rtype: torch.Tensor
        """

        batch_size = inputs.size(0)

        if self.subsampling[0] > 1:
            inputs = inputs[:, ::self.subsampling[0], :]
            input_lengths = (input_lengths.float() / self.subsampling[0]).ceil().type_as(input_lengths)

        for layer in self.layers:
            inputs, input_lengths, states = layer(inputs, input_lengths)

        # initial decoder hidden is final hidden state of the forwards and backwards 
        # encoder RNNs fed through a linear layer
        if isinstance(states, tuple):
            states = states[0]

        states = torch.tanh(states.view(batch_size, 2, self.hidden_units).mean(1))
        return inputs, input_lengths, states


class Encoder(nn.Module):
    
    def __init__(self, input_dim,
                 hidden_units,
                 output_dim,
                 num_layers,
                 recurrent_unit_type="lstm", 
                 bidirectional=True, 
                 subsampling=None, 
                 frontend='vgg',
                 in_channel=1,
                 dropout=.1):

        super(Encoder, self).__init__()

        if frontend and (subsampling is not None):
            logging.warning("No subsampling when using frontend")

        self.input_dim = input_dim
        self.output_dim = output_dim

        if frontend is None:
            pass
        elif frontend.lower() == "vgg":
            self.frontend = VGG2L(in_channel)
            input_dim = VGG2L.get_output_dim(input_dim)
        else:
            raise NotImplementedError(f"{frontend}")

        options = dict(
            input_dim=input_dim,
            hidden_units=hidden_units,
            output_dim=output_dim,
            num_layers=num_layers,
            recurrent_unit_type=recurrent_unit_type,
            bidirectional=bidirectional,
            dropout=dropout
        )

        if subsampling is not None:
            options["subsampling"] = subsampling
            RNN = PyramidalRNN
        else:
            RNN = RNNEncoder

        self.rnn = RNN(**options)

    def forward(self, inputs, input_lengths, prev_state=None):
        """Encoder forward

        :param torch.Tensor inputs: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor input_lengths: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """

        if hasattr(self, 'frontend'):
            inputs, input_lengths = self.frontend(inputs, input_lengths)

        enc_out, enc_lens, hidden_states = self.rnn(inputs, input_lengths, prev_state=prev_state)
        mask, _ = lengths_to_padding_mask(enc_lens, batch_first=True)
        return enc_out.masked_fill(mask.unsqueeze(-1), 0.0), enc_lens, hidden_states


def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes - useful in processing of sliding windows over the inputs"""
    if isinstance(states, (list, tuple)):
        return [reset_backward_rnn_state(state) for state in states]
    states[1::2] = 0.
    return states


def encoder_for(input_dim, args):
    # HACK: the encoder_type should no require parsing + improve arg structure

    def parse_encoder_type(encoder_type):
        encoder_type = encoder_type.lower()
        frontend = "vgg" if encoder_type.startswith("vgg") else None
        # pyramidal = encoder_type.endswith("p")
        recurrent_unit_type = encoder_type.lstrip("vgg").rstrip("p")
        bidirectional = recurrent_unit_type.startswith("b")
        recurrent_unit_type = recurrent_unit_type.lstrip("b")
        return {
            "frontend": frontend, 
            "bidirectional": bidirectional,
            "recurrent_unit_type": recurrent_unit_type,
        }        

    return Encoder(
        input_dim=input_dim,
        hidden_units=args.eunits,
        output_dim=args.eprojs,
        num_layers=args.elayers,
        subsampling=args.subsample,
        dropout=args.dropout_rate,
        **parse_encoder_type(args.etype)
    )
