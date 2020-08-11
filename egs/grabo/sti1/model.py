import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers=1, bidirectional=True, dropout_rate=0.1):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_directions = 1 + bidirectional
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=output_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        if bidirectional:
            self.combine = nn.Linear(
                num_layers * self.num_directions, 1
            )

    def forward(self, inputs, input_lengths):
        batch_size = inputs.size(0)
        rnn_inputs = self.input_layer(inputs)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, input_lengths, batch_first=True)
        _, (hn, cn) = self.lstm(packed_inputs, self.init_hidden(batch_size))
        output = self.combine(hn) if self.num_directions > 1 else hn.squeeze(1)
        return output

    def init_hidden(self, batch_size):
        dim = (
            self.num_directions * batch_size, 
            self.num_layers, 
            self.output_dim, 
        )
        return torch.zeros(dim, device=self.lstm.device), torch.zeros(dim, device=self.lstm.device) 


class IntentClassifier(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(IntentClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.pre_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.output_layer = nn.Linear(input_dim, output_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, labels=None):
        pooled_output = self.pre_classifier(inputs)
        logits = self.output_layer(pooled_output)
        # outputs = (torch.sigmoid(logits),)
        outputs = (logits,)

        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
            outputs = (loss,) + outputs

        return outputs


class HierarchicalIntentClassifier(nn.Module):

    def __init__(self, input_dim, n_actions, n_instructions, dropout_rate=0.2):
        super(IntentClassifier, self).__init__()
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.n_instructions = n_instructions

        self.pre_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.action_layer = nn.Linear(input_dim, n_actions)
        self.instruction_layer = nn.Linear(input_dim + n_actions, n_instructions)
        # self.init_weights()        

    def forward(self, inputs, actions=None, instructions=None):
        pooled_output = self.pre_classifier(inputs)
        action_logits = self.action_layer(pooled_output)
        instruction_logits = self.instruction_layer(torch.cat([pooled_output, action_logits], dim=-1))
        outputs = (action_logits, instruction_logits)
        
        if actions is not None:
            action_loss = nn.CrossEntropyLoss()(action_logits.view(-1, self.n_actions), actions.view(-1))
            instruction_loss = 0

            if instructions is not None:
                instruction_loss = nn.MSELoss()(
                    instruction_logits.view(-1, self.n_instructions), 
                    instructions.view(-1, self.n_instructions).float()
                )

            total_loss = action_loss + instruction_loss
            
            outputs = (total_loss, action_loss, instruction_loss) + outputs

        return outputs

