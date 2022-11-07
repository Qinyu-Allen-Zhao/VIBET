# This python file is the rewritten version of self-attention in VIBE https://github.com/mkocabas/VIBE

import torch
from torch import nn

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.1, 0.1)
        m.bias.data.fill_(0.01)


class SelfAttention(nn.Module):
    """
    The self-attention structure to improve the model's performance
    This will be used to combine the output from the GRU module
    """

    def __init__(self, attention_size,
                 num_layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        activation = nn.ReLU() if non_linearity == "relu" else nn.Tanh()

        modules = []
        for i in range(num_layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)
        self.attention.apply(init_weights)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        # representations = weighted.sum(1).squeeze()
        representations = weighted.sum(1).squeeze()
        return representations, scores

