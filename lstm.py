"""
https://github.com/piEsposito/pytorch-lstm-by-hand/blob/master/LSTM.ipynb
"""

import torch
import torch.nn as nn
import math

# TODO:udpate lstm layer to return not only final hiden state and cell state
# but full list of output cells


class LstmLayer(nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz

        # Input gate parameters
        self.W_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # Forget gate parameters
        self.W_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # Cell state parameters
        self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # Output gate parameters
        self.W_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.U_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        """
        assumes x.shape represents (batch_size, sequence_size, input_size)
        """
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i + self.b_i)

            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f + self.b_f)

            g_t = torch.tanh(x_t @ self.W_c + h_t @ self.U_c + self.b_c)

            c_t = f_t * c_t + i_t * g_t

            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o + self.b_o)

            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # Reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return o_t, (h_t, c_t)


class LstmModel(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, num_layers: int) -> None:
        super().__init__()
        self.lstm_layers = nn.ModuleList(
            [LstmLayer(input_sz, hidden_sz) for _ in range(num_layers)]
        )

        # bring the output to the correct level
        self.fc = nn.Linear(hidden_sz, 1)

    def forward(self, x):
        h_t, c_t = None, None
        for layer in self.lstm_layers:
            x, (h_t, c_t) = layer(x, (h_t, c_t))

        h_out = h_t.view(-1, self.hidden_size)
        return self.fc(h_out)
