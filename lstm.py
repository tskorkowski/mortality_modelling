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
        output_sq = []
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
            output_sq.append(o_t.unsqueeze(0))

        # Reshape hidden_seq p/ retornar
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        # Reshape output_seq and store for the next layer
        output_sq = torch.cat(output_sq, dim=0)
        output_sq = output_sq.transpose(0, 1).contiguous()

        return output_sq, h_t


class LstmModel(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_size = hidden_sz
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.lstm_layers.append(LstmLayer(input_sz, hidden_sz))
            else:
                self.lstm_layers.append(LstmLayer(hidden_sz, hidden_sz))

        # bring the output to the correct level
        self.fc = nn.Linear(hidden_sz, 1)

    def forward(self, x):
        for layer in self.lstm_layers:
            x, h_t = layer(x)

        h_out = h_t.view(-1, self.hidden_size)
        return self.fc(h_out)
