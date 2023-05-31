from torch import nn
from dataloader import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class LSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_layer_size=100, output_size=1):
        """
        LSTM二分类任务
        :param input_size: 输入数据的维度
        :param hidden_layer_size:隐层的数目
        :param output_size: 输出的个数
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        input_x = input_x.view(len(input_x), 1, -1)
        hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),  # shape: (n_layers, batch, hidden_size)
                       torch.zeros(1, 1, self.hidden_layer_size))
        lstm_out, (h_n, h_c) = self.lstm(input_x, hidden_cell)
        linear_out = self.linear(lstm_out.view(len(input_x), -1))  # =self.linear(lstm_out[:, -1, :])
        predictions = self.sigmoid(linear_out)
        return predictions


