import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize LSTM model for gait cycle prediction

        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of features in the hidden state
            num_layers (int): Number of recurrent layers
            dropout (float): Dropout rate between LSTM layers
        """
        super(LSTMModel, self).__init__()

        # 添加初始状态作为可学习参数
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
        self.c0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_size))
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 修改为输出2个值(sin和cos)
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        """
        Forward pass of the model

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Predicted gait cycle values
        """
        # Initialize hidden state and cell state
        # # copy data and use extra memory
        # h0 = self.h0.repeat(1, x.size(0), 1).to(x.device)
        # c0 = self.c0.repeat(1, x.size(0), 1).to(x.device)
        # use shared memory
        h0 = self.h0.expand(-1, x.size(0), -1).contiguous().to(x.device) # -1: same dimension as original
        c0 = self.c0.expand(-1, x.size(0), -1).contiguous().to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Use only the last time step output for prediction
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions