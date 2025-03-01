import torch
import torch.nn as nn
import torch.optim as optim


# Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=7):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.input_proj(src)
        output = self.transformer_encoder(src)
        output = self.output_proj(output[:, -1, :])  # 取最后一个时间步
        return output

