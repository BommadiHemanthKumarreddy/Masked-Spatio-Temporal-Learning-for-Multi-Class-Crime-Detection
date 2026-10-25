import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, d_model=768, hidden_dim=256):
        super().__init__()
        self.w_h = nn.Linear(d_model, hidden_dim)
        self.w_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        score = self.w_a(torch.tanh(self.w_h(x))) # [B, T, 1]
        alpha = F.softmax(score, dim=1) # [B, T, 1]
        weighted_summary = torch.sum(alpha * x, dim=1) # [B, D]
        return weighted_summary, alpha.squeeze(-1)
