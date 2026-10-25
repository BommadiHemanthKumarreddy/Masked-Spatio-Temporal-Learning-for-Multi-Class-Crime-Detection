import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, d_model=768, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, T, D = x.shape

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = x_conv.transpose(1, 2)
        x_act = F.silu(x_conv)

        ssm_params = self.x_proj(x_act)
        B_ssm, C_ssm, dt = torch.split(ssm_params, [self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log)

        y_list = []
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)

        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1)
            A_bar = torch.exp(dt_t * A)
            B_t = B_ssm[:, t, :].unsqueeze(1)
            B_bar = dt_t * B_t

            x_t = x_act[:, t, :].unsqueeze(-1)
            h = A_bar * h + B_bar * x_t

            C_t = C_ssm[:, t, :].unsqueeze(-1)
            y_t = torch.matmul(h, C_t).squeeze(-1)
            y_t = y_t + self.D * x_act[:, t, :]
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        y = y * F.silu(z)
        out = self.out_proj(y) + residual
        return out

class MambaTemporalModel(nn.Module):
    def __init__(self, d_model=768, num_layers=2, d_state=16):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x
