import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def selective_scan_pytorch(u, delta, A, B, C, D=None):
    """
    Pure PyTorch implementation of the selective scan operation.
    
    Args:
        u: Tensor of shape (B, D, L)
        delta: Tensor of shape (B, D, L)
        A: Tensor of shape (D, N)
        B: Tensor of shape (B, N, L)
        C: Tensor of shape (B, N, L)
        D: Tensor of shape (D,) or None
        
    Returns:
        y: Tensor of shape (B, D, L)
    """
    B_size, D_size, L_size = u.shape
    N_size = A.shape[1]
    
    # Discretization
    # dA = exp(delta * A)
    # delta.unsqueeze(-1): (B, D, L, 1)
    # A.unsqueeze(0).unsqueeze(2): (1, D, 1, N)
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)) # (B, D, L, N)
    
    # dB = delta * B
    # B.transpose(1, 2).unsqueeze(1): (B, 1, L, N)
    dB = delta.unsqueeze(-1) * B.transpose(1, 2).unsqueeze(1) # (B, D, L, N)
    
    # dB_u = dB * u
    dB_u = dB * u.unsqueeze(-1) # (B, D, L, N)
    
    # Initialize hidden state h: (B, D, N)
    h = torch.zeros(B_size, D_size, N_size, device=u.device, dtype=u.dtype)
    
    ys = []
    C_trans = C.transpose(1, 2) # (B, L, N)
    
    # Sequential scan loop
    for t in range(L_size):
        h = dA[:, :, t, :] * h + dB_u[:, :, t, :]
        # y_t = sum_n (h_t * C_t)
        # C_trans[:, t, :].unsqueeze(1): (B, 1, N)
        # Multiplying with h (B, D, N) and summing over state dim N (dim=-1)
        y_t = torch.sum(h * C_trans[:, t, :].unsqueeze(1), dim=-1) # (B, D)
        ys.append(y_t)
        
    y = torch.stack(ys, dim=-1) # (B, D, L)
    
    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)
        
    return y


class MambaBlock(nn.Module):
    """
    Standard Mamba-1 layer with bidirectional support.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        bimamba_type: str = "none", # "none", "v1"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.bimamba_type = bimamba_type
        
        # Input projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # 1D Convolution along the sequence length
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # Activation function
        self.act = nn.SiLU()
        
        # Projections for selective scan parameters (dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt projection weight
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
            
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # S4D A parameter initialization (Forward)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # D skip parameter (Forward)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        # Bidirectional parameters (Backward)
        if self.bimamba_type == "v1":
            A_b = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_b_log = nn.Parameter(torch.log(A_b))
            self.A_b_log._no_weight_decay = True
            
            self.D_b = nn.Parameter(torch.ones(self.d_inner))
            self.D_b._no_weight_decay = True
            
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        
    def forward(self, x):
        # x shape: (B, L, D)
        B, L, _ = x.shape
        
        # 1. Input projection and split
        xz = self.in_proj(x) # (B, L, 2 * d_inner)
        x_proj, z = torch.split(xz, [self.d_inner, self.d_inner], dim=-1) # (B, L, d_inner) each
        
        # 2. 1D Convolution
        # nn.Conv1d expects shape (B, C, L)
        x_conv = x_proj.transpose(1, 2) # (B, d_inner, L)
        # Conv padding was kernel_size - 1, we slice to keep length L
        x_conv = self.conv1d(x_conv)[:, :, :L] # (B, d_inner, L)
        x_conv = self.act(x_conv)
        
        # Transpose back to (B, L, d_inner) for parameter projection
        x_ssm = x_conv.transpose(1, 2)
        
        # 3. Project selective scan parameters (dt, B, C)
        x_db = self.x_proj(x_ssm) # (B, L, dt_rank + 2 * d_state)
        dt, B_ssm, C_ssm = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # dt: (B, L, dt_rank) -> (B, L, d_inner)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt) # (B, L, d_inner)
        
        # Transpose to shape expected by selective_scan
        u = x_conv # (B, d_inner, L)
        delta = dt.transpose(1, 2) # (B, d_inner, L)
        B_ssm = B_ssm.transpose(1, 2) # (B, d_state, L)
        C_ssm = C_ssm.transpose(1, 2) # (B, d_state, L)
        
        # Get parameter A (Forward)
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        # Run forward selective scan
        y_f = selective_scan_pytorch(u, delta, A, B_ssm, C_ssm, self.D)
        
        if self.bimamba_type == "v1":
            # Run backward selective scan
            u_b = torch.flip(u, dims=[-1])
            delta_b = torch.flip(delta, dims=[-1])
            B_b = torch.flip(B_ssm, dims=[-1])
            C_b = torch.flip(C_ssm, dims=[-1])
            A_b = -torch.exp(self.A_b_log.float())
            
            y_b = selective_scan_pytorch(u_b, delta_b, A_b, B_b, C_b, self.D_b)
            y_b_flipped = torch.flip(y_b, dims=[-1])
            
            # Combine forward and backward scan outputs
            y = y_f + y_b_flipped
        else:
            y = y_f
            
        # Transpose back to (B, L, d_inner)
        y = y.transpose(1, 2)
        
        # 4. Gating connection with z
        out = y * self.act(z)
        
        # 5. Output projection
        out = self.out_proj(out)
        return out
