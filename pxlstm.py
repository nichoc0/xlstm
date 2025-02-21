import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ParallelxLSTMCell(nn.Module):
    def __init__(self, input_size, d, stabilizer_threshold=1.0):
        super().__init__()
        self.C_init = nn.Parameter(torch.zeros(1, d, d))  # d×d matrix
        self.n_init = nn.Parameter(torch.zeros(1, d))
        self.m_init = nn.Parameter(torch.zeros(1, 1))
        self.input_size = input_size
        self.d = d
        self.stabilizer_threshold = stabilizer_threshold
        
        # Projections
        self.W_q = nn.Linear(input_size, d)
        self.W_k = nn.Linear(input_size, d)
        self.W_v = nn.Linear(input_size, d)
        self.W_i = nn.Linear(input_size, 1)
        self.W_f = nn.Linear(input_size, 1)
        self.W_o = nn.Linear(input_size, d)
        
        self.layer_norm = nn.LayerNorm(input_size)


    def stabilize_gates(self, i_tilde, f_tilde, m_prev):
        """Stabilize gates using m_t for numerical stability."""
        # only unsqueeze once here
        m_prev_exp = m_prev.unsqueeze(-1)  # e.g. [batch, 1] -> [batch, 1, 1]
        
        # shapes: i_tilde, f_tilde => [batch, seq_len, 1]
        # m_prev_exp => [batch, 1, 1]
        m_t = torch.maximum(f_tilde + m_prev_exp, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev_exp - m_t)
        return i, f, m_t

    def store_key_value(self, key, value, i):
        """Compute key-value storage via outer product."""
        outer_product = torch.einsum('bsd,bse->bsde', value, key)
        return i.unsqueeze(-1) * outer_product

    def parallel_gate_computation(self, x_norm):
        """
        Make sure i_tilde and f_tilde end up [batch, seq_len, 1],
        rather than [batch, something, seq_len, 1].
        """
        batch_size, seq_len, _ = x_norm.shape
        
        q = self.W_q(x_norm)             # => [b, seq_len, d]
        k = self.W_k(x_norm) / (self.d**0.5)
        v = self.W_v(x_norm)
        i_tilde = self.W_i(x_norm)       # => [b, seq_len, 1] ideally
        f_tilde = self.W_f(x_norm)       # => [b, seq_len, 1] ideally
        o = torch.sigmoid(self.W_o(x_norm))

        # Force i_tilde, f_tilde to [batch, seq_len, 1]
        i_tilde = i_tilde.view(batch_size, seq_len, 1)
        f_tilde = f_tilde.view(batch_size, seq_len, 1)

        return q, k, v, i_tilde, f_tilde, o

    def parallel_memory_update(self, v, k, i, f, C_prev, seq_len):
        """
        No longer gets a 5D f. i, f should each be [batch, seq_len, 1].
        """
        # 1) Outer product updates => [batch, seq_len, d, d]
        update = self.store_key_value(k, v, i)

        # 2) Cumulative sum => [batch, seq_len, d, d]
        C_updates = torch.cumsum(update, dim=1)

        # 3) (Optional) extra squeeze check if you want
        if f.dim() > 3:
            f = f.squeeze(-1)

        # 4) [batch, seq_len, 1, 1]
        f_cum = torch.cumprod(f.squeeze(-1), dim=1).unsqueeze(-1).unsqueeze(-1)

        # 5) Broadcast old memory => [batch, seq_len, d, d]
        C_prev_expanded = C_prev.unsqueeze(1).expand(-1, seq_len, -1, -1)

        # 6) Expand => [batch, seq_len, d, d]
        f_cum_expanded = f_cum.expand_as(C_prev_expanded)

        # 7) Weighted sum of old memory + updates
        C_t = f_cum_expanded * C_prev_expanded + C_updates
        return C_t

    def parallel_forward(self, x, state):
        batch_size, seq_len, _ = x.size()
        C_prev, n_prev, m_prev = state

        x_norm = self.layer_norm(x)
        q, k, v, i_tilde, f_tilde, o = self.parallel_gate_computation(x_norm)

        # Remove the extra .expand(-1, seq_len, -1) on m_prev
        # because inside stabilize_gates() we already do m_prev_exp = m_prev.unsqueeze(1)
        i, f, m_t = self.stabilize_gates(i_tilde, f_tilde, m_prev)

        # Force i and f to be [batch, seq_len, 1]
        i = i.view(batch_size, seq_len, 1)
        f = f.view(batch_size, seq_len, 1)

        # Now proceed with memory update
        C_t = self.parallel_memory_update(v, k, i, f, C_prev, seq_len)
        
        n_prev_expanded = n_prev.unsqueeze(1).expand(-1, seq_len, -1)
        i_k_product = i * k
        f_cum = torch.cumprod(f, dim=1)
        n_t = torch.cumsum(i_k_product, dim=1) + f_cum * n_prev_expanded

        h_tilde = torch.einsum('bsde,bse->bsd', C_t, q)
        q_n_dot = torch.sum(n_t * q, dim=-1)
        denominator = torch.maximum(
            torch.abs(q_n_dot),
            torch.tensor(self.stabilizer_threshold, device=x.device)
        ).unsqueeze(-1)

        h_tilde = h_tilde / denominator
        h_t = o * h_tilde

        final_state = (C_t[:, -1], n_t[:, -1], m_t[:, -1])
        return h_t, final_state

class ParallelxLSTM(nn.Module):
    def __init__(self, input_size, d, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.d = d
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.cells = nn.ModuleList([
            ParallelxLSTMCell(input_size if i == 0 else d, d)
            for i in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)
    
    def init_state(self, batch_size, device):
        states = []
        for _ in range(self.num_layers):
            C = torch.zeros(batch_size, self.d, self.d, device=device)
            n = torch.zeros(batch_size, self.d, device=device)
            m = torch.zeros(batch_size, 1, device=device)
            states.append((C, n, m))
        return states
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device
        states = self.init_state(batch_size, device)
    

        layer_input = x
        for layer_idx, cell in enumerate(self.cells):
            layer_output, states[layer_idx] = cell.parallel_forward(layer_input, states[layer_idx])

            layer_input = self.dropout_layer(layer_output) if layer_idx < self.num_layers - 1 else layer_output
        return layer_output

class ParallelMLSTMBlock(nn.Module):
    def __init__(self, input_dim, expanded_dim, num_layers=2, dropout=0.2, expansion_factor=4, lstm_class=ParallelxLSTM):
        super().__init__()
        expanded_dim = input_dim * expansion_factor
        self.up_proj = nn.Linear(input_dim, expanded_dim)
        self.mlstm = lstm_class(expanded_dim, expanded_dim, num_layers=num_layers, dropout=dropout)
        self.down_proj = nn.Linear(expanded_dim, input_dim)

    def forward(self, x):
        x_up = self.up_proj(x)
        x_mlstm = self.mlstm(x_up)
        x_down = self.down_proj(x_mlstm)
    
        if x.shape != x_down.shape:
            print(f"Shape mismatch in residual connection: {x.shape} vs {x_down.shape}")
            x_down = x_down[:, :, :x.shape[2]]
    
        return x + x_down

class ParallelExtendedMLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=16, matrix_size=2, num_layers=2, dropout=0.2, expansion_factor=4, lstm_class=ParallelxLSTM):
        super().__init__()
        self.pre_fc_dim = hidden_size * matrix_size * matrix_size  # Now 16 * 2 * 2 = 64
        self.input_projection = nn.Linear(input_size, self.pre_fc_dim)
        self.layer_norm = nn.LayerNorm(self.pre_fc_dim)         
        self.mlstm_block = ParallelMLSTMBlock(
            input_dim=self.pre_fc_dim,
            expanded_dim=self.pre_fc_dim,
            num_layers=num_layers,
            dropout=dropout,
            expansion_factor=expansion_factor,
            lstm_class=lstm_class
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(self.pre_fc_dim, self.pre_fc_dim)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.layer_norm(x)
        x = F.gelu(self.mlstm_block(x))
        x = self.dropout(x)
        x = self.output_projection(x)
        return x