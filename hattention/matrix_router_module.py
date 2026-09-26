"""Per-head routing over matrix-valued LLA-GDN buckets."""
import math

import torch
from torch import nn

from hattention.softmax_matrix_gdn import softmax_matrix_gdn


class MatrixMemoryRouter(nn.Module):
    def __init__(
        self, hidden_size, heads, key_dim, value_dim,
        key_mode="single_probe", norm_floor=1e-6, vector_eps=1e-6,
        initializer_range=0.006,
    ):
        super().__init__()
        if key_mode not in {"single_probe", "bilinear_frobenius"}:
            raise ValueError(f"Unknown matrix_router_key: {key_mode!r}")
        if not math.isfinite(norm_floor) or norm_floor <= 0:
            raise ValueError("matrix_router_norm_floor must be finite and positive")
        if not math.isfinite(vector_eps) or vector_eps <= 0:
            raise ValueError("matrix_router_vector_eps must be finite and positive")
        self.heads, self.key_dim, self.value_dim = heads, key_dim, value_dim
        self.key_mode = key_mode
        self.norm_floor, self.vector_eps = norm_floor, vector_eps
        self.initializer_range = initializer_range
        if key_mode == "single_probe":
            # Keep parameter names and shapes for existing checkpoints.
            self.probe = nn.Parameter(torch.empty(heads, key_dim))
        else:
            self.probe_projection = nn.Linear(hidden_size, heads * key_dim, bias=False)
        self.query = nn.Linear(hidden_size, heads * value_dim, bias=False)
        self.log_temperature = nn.Parameter(torch.empty(heads))
        self.reset_parameters()

    def reset_parameters(self):
        if self.key_mode == "single_probe":
            nn.init.normal_(self.probe, std=self.key_dim ** -.5)
            temperature = self.value_dim ** .5
        else:
            nn.init.normal_(self.query.weight, std=self.initializer_range)
            nn.init.normal_(self.probe_projection.weight, std=self.initializer_range)
            temperature = (self.key_dim * self.value_dim) ** .5
        nn.init.constant_(self.log_temperature, math.log(temperature))

    def forward(self, hidden_states, q, k, v, g, beta):
        # q is the existing GDN value-read query r_t, independent of both
        # routing projections below.
        shape = (*hidden_states.shape[:2], self.heads)
        query = self.query(hidden_states).reshape(*shape, self.value_dim)
        if self.key_mode == "single_probe":
            return softmax_matrix_gdn(q, k, v, g, beta, query, self.probe, self.log_temperature)

        from hattention.bilinear_matrix_gdn import bilinear_matrix_gdn

        probe = self.probe_projection(hidden_states).reshape(*shape, self.key_dim)
        # The operator normalizes both routing vectors in FP32.
        return bilinear_matrix_gdn(
            q, k, v, g, beta, query, probe, self.log_temperature,
            self.norm_floor, self.vector_eps,
        )
