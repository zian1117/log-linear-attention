# -*- coding: utf-8 -*-

import math

from fla.models.gated_deltanet import GatedDeltaNetConfig


class HGatedDeltaNetConfig(GatedDeltaNetConfig):

    model_type = "h_gated_deltanet"

    def __init__(
        self,
        *args,
        matrix_router=False,
        matrix_router_key="single_probe",
        matrix_router_norm_floor=1e-6,
        matrix_router_vector_eps=1e-6,
        **kwargs,
    ):
        if matrix_router_key not in {"single_probe", "bilinear_frobenius"}:
            raise ValueError(f"Unknown matrix_router_key: {matrix_router_key!r}")
        for name, value in (
            ("matrix_router_norm_floor", matrix_router_norm_floor),
            ("matrix_router_vector_eps", matrix_router_vector_eps),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        super().__init__(*args, **kwargs)
        self.matrix_router = matrix_router
        self.matrix_router_key = matrix_router_key
        self.matrix_router_norm_floor = matrix_router_norm_floor
        self.matrix_router_vector_eps = matrix_router_vector_eps
