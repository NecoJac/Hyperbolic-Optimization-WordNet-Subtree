from __future__ import annotations

import torch
from torch import nn

from .poincare_ops import logmap0, mobius_add, project_to_ball


class EuclideanLogisticRegression(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def norm_tensors(self) -> list[torch.Tensor]:
        return [self.linear.weight]


class LogMapEuclideanLogisticRegression(EuclideanLogisticRegression):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(logmap0(x)).squeeze(-1)


class HyperbolicMLR(nn.Module):
    """Binary hyperbolic MLR with Poincare hyperplanes.

    Each class learns a point p_k in the Poincare ball and a tangent normal
    a_k parametrized from the origin. The returned BCE logit is the class-1
    hyperbolic score minus the class-0 hyperbolic score.
    """

    manifold_parameter_names = ("points",)

    def __init__(self, dim: int, num_classes: int = 2, c: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.c = c
        points = torch.zeros(num_classes, dim)
        try:
            import geoopt
            ball = geoopt.PoincareBall(c=c)
            self.points = geoopt.ManifoldParameter(points, manifold=ball)
        except ImportError:
            self.points = nn.Parameter(points)
        self.normals_origin = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.normal_(self.points, std=0.02)
        nn.init.normal_(self.normals_origin, std=0.02)
        with torch.no_grad():
            self.project_manifold_parameters_()

    def _class_logits(self, x: torch.Tensor) -> torch.Tensor:
        sqrt_c = self.c ** 0.5
        points = project_to_ball(self.points, c=self.c)
        lambda_p = 2.0 / (1.0 - self.c * (points * points).sum(dim=-1, keepdim=True)).clamp_min(1e-15)

        # Parallel transport from the origin to p in the Poincare ball is
        # (lambda_0 / lambda_p) a_0, with lambda_0 = 2.
        normals = (2.0 / lambda_p) * self.normals_origin
        normal_norm = normals.norm(dim=-1, keepdim=True).clamp_min(1e-15)

        translated = mobius_add(-points.unsqueeze(0), x.unsqueeze(1), c=self.c)
        translated_norm_sq = (translated * translated).sum(dim=-1).clamp(max=(1.0 - 1e-5) / self.c)
        inner = (translated * normals.unsqueeze(0)).sum(dim=-1)
        denom = (1.0 - self.c * translated_norm_sq).clamp_min(1e-15) * normal_norm.squeeze(1).unsqueeze(0)
        signed_distance_arg = 2.0 * sqrt_c * inner / denom
        scale = lambda_p.squeeze(1).unsqueeze(0) * normal_norm.squeeze(1).unsqueeze(0) / sqrt_c
        return scale * torch.asinh(signed_distance_arg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._class_logits(x)
        return logits[:, 1] - logits[:, 0]

    def project_manifold_parameters_(self) -> int:
        count = 0
        with torch.no_grad():
            projected = project_to_ball(self.points, c=self.c)
            count += int(torch.norm(projected - self.points).item() > 0)
            self.points.copy_(projected)
        return count

    def norm_tensors(self) -> list[torch.Tensor]:
        return [self.points, self.normals_origin]

    def manifold_tensors(self) -> list[torch.Tensor]:
        return [self.points]

    def trajectory_point(self) -> torch.Tensor:
        return self.points[1].detach().cpu()
