from __future__ import annotations

import torch


EPS = 1e-5


def artanh(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=-1 + EPS, max=1 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def project_to_ball(x: torch.Tensor, c: float = 1.0, eps: float = EPS) -> torch.Tensor:
    max_norm = (1.0 - eps) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def count_outside_ball(x: torch.Tensor, c: float = 1.0, eps: float = EPS) -> int:
    max_norm = (1.0 - eps) / (c ** 0.5)
    return int((x.norm(dim=-1) > max_norm).sum().item())


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return project_to_ball(num / den.clamp_min(1e-15), c=c)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    sqrt_c = c ** 0.5
    diff = mobius_add(-x, y, c=c)
    norm = diff.norm(dim=-1).clamp_min(1e-15)
    return 2.0 / sqrt_c * artanh(sqrt_c * norm)


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    sqrt_c = c ** 0.5
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return project_to_ball(torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm), c=c)


def logmap0(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    sqrt_c = c ** 0.5
    x = project_to_ball(x, c=c)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    return artanh(sqrt_c * norm) * x / (sqrt_c * norm)


def lambda_x(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 / (1.0 - c * x2).clamp_min(1e-15)


def egrad_to_rgrad(x: torch.Tensor, grad: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    return grad / (lambda_x(x, c=c) ** 2)
