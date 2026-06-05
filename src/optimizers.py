from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class OptimizerBundle:
    optimizer: torch.optim.Optimizer
    projected: bool = False
    name: str = ""

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, model) -> int:
        self.optimizer.step()
        if self.projected and hasattr(model, "project_manifold_parameters_"):
            return model.project_manifold_parameters_()
        return 0


def make_optimizer(model, optimizer_name: str, lr: float, weight_decay: float = 0.0) -> OptimizerBundle:
    name = optimizer_name.lower()
    if name in {"adam", "projected_adam"}:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return OptimizerBundle(opt, projected=name.startswith("projected"), name=name)
    if name in {"sgd", "projected_sgd"}:
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        return OptimizerBundle(opt, projected=name.startswith("projected"), name=name)
    if name in {"rsgd", "radam"}:
        try:
            import geoopt
        except ImportError as exc:
            raise RuntimeError("Geoopt is required for rsgd/radam. Install requirements.txt first.") from exc
        cls = geoopt.optim.RiemannianSGD if name == "rsgd" else geoopt.optim.RiemannianAdam
        opt = cls(model.parameters(), lr=lr, weight_decay=weight_decay)
        return OptimizerBundle(opt, projected=True, name=name)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
