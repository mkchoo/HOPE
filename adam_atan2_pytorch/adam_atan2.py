from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class AdamAtan2(Optimizer):
    """Adam-atan2 with the settings used by HOPE."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=(0.9, 0.99),
            a=1.27,
            b=1.0,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            a = group["a"]
            b = group["b"]

            for p in (parameter for parameter in group["params"] if parameter.grad is not None):
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["steps"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                steps = state["steps"] + 1

                bias_correct1 = 1.0 - beta1**steps
                bias_correct2 = 1.0 - beta2**steps

                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.lerp_(grad * grad, 1.0 - beta2)

                den = exp_avg_sq.mul(b * b / bias_correct2).sqrt_()
                update = exp_avg.mul(1.0 / bias_correct1).atan2_(den)

                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr * a)
                state["steps"] = steps
