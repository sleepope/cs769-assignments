from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]
                # State init
                if len(state) == 0:
                    state["step"] = 0                                    # t
                    state["exp_avg_grad"] = torch.zeros_like(p.data)     # m_0
                    state["exp_avg_grad_sq"] = torch.zeros_like(p.data)  # t_0

                state["step"] += 1

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                m = state["exp_avg_grad"]
                v = state["exp_avg_grad_sq"]
                beta1, beta2 = group["betas"]

                # BUG: why not work???
                # m = beta1 * m + (1-beta1) * grad
                # v = beta2 * v + (1-beta2) * grad**2
                m *= beta1
                m += (1 - beta1) * grad
                v *= beta2
                v += (1 - beta2) * grad**2

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if group["correct_bias"]:
                    m_hat = m / (1 - beta1 ** state["step"])
                    v_hat = v / (1 - beta2 ** state["step"])

                # Update parameters
                    p.data -= (alpha * m_hat / (v_hat**0.5 + group["eps"]))
                else:
                    p.data -= (alpha * m / (v**0.5 + group["eps"]))


                # Add weight decay after the main gradient-based updates.
                # p.data *= (1 - alpha * group["weight_decay"]) # BUG: why not work???
                p.data = p.data - p.data * alpha * group["weight_decay"]
                # Please note that the learning rate should be incorporated into this update.

        return loss
