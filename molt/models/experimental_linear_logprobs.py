"""Opt-in, first-order chunked LM-head logprobs for native Qwen, TP=1."""

from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.distributed.tensor import DTensor


class _LinearLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, temperature, monitor_entropy):
        if temperature <= 0:
            raise ValueError("Logprob temperature must be positive")
        shape = hidden.shape[:-1]
        hidden = hidden.reshape(-1, hidden.shape[-1])
        targets = targets.reshape(-1)
        if hidden.shape[0] != targets.numel():
            raise ValueError("Chunked LM head requires sequence-local targets aligned with hidden states")
        result = torch.empty((hidden.shape[0], 2), device=hidden.device, dtype=torch.float32)
        for start in range(0, hidden.shape[0], 256):
            stop = start + 256
            logits = F.linear(hidden[start:stop], weight).float()
            logits.div_(temperature)
            normalizer = logits.logsumexp(-1)
            result[start:stop, 0] = logits.gather(-1, targets[start:stop, None]).squeeze(-1) - normalizer
            result[start:stop, 1] = (
                normalizer - (logits.softmax(-1) * logits).sum(-1) if monitor_entropy else 0
            )
        ctx.save_for_backward(hidden, weight, targets)
        ctx.temperature = temperature
        ctx.hidden_shape = (*shape, hidden.shape[-1])
        return result.reshape(*shape, 2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        hidden, weight, targets = ctx.saved_tensors
        upstream = grad_output.reshape(-1, 2)[:, 0]
        grad_hidden = torch.empty_like(hidden) if ctx.needs_input_grad[0] else None
        # Accumulate across token chunks in fp32, then return the parameter's dtype.
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        for start in range(0, hidden.shape[0], 256):
            stop = start + 256
            x = hidden[start:stop]
            grad_logits = F.linear(x, weight).float()
            grad_logits.div_(ctx.temperature)
            grad_logits = grad_logits.softmax(-1).neg_()
            grad_logits.scatter_add_(
                -1, targets[start:stop, None], torch.ones_like(targets[start:stop, None], dtype=grad_logits.dtype)
            )
            grad_logits.mul_(upstream[start:stop, None] / ctx.temperature)
            grad_logits = grad_logits.to(weight.dtype)
            if grad_hidden is not None:
                torch.mm(grad_logits, weight, out=grad_hidden[start:stop])
            if grad_weight is not None:
                grad_weight.add_(grad_logits.T @ x)
        return (
            grad_hidden.reshape(ctx.hidden_shape) if grad_hidden is not None else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            None, None, None,
        )


@contextmanager
def chunked_lm_head(model, targets, temperature, monitor_entropy):
    """Replace only Linear.forward; retain FSDP's module pre/post-forward hooks."""
    head = model.lm_head
    if not isinstance(head, torch.nn.Linear) or head.bias is not None:
        raise ValueError("Experimental logprobs require a bias-free torch Linear LM head")
    if getattr(model, "mtp", None) is not None:
        raise ValueError("Experimental logprobs require MTP disabled")

    def forward(module, hidden):
        # FSDP has already unsharded this parameter when Linear.forward executes.
        weight = module.weight
        weight = weight.to_local() if isinstance(weight, DTensor) else weight
        if isinstance(hidden, DTensor):
            raise ValueError("Experimental logprobs do not support tensor parallel inputs")
        if weight.shape != (model.config.text_config.vocab_size, hidden.shape[-1]):
            raise ValueError("Experimental logprobs require an unsharded full-vocabulary weight")
        return _LinearLogprobs.apply(hidden, weight, targets, temperature, monitor_entropy)

    previous = head.__dict__.get("forward")
    head.forward = MethodType(forward, head)
    try:
        yield
    finally:
        if previous is None:
            del head.forward
        else:
            head.forward = previous
