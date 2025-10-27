import math
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


def sanitize_name(name: str) -> str:
    return name.replace(".", "||")


def _extract_layer_index(name: str) -> Optional[int]:
    parts = name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part in {"layer", "layers", "block", "blocks"} and idx + 1 < len(parts):
            nxt = parts[idx + 1]
            if nxt.isdigit():
                return int(nxt)
    return None


def _resolve_module(root: nn.Module, path_parts: Iterable[str]) -> nn.Module:
    module: nn.Module = root
    for part in path_parts:
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def _is_attention_weight(name: str, extra_tokens: Optional[Iterable[str]] = None) -> bool:
    if not name.endswith("weight"):
        return False
    attention_tokens = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "query",
        "key",
        "value",
    )
    if extra_tokens:
        attention_tokens = (*attention_tokens, *tuple(extra_tokens))
    if any(tok in name for tok in attention_tokens):
        return True
    if "attention.output.dense.weight" in name:
        return True
    if "self_attn.out_proj.weight" in name:
        return True
    return False


class SubnetworkDayhoff(nn.Module):
    """Wrap a Dayhoff MLM model with attention-weight masking capability."""

    def __init__(
        self,
        model: nn.Module,
        layers_to_mask,
        mask_threshold: float = 0.5,
        *,
        extra_attention_tokens: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.layers_to_mask = sorted(set(layers_to_mask))
        self.mask_threshold = mask_threshold

        self.extra_attention_tokens = tuple(extra_attention_tokens or ())

        for param in self.model.parameters():
            param.requires_grad = False

        self.mask_modules = self._collect_mask_modules()

    def _collect_mask_modules(self) -> Dict[str, nn.Module]:
        modules: Dict[str, nn.Module] = {}
        for name, _ in self.model.named_parameters():
            if not _is_attention_weight(name, self.extra_attention_tokens):
                continue
            layer_idx = _extract_layer_index(name)
            if layer_idx is None or layer_idx not in self.layers_to_mask:
                continue
            module_path = name.split(".")[:-1]
            try:
                module = _resolve_module(self.model, module_path)
            except AttributeError:
                continue
            key = sanitize_name(name)
            modules[key] = module
        return modules

    def _masked_forward(self, module: nn.Module, mask: torch.Tensor, inverse: bool):
        if isinstance(module, nn.Linear):
            def masked_forward(x, _module=module, _mask=mask, _inverse=inverse):
                weight = _module.weight * (1 - _mask) if _inverse else _module.weight * _mask
                return F.linear(x, weight, _module.bias)

            return masked_forward
        raise TypeError(f"Unsupported module type for masking: {type(module)}")

    def _apply_masks(self, masks: Dict[str, torch.Tensor], inverse: bool, model_kwargs: Dict):
        patched = []
        originals = {}
        for key, module in self.mask_modules.items():
            if key not in masks:
                continue
            originals[module] = module.forward
            module.forward = self._masked_forward(module, masks[key], inverse)
            patched.append(module)

        try:
            return self.model(**model_kwargs)
        finally:
            for module in patched:
                module.forward = originals[module]

    def forward(self, masks=None, inverse: bool = False, base: bool = False, **model_kwargs):
        if base or masks is None:
            return self.model(**model_kwargs)
        return self._apply_masks(masks, inverse, model_kwargs)


class WeightedDifferentiableMaskDayhoff(nn.Module):
    """Differentiable binary masks over Dayhoff attention weights."""

    def __init__(
        self,
        model: nn.Module,
        temp_init: float = 0.5,
        temp_final: float = 0.05,
        temp_decay: int = 50,
        mask_threshold: float = 0.37,
        init_value: float = 0.5,
        num_model_layers: int = 24,
        mask_top_layer_frac: float = 1.0,
        mask_layer_range: Optional[tuple] = None,
        temp_hold: Optional[int] = 10,
        extra_attention_tokens: Optional[Sequence[str]] = None,
    ):
        super().__init__()

        self.extra_attention_tokens = tuple(extra_attention_tokens or ())

        self._config = dict(
            temp_init=temp_init,
            temp_final=temp_final,
            temp_decay=temp_decay,
            mask_threshold=mask_threshold,
            init_value=init_value,
            num_model_layers=num_model_layers,
            mask_top_layer_frac=mask_top_layer_frac,
            mask_layer_range=mask_layer_range,
            extra_attention_tokens=list(self.extra_attention_tokens),
        )

        self.temperature = temp_init
        self.temp_init = temp_init
        self.temp_final = temp_final
        self.temp_decay = temp_decay
        self.temp_hold = temp_hold
        self.mask_threshold = mask_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_params = nn.ParameterDict()

        if mask_layer_range is not None:
            start, end = mask_layer_range
            layers_to_mask = set(range(start, end))
        else:
            first = max(0, num_model_layers - int(mask_top_layer_frac * num_model_layers))
            layers_to_mask = set(range(first, num_model_layers))
        self.layers_to_mask = sorted(layers_to_mask)

        # allow passing logits directly
        if 0.0 < init_value < 1.0:
            score_mu = math.log(init_value / (1 - init_value))
        else:
            score_mu = init_value

        for name, param in model.named_parameters():
            if not _is_attention_weight(name, self.extra_attention_tokens):
                continue
            layer_idx = _extract_layer_index(name)
            if layer_idx is None or layer_idx not in self.layers_to_mask:
                continue
            key = sanitize_name(name)
            self.mask_params[key] = nn.Parameter(torch.full_like(param, score_mu)).requires_grad_(True)

        self.init_mask_scores = self._compute_mask_scores()
        self.masks = self._binarize_masks(self.init_mask_scores)

    def config(self) -> dict:
        return self._config.copy()

    def compute_sparsity_loss(self) -> torch.Tensor:
        total = sum(p.numel() for p in self.mask_params.values())
        if total == 0:
            return torch.tensor(0.0, device=self.device)
        accum = torch.zeros(1, device=self.device)
        for param in self.mask_params.values():
            accum.add_(param.sigmoid().sum())
        return accum / total

    @torch.no_grad()
    def get_sparsity(self) -> float:
        masks = self.forward()
        total = sum(mask.numel() for mask in masks.values())
        zeros = sum((mask == 0).sum(dtype=torch.int32) for mask in masks.values())
        return float(zeros) / float(total) * 100 if total > 0 else 0.0

    def _compute_mask_scores(self) -> Dict[str, torch.Tensor]:
        scores: Dict[str, torch.Tensor] = {}
        hist: Dict[str, wandb.Histogram] = {}
        eps = 1e-7
        for name, param in self.mask_params.items():
            u = torch.rand_like(param, dtype=torch.float32)
            u = torch.clamp(u, eps, 1 - eps)
            noise = torch.log(u) - torch.log1p(-u)
            logits = (param.float() + noise) / float(self.temperature)
            probs = torch.sigmoid(logits)
            scores[name] = probs.to(param.dtype)

            if getattr(wandb, "run", None) is not None:
                hist[f"pre_sigmoid_scores/{name}"] = wandb.Histogram(logits.detach().cpu().numpy())
        if getattr(wandb, "run", None) is not None and hist:
            wandb.log(hist)
        return scores

    def _binarize_masks(self, mask_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hard: Dict[str, torch.Tensor] = {}
        for name, scores in mask_scores.items():
            binary = (scores > self.mask_threshold).float()
            hard[name] = binary.detach() + scores - scores.detach()
        return hard

    def forward(self) -> Dict[str, torch.Tensor]:
        mask_scores = self._compute_mask_scores()
        return self._binarize_masks(mask_scores)

    def scale_temp(self, epoch: int, total_epochs: int = 300):
        if self.temp_hold is not None and epoch < self.temp_hold:
            return
        T0, TF = self.temp_init, self.temp_final
        frac = min(max((epoch - (self.temp_hold or 0)) / max(total_epochs - 1, 1), 0.0), 1.0)
        self.temperature = TF + 0.5 * (T0 - TF) * (1 + math.cos(math.pi * frac))


__all__ = ["SubnetworkDayhoff", "WeightedDifferentiableMaskDayhoff"]
