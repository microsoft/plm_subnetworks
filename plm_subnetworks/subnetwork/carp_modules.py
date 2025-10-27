import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from sequence_models.convolutional import MaskedConv1d, MaskedCausalConv1d


def sanitize_name(name: str) -> str:
    name = name.replace(".", "||")
    name = name.replace(".", "_")
    return name


def _extract_layer_idx(name: str) -> Optional[int]:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def _resolve_module(root: nn.Module, path_parts: Tuple[str, ...]) -> nn.Module:
    module = root
    for part in path_parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _is_convolution_weight(name: str) -> bool:
    return name.endswith("conv.weight")


class SubnetworkCARP(nn.Module):
    def __init__(self, carp_model: nn.Module, layers_to_mask, mask_threshold: float = 0.5):
        super().__init__()
        self.carp = carp_model
        self.mask_threshold = mask_threshold
        self.layers_to_mask = sorted(set(layers_to_mask))

        for param in self.carp.parameters():
            param.requires_grad = False

        self.mask_modules = self._collect_mask_modules()

    def _collect_mask_modules(self) -> Dict[str, nn.Module]:
        modules = {}
        for name, _ in self.carp.named_parameters():
            if not _is_convolution_weight(name):
                continue
            if ".embedder.layers." not in name:
                continue
            layer_idx = _extract_layer_idx(name)
            if layer_idx is None or layer_idx not in self.layers_to_mask:
                continue
            path = name.split(".")[:-1]
            module = _resolve_module(self.carp, tuple(path))
            key = sanitize_name(name)
            modules[key] = module
        return modules

    def _mask_conv_forward(self, module: nn.Module, mask: torch.Tensor, inverse: bool):
        if isinstance(module, MaskedConv1d):
            def masked_forward(x, input_mask=None, _module=module, _mask=mask, _inverse=inverse):
                if input_mask is not None:
                    x = x * input_mask
                weight = _module.weight * (1 - _mask) if _inverse else _module.weight * _mask
                out = F.conv1d(
                    x.transpose(1, 2),
                    weight,
                    _module.bias,
                    _module.stride,
                    _module.padding,
                    _module.dilation,
                    _module.groups,
                )
                return out.transpose(1, 2)

            return masked_forward

        if isinstance(module, MaskedCausalConv1d):
            def masked_forward(x, input_mask=None, _module=module, _mask=mask, _inverse=inverse):
                if input_mask is not None:
                    x = x * input_mask
                x = torch.transpose(x, 1, 2)
                if not _module.sequential:
                    weight = _module.conv.weight * (1 - _mask) if _inverse else _module.conv.weight * _mask
                    if _module.kernel_size == 1:
                        out = F.conv1d(
                            x,
                            weight,
                            _module.conv.bias,
                            _module.conv.stride,
                            _module.conv.padding,
                            _module.conv.dilation,
                            _module.conv.groups,
                        )
                    else:
                        out = _module._pad(x)
                        out = F.conv1d(
                            out,
                            weight,
                            _module.conv.bias,
                            _module.conv.stride,
                            _module.conv.padding,
                            _module.conv.dilation,
                            _module.conv.groups,
                        )
                        out = _module._unpad(out)
                    return out.transpose(1, 2)
                raise RuntimeError("Sequential causal masking is not supported with masking")

            return masked_forward

        if isinstance(module, nn.Conv1d):
            def masked_forward(x, _module=module, _mask=mask, _inverse=inverse):
                weight = _module.weight * (1 - _mask) if _inverse else _module.weight * _mask
                return F.conv1d(
                    x,
                    weight,
                    _module.bias,
                    _module.stride,
                    _module.padding,
                    _module.dilation,
                    _module.groups,
                )

            return masked_forward

        raise TypeError(f"Unsupported module type for masking: {type(module)}")

    def _apply_masks(self, x, masks, inverse, repr_layers, logits):
        original_forwards = {}
        patched = []
        for key, module in self.mask_modules.items():
            if key not in masks:
                continue
            original_forwards[module] = module.forward
            module.forward = self._mask_conv_forward(module, masks[key], inverse)
            patched.append(module)

        try:
            return self.carp(x, repr_layers=repr_layers, logits=logits)
        finally:
            for module in patched:
                module.forward = original_forwards[module]

    def forward(self, x, masks=None, inverse: bool = False, base: bool = False, repr_layers=None, logits: bool = True):
        if repr_layers is None:
            repr_layers = [-1]
        if base or masks is None:
            return self.carp(x, repr_layers=repr_layers, logits=logits)
        return self._apply_masks(x, masks, inverse, repr_layers, logits)


class WeightedDifferentiableMaskCARP(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        temp_init: float = 0.5,
        temp_final: float = 0.05,
        temp_decay: int = 50,
        mask_threshold: float = 0.37,
        init_value: float = 0.5,
        num_model_layers: int = 33,
        mask_top_layer_frac: float = 0.8,
        mask_layer_range: Optional[Tuple[int, int]] = None,
        temp_hold: Optional[int] = 10,
    ):
        super().__init__()

        self._config = dict(
            temp_init=temp_init,
            temp_final=temp_final,
            temp_decay=temp_decay,
            mask_threshold=mask_threshold,
            init_value=init_value,
            num_model_layers=num_model_layers,
            mask_top_layer_frac=mask_top_layer_frac,
            mask_layer_range=mask_layer_range,
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
            start_layer, end_layer = mask_layer_range
            layers_to_mask = set(range(start_layer, end_layer))
        else:
            first_layer = max(0, num_model_layers - int(mask_top_layer_frac * num_model_layers))
            layers_to_mask = set(range(first_layer, num_model_layers))
        self.layers_to_mask = sorted(layers_to_mask)

        if 0.0 < init_value < 1.0:
            p = init_value
        else:
            p = torch.sigmoid(torch.tensor(float(init_value))).item()
        score_mu = math.log(p / (1 - p))

        mask_param_dtype = torch.bfloat16

        for name, param in model.named_parameters():
            if not _is_convolution_weight(name):
                continue
            if ".embedder.layers." not in name:
                continue
            layer_idx = _extract_layer_idx(name)
            if layer_idx is None or layer_idx not in layers_to_mask:
                continue
            key = sanitize_name(name)
            mask_tensor = torch.full_like(param, score_mu, dtype=mask_param_dtype)
            self.mask_params[key] = nn.Parameter(mask_tensor).requires_grad_(True)

        self.init_mask_scores = self._compute_mask_scores()
        self.masks = self._binarize_masks(self.init_mask_scores)

    def config(self) -> dict:
        return self._config.copy()

    def compute_sparsity_loss(self):
        total_params = sum(param.numel() for param in self.mask_params.values())
        if total_params == 0:
            return torch.tensor(0.0, device=self.device)
        running_sum = torch.zeros(1, device=self.device, dtype=torch.float32)
        for param in self.mask_params.values():
            param_f32 = torch.nan_to_num(param.float(), nan=0.0, posinf=20.0, neginf=-20.0)
            running_sum.add_(param_f32.sigmoid().sum())
        return running_sum.div_(float(total_params))

    @torch.no_grad()
    def get_sparsity(self) -> float:
        mask_scores = self._compute_mask_scores()
        masks = self._binarize_masks(mask_scores)
        total_params = sum(mask.numel() for mask in masks.values())
        if total_params == 0:
            return 0.0
        zero_count = sum((mask == 0).sum(dtype=torch.int32) for mask in masks.values())
        return (zero_count / float(total_params)) * 100

    def _compute_mask_scores(self) -> Dict[str, torch.Tensor]:
        hist_dict = {}
        mask_scores = {}
        eps = 1e-7

        for name, param in self.mask_params.items():
            with torch.amp.autocast(device_type="cuda", enabled=False):
                param_fp32 = param.float()
                u = torch.rand_like(param_fp32)
                u = torch.clamp(u, eps, 1 - eps)
                noise = torch.log(u) - torch.log1p(-u)
                scores_fp32 = (param_fp32 + noise) / float(self.temperature)

            scores = torch.nan_to_num(scores_fp32.to(torch.float32), nan=0.0, posinf=50.0, neginf=-50.0)
            probs = torch.sigmoid(scores)
            mask_scores[name] = probs

            if wandb.run is not None and torch.isfinite(scores).all():
                hist_dict[f"pre_sigmoid_scores/{name}"] = wandb.Histogram(scores.detach().cpu().numpy())

        if wandb.run is not None and hist_dict:
            wandb.log(hist_dict)

        return mask_scores

    def _binarize_masks(self, mask_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hard_masks = {}
        for name, scores in mask_scores.items():
            hard = (scores > self.mask_threshold).to(dtype=torch.float32)
            hard_masks[name] = hard.detach() + scores - scores.detach()
        return hard_masks

    def forward(self) -> Dict[str, torch.Tensor]:
        scores = self._compute_mask_scores()
        return self._binarize_masks(scores)

    def scale_temp(self, epoch: int, total_epochs: int = 300):
        T0, TF, E = self.temp_init, self.temp_final, total_epochs
        frac = min(epoch / max(E - 1, 1), 1.0)
        self.temperature = TF + 0.5 * (T0 - TF) * (1 + math.cos(math.pi * frac))
