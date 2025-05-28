import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

def sanitize_name(name):
    name = name.replace(".", "||")
    name = name.replace(".", "_")
    return name


class SubnetworkESM(nn.Module):
    def __init__(self, esm_model, layers_to_mask, mask_threshold=0.5):
        super().__init__()
        self.esm = esm_model
        self.mask_threshold = mask_threshold
        self.layers_to_mask = layers_to_mask

        for param in self.esm.parameters():
            param.requires_grad = False

    def _verify_grad_flow(self, masks):
        for name, mask in masks.items():
            if mask.grad is not None:
                print(f"Mask {name} grad norm: {mask.grad.norm()}")
            else:
                print(f"Warning: No grad for mask {name}")

    def _apply_masks(self, x, masks, inverse, return_contacts, repr_layers=[], need_head_weights=False):
        original_forwards = {}
        for layer_idx in self.layers_to_mask:
            layer = self.esm.layers[layer_idx]
            for proj_name in ['k_proj', 'v_proj', 'q_proj', 'out_proj']:
                proj_module = getattr(layer.self_attn, proj_name)
                key = f'layers||{layer_idx}||self_attn||{proj_name}||weight'
                original_forwards[(layer_idx, proj_name)] = proj_module.forward

                def masked_forward(input, module=proj_module, key=key):
                    masked_weight = module.weight * (1 - masks[key]) if inverse else module.weight * masks[key]
                    return F.linear(input, masked_weight, module.bias)

                proj_module.forward = masked_forward

        output = self.esm(
            x,
            return_contacts=return_contacts,
            repr_layers=repr_layers,
            need_head_weights=need_head_weights,
        )

        for layer_idx in self.layers_to_mask:
            layer = self.esm.layers[layer_idx]
            for proj_name in ['k_proj', 'v_proj', 'q_proj', 'out_proj']:
                proj_module = getattr(layer.self_attn, proj_name)
                proj_module.forward = original_forwards[(layer_idx, proj_name)]

        return output

    def forward(self, x, masks, inverse=False, return_contacts=True, base=False, repr_layers=[], need_head_weights=False):
        if base:
            return self.esm(
                x,
                return_contacts=return_contacts,
                repr_layers=repr_layers,
                need_head_weights=need_head_weights,
            )
        return self._apply_masks(
            x,
            masks,
            inverse=inverse,
            return_contacts=return_contacts,
            repr_layers=repr_layers,
            need_head_weights=need_head_weights,
        )


class WeightedDifferentiableMask(nn.Module):
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
        mask_layer_range: tuple = None,
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
            first_layer_to_mask = num_model_layers - int(mask_top_layer_frac * num_model_layers)
            layers_to_mask = set(range(first_layer_to_mask - 1, num_model_layers))

        self.layers_to_mask = layers_to_mask

        p = init_value
        score_mu = math.log(p / (1 - p))
        param0 = score_mu

        for name, param in model.named_parameters():
            if name.startswith('layers'):
                layer_num = int(name.split('.')[1])
                if layer_num in layers_to_mask:
                    if any(proj in name for proj in ["k_proj", "v_proj", "q_proj", "out_proj"]):
                        self.mask_params[sanitize_name(name)] = nn.Parameter(torch.full_like(param, param0)).requires_grad_(True)

        self.init_mask_scores = self._compute_mask_scores()
        self.masks = self._binarize_masks(self.init_mask_scores)

    def config(self) -> dict:
        return self._config.copy()

    def compute_sparsity_loss(self):
        total_params = sum(param.numel() for param in self.mask_params.values())
        running_sum = torch.zeros(1, device=self.device)
        for param in self.mask_params.values():
            running_sum.add_(param.sigmoid().sum())
        return running_sum.div_(total_params)

    @torch.no_grad()
    def get_sparsity(self) -> float:
        mask_scores = self._compute_mask_scores()
        masks = self._binarize_masks(mask_scores)
        total_params = sum(mask.numel() for mask in masks.values())
        zero_count = sum((mask == 0).sum(dtype=torch.int32) for mask in masks.values())
        return (zero_count / total_params) * 100

    def _compute_mask_scores(self) -> Dict[str, torch.Tensor]:
        hist_dict = {}
        mask_scores = {}
        for name, param in self.mask_params.items():
            eps = 1e-7
            u = torch.clamp(torch.rand_like(param), eps, 1 - eps)
            noise = torch.log(u) - torch.log(1 - u)
            scores = (param + noise) / self.temperature
            probs = torch.sigmoid(scores)
            mask_scores[name] = probs

            if wandb.run is not None:
                hist_dict[f"pre_sigmoid_scores/{name}"] = wandb.Histogram(scores.detach().cpu().numpy())

        if wandb.run is not None and hist_dict:
            wandb.log(hist_dict)

        return mask_scores

    def _binarize_masks(self, mask_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        binary_masks = {}
        for name, scores in mask_scores.items():
            hard_threshold = (scores > self.mask_threshold).float()
            binary_masks[name] = hard_threshold.detach() + scores - scores.detach()
        return binary_masks

    def forward(self) -> Dict[str, torch.Tensor]:
        mask_scores = self._compute_mask_scores()
        return self._binarize_masks(mask_scores)

    def scale_temp(self, epoch):
        T0, TF, E = self.temp_init, self.temp_final, 300
        frac = min(epoch / (E - 1), 1.0)
        self.temperature = TF + 0.5 * (T0 - TF) * (1 + math.cos(math.pi * frac))

    def debug_binarize_masks(self, mask_scores):
        for name, scores in mask_scores.items():
            print(f"\nScores for {name}:")
            print(f"scores grad_fn: {scores.grad_fn}")
            print(f"scores requires_grad: {scores.requires_grad}")
            hard_threshold = (scores > 0.5).float()
            binary_mask = hard_threshold.detach() + scores - scores.detach()
            print(f"binary_mask grad_fn: {binary_mask.grad_fn}")
            print(f"binary_mask requires_grad: {binary_mask.requires_grad}")
            break

    def debug_compute_masks(self):
        for name, param in self.mask_params.items():
            print(f"\nMask computation for {name}:")
            print(f"param grad_fn: {param.grad_fn}")
            print(f"param requires_grad: {param.requires_grad}")
            noise = -torch.log(-torch.log(torch.rand_like(param)))
            scores = (param + noise.detach()) / self.temperature
            print(f"scores grad_fn: {scores.grad_fn}")
            break
