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

class SubnetworkProtBert(nn.Module):
    """
    Attention-only masking (no MLP). Mirrors your SubnetworkESM behavior.
    """
    def __init__(self, bert_mlm, layers_to_mask, mask_threshold=0.5):
        super().__init__()
        self.bert_mlm = bert_mlm                 # BertForMaskedLM
        self.bert = bert_mlm.bert                # BertModel
        self.cls = bert_mlm.cls                  # MLM head (left untouched)
        self.layers_to_mask = sorted(set(layers_to_mask))
        self.mask_threshold = mask_threshold

        print("Masked layers (BERT, attention-only):", self.layers_to_mask)

        # Freeze backbone + head (paper setup)
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.cls.parameters():
            p.requires_grad = False

    def _apply_masks(self, input_ids, attention_mask, masks, inverse: bool):
        original_forwards = {}

        def wrap_linear_with_mask(mod, key):
            orig_fwd = mod.forward
            def masked_forward(x, module=mod, mkey=key):
                w = module.weight
                m = masks[mkey]
                w_masked = w * (1 - m) if inverse else w * m
                return F.linear(x, w_masked, module.bias)
            return orig_fwd, masked_forward

        # Install masked forwards on attention projections only
        for layer_idx in self.layers_to_mask:
            layer = self.bert.encoder.layer[layer_idx]

            for proj_name in ["query", "key", "value"]:
                mod = getattr(layer.attention.self, proj_name)
                key = f"bert||encoder||layer||{layer_idx}||attention||self||{proj_name}||weight"
                orig, masked = wrap_linear_with_mask(mod, key)
                original_forwards[(layer_idx, f"self.{proj_name}")] = (mod, orig)
                mod.forward = masked

            mod = layer.attention.output.dense
            key = f"bert||encoder||layer||{layer_idx}||attention||output||dense||weight"
            orig, masked = wrap_linear_with_mask(mod, key)
            original_forwards[(layer_idx, "attn.out")] = (mod, orig)
            mod.forward = masked

        # Forward through MLM head as usual
        out = self.bert_mlm(input_ids=input_ids, attention_mask=attention_mask)

        # Restore originals
        for (_, _tag), (mod, orig) in original_forwards.items():
            mod.forward = orig

        return out

    def forward(self, input_ids, attention_mask=None, masks=None, inverse=False, base=False):
        if base or masks is None:
            return self.bert_mlm(input_ids=input_ids, attention_mask=attention_mask)
        return self._apply_masks(input_ids, attention_mask, masks, inverse)
    
    
class WeightedDifferentiableMaskProtBert(nn.Module):
    """
    Collects ONLY attention weight matrices in the selected (top-half) layers:
      - attention.self.{query,key,value}.weight
      - attention.output.dense.weight
    Leaves embeddings, LM head, LayerNorms, and ALL biases untouched.
    """
    def __init__(
        self,
        model: nn.Module,                # BertForMaskedLM
        temp_init: float = 0.5,
        temp_final: float = 0.05,
        temp_decay: int = 50,
        mask_threshold: float = 0.37,
        init_value: float = 0.45,        # ~paper init prob
        num_model_layers: int = 30,
        mask_top_layer_frac: float = 0.5, # top 50%
        mask_layer_range: Optional[tuple] = None,
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

        # layer selection
        if mask_layer_range is not None:
            start, end = mask_layer_range  # [start, end)
            layers_to_mask = set(range(start, end))
        else:
            first = num_model_layers - int(mask_top_layer_frac * num_model_layers)
            layers_to_mask = set(range(first, num_model_layers))
        self.layers_to_mask = sorted(layers_to_mask)

        # init scores: sigmoid(score_mu) ~= init_value
        p = init_value
        score_mu = math.log(p / (1 - p))

        wanted_suffixes = [
            "attention.self.query.weight",
            "attention.self.key.weight",
            "attention.self.value.weight",
            "attention.output.dense.weight",
        ]

        # collect only attention weights in selected layers
        for name, param in model.named_parameters():
            # expect: bert.encoder.layer.{idx}.attention.self.query.weight
            if not name.startswith("bert.encoder.layer."):
                continue
            parts = name.split(".")
            if len(parts) < 8:  # safety
                continue
            try:
                layer_idx = int(parts[3])
            except Exception:
                continue
            if layer_idx not in layers_to_mask:
                continue
            if any(name.endswith(sfx) for sfx in wanted_suffixes):
                key = sanitize_name(name)
                self.mask_params[key] = nn.Parameter(torch.full_like(param, score_mu)).requires_grad_(True)

        # precompute initial (optional)
        self.init_mask_scores = self._compute_mask_scores()
        self.masks = self._binarize_masks(self.init_mask_scores)

    def config(self) -> dict:
        return self._config.copy()
    
    def compute_sparsity_loss(self):
        total = sum(p.numel() for p in self.mask_params.values())
        if total == 0:
            return torch.tensor(0.0, device=self.device)
        s = torch.zeros(1, device=self.device)
        for p in self.mask_params.values():
            s.add_(p.sigmoid().sum())
        return s / total

    @torch.no_grad()
    def get_sparsity(self) -> float:
        masks = self.forward()
        total = sum(m.numel() for m in masks.values())
        zeros = sum((m == 0).sum(dtype=torch.int32) for m in masks.values())
        return (zeros / total) * 100 if total > 0 else 0.0

    def _compute_mask_scores(self) -> Dict[str, torch.Tensor]:
        out, hist = {}, {}
        eps = 1e-7
        for name, param in self.mask_params.items():

            # u = torch.clamp(torch.rand_like(param), eps, 1 - eps)
            # noise = torch.log(u) - torch.log(1 - u)  # logistic noise (Binary Concrete)
            # scores = (param + noise) / self.temperature
            # probs = torch.sigmoid(scores)

            u = torch.rand_like(param, dtype=torch.float32)
            u = torch.clamp(u, eps, 1 - eps)
            noise = torch.log(u) - torch.log1p(-u)
            scores = (param.float() + noise) / float(self.temperature)
            scores = scores.to(param.dtype)
            probs = torch.sigmoid(scores)

            out[name] = probs

            if getattr(wandb, "run", None) is not None:
                hist[f"pre_sigmoid_scores/{name}"] = wandb.Histogram(scores.detach().cpu().numpy())
        if getattr(wandb, "run", None) is not None and hist:
            wandb.log(hist)
        return out

    def _binarize_masks(self, mask_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hard = {}
        for name, scores in mask_scores.items():
            h = (scores > self.mask_threshold).float()
            hard[name] = h.detach() + scores - scores.detach()   # STE
        return hard

    def forward(self) -> Dict[str, torch.Tensor]:
        scores = self._compute_mask_scores()
        return self._binarize_masks(scores)

    def scale_temp(self, epoch, total_epochs=300):
        T0, TF, E = self.temp_init, self.temp_final, total_epochs
        frac = min(epoch / max(E - 1, 1), 1.0)
        self.temperature = TF + 0.5 * (T0 - TF) * (1 + math.cos(math.pi * frac))


def main():
    import argparse
    from transformers import BertTokenizer, BertForMaskedLM

    parser = argparse.ArgumentParser(description="Test SubnetworkProtBert + WeightedDifferentiableMaskProtBert")
    parser.add_argument("--model_name", type=str, default="Rostlab/prot_bert_bfd")
    parser.add_argument("--mask_top_layer_frac", type=float, default=0.5, help="fraction of top layers to mask")
    parser.add_argument("--mask_threshold", type=float, default=0.37)
    parser.add_argument("--temp_init", type=float, default=0.5)
    parser.add_argument("--temp_final", type=float, default=0.05)
    parser.add_argument("--temp_decay", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inverse", action="store_true", help="use inverse masking (mask zeros instead of ones)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1) Load tokenizer + model
    tokenizer = BertTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    bert_mlm = BertForMaskedLM.from_pretrained(args.model_name).to(device)
    bert_mlm.eval()

    # 2) Build a tiny test batch with a KNOWN masked token so CE is meaningful
    #    Ground-truth (gt) sequence vs masked (test) sequence.
    #    ProtBert expects SPACE-SEPARATED amino acids.
    sequence_gt = "M G P P R W W K G I T G L A A V V H R A D P E D K A D L Y A K M G L Y L E Y H P E T R I V E A R I K P R L H D V C E S K V S E G G L E P P C P"
    sequence_test = "M G [MASK] P R W W K G I T G L [MASK] A V V H R A D [MASK] E D K A D L Y A K M G L Y L E Y H P E T R I V [MASK] A R I K P R L H D V C E S K V S E [MASK] G L E P P C P"   # toy example


    enc_gt   = tokenizer(sequence_gt, return_tensors="pt")
    enc_test = tokenizer(sequence_test, return_tensors="pt")

    input_ids      = enc_test["input_ids"].to(device)           # contains [MASK]
    attention_mask = enc_test["attention_mask"].to(device)

    # Labels: only score the [MASK] position(s); elsewhere -100.
    labels = enc_gt["input_ids"].clone().to(device)             # true token IDs
    mask_token_id = tokenizer.mask_token_id
    labels[input_ids != mask_token_id] = -100                   # CE only at masked positions

    # 3) Baseline forward (no masking)
    with torch.no_grad():
        base_out = bert_mlm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        base_loss = base_out.loss.item()

    # 4) Build differentiable masks over top-half BERT layers (attention-only weights)
    num_layers = bert_mlm.config.num_hidden_layers
    mask_module = WeightedDifferentiableMaskProtBert(
        model=bert_mlm,
        temp_init=args.temp_init,
        temp_final=args.temp_final,
        temp_decay=args.temp_decay,
        mask_threshold=args.mask_threshold,
        num_model_layers=num_layers,
        mask_top_layer_frac=args.mask_top_layer_frac,
    ).to(device)

    print("Initial sparsity:", mask_module.get_sparsity(), "%")

    # Produce a single set of (hard STE) masks
    with torch.no_grad():
        masks = mask_module.forward()


    print("Initial sparsity:", mask_module.get_sparsity(), "%")

    # 5) Wrap with SubnetworkProtBert to apply attention-only masks during forward
    layers_to_mask = mask_module.layers_to_mask
    subnet = SubnetworkProtBert(
        bert_mlm=bert_mlm,
        layers_to_mask=layers_to_mask,
        mask_threshold=args.mask_threshold
    ).to(device)
    subnet.eval()

    # 6) Masked forward
    with torch.no_grad():
        masked_out = subnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            masks=masks,
            inverse=args.inverse,     # if set, keeps zeros and drops ones
            base=False
        )
        # Compute CE on masked tokens
        ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss_masked = ce(masked_out.logits.view(-1, masked_out.logits.size(-1)),
                         labels.view(-1)).item()

    # 7) Report
    print(f"ProtBert model: {args.model_name}")
    print(f"Num layers: {num_layers}, masking layers: {layers_to_mask}")
    print(f"Baseline CE (no subnet masks): {base_loss:.6f}")
    print(f"Masked   CE (attention-only subnet): {loss_masked:.6f}")
    print(f"Inverse masking: {args.inverse}")
    print("Done.")


if __name__ == "__main__":
    main()