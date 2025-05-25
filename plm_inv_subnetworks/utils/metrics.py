import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.

    Evaluates the cross-entropy loss at specified locations in a sequence.

    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - weight: (C, ): class weights for nn.CrossEntropyLoss
    """

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, pred, tgt, mask):
        # Make sure we have that empty last dimension

        if len(mask.shape) == len(pred.shape) - 1:
            mask = mask.unsqueeze(-1)
        # Make sure mask is boolean
        mask = mask.bool()
        # Number of locations to calculate loss
        n = mask.sum()
        # Select
        p = torch.masked_select(pred, mask).view(n, -1)
        t = torch.masked_select(tgt, mask.squeeze())
        return super().forward(p, t)

class PerSequenceMaskedCrossEntropyLoss(nn.Module):
    """
    Computes masked cross-entropy loss per sequence, then scales each
    sequence's mean so that the overall batch mean matches a single global
    mean over all masked tokens.

    Returns a tensor of shape (B,), and calling .mean() on that tensor
    reproduces the MaskedCrossEntropyLoss(reduction='mean') result.
    """
    def __init__(self, weight=None):
        super().__init__()
        # per-token loss
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, pred, tgt, mask):
        """
        Inputs:
            pred: (B, L, V)   — logits
            tgt:  (B, L)      — target token indices
            mask: (B, L) bool — which positions to include
        Returns:
            scaled_per_seq: (B,)
        """
        B, L, V = pred.shape

        # flatten everything
        pred_flat = pred.view(-1, V)           # (B*L, V)
        tgt_flat  = tgt.view(-1)               # (B*L,)
        mask_flat = mask.view(-1).bool()       # (B*L,)

        # per-token loss, zeroing out unmasked
        loss_flat   = self.ce_loss(pred_flat, tgt_flat)  # (B*L,)
        masked_loss = loss_flat * mask_flat              # (B*L,)
        masked_loss = masked_loss.view(B, L)             # (B, L)

        # compute raw per-sequence mean
        mask_counts   = mask.view(B, L).sum(dim=1).float()        # (B,)
        per_seq_loss  = masked_loss.sum(dim=1) / mask_counts.clamp(min=1)

        # scale each sequence so that mean(per_seq_scaled) = global mean
        total_masked  = mask_counts.sum()                         # scalar
        # factor = (mask_counts * B / total_masked)
        scaled = per_seq_loss * (mask_counts * B / total_masked.clamp(min=1))

        return scaled

def logits_kl(P_logits, Q_logits, masks, epsilon=1e-6):
    """
    Computes the regular KL divergence between probability distributions derived from logits,
    considering only valid residues indicated by `masks`.

    Args:
        P_logits (torch.Tensor): Predicted logits (B, N, V) - batch, sequence length, vocab size
        Q_logits (torch.Tensor): Ground truth logits (B, N, V) - batch, sequence length, vocab size
        masks (torch.Tensor): Binary mask indicating valid residues (B, N)
        epsilon (float): Small value to prevent numerical instability

    Returns:
        torch.Tensor: KL divergence per sequence (B,)
    """

    # Convert logits to probabilities using softmax
    P = F.softmax(P_logits, dim=-1)  # (B, N, V)
    Q = F.softmax(Q_logits, dim=-1)  # (B, N, V)

    # Ensure proper dtype
    P = P.to(dtype=torch.float32)
    Q = Q.to(dtype=torch.float32)
    masks = masks.to(dtype=torch.float32)

    # **Check if P and Q are normalized** (sum to 1 along last dimension)
    if not (torch.allclose(P.sum(dim=-1), torch.ones_like(P.sum(dim=-1)), atol=1e-4) and
            torch.allclose(Q.sum(dim=-1), torch.ones_like(Q.sum(dim=-1)), atol=1e-4)):
        raise ValueError("P and Q must be properly normalized probability distributions (summing to 1).")

    # Clamp values to avoid log(0) or division by zero
    P_clamped = torch.clamp(P, min=epsilon, max=1)
    Q_clamped = torch.clamp(Q, min=epsilon, max=1)


    # Compute KL divergence: sum(P * log(P / Q)) over the vocab dimension
    kl_terms = P_clamped * torch.log(P_clamped / Q_clamped)  # (B, N, V)

    # Sum over vocab dimension to get per-position KL divergence
    kl_per_position = kl_terms.sum(dim=-1)  # (B, N)

    # Apply mask to ignore padding positions
    masked_kl = kl_per_position * masks  # (B, N)

    return masked_kl

def aggregate_over_seq(masked_kl, masks, epsilon=1e-6):
    sequence_sums = masked_kl.sum(dim=1)  # (B,)
    # Normalize by the number of valid residues
    valid_residues_per_sequence = masks.sum(dim=1).clamp(min=epsilon)  # (B,)
    return sequence_sums / valid_residues_per_sequence 


