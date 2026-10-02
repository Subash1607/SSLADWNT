# Adaptive Neural Transformations for Anomaly Detection (NeuTraL-AD Extension)
# Research Extension: Adaptive Transformation Importance Scorer
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformationScorer(nn.Module):
    """
    Transformation Importance Scorer for NeuTraL-AD.

    Evaluates the usefulness and relevance of K learned neural transformations
    relative to the anchor representation z_0.

    Inputs:
        zs: Tensor of shape (B, K+1, z_dim) or (B, K, z_dim)
            - If (B, K+1, z_dim): zs[:, 0] is original anchor z_0, zs[:, 1:] are K transformations.
            - If z_0 is passed explicitly: z_0 of shape (B, z_dim) and z_trans of shape (B, K, z_dim).

    Outputs:
        raw_scores: Tensor of shape (B, K) - unnormalized scalar scores per transformation
        weights: Tensor of shape (B, K) - Softmax-normalized importance weights (sum to 1.0 per sample)
    """

    def __init__(self, z_dim=64, num_trans=11, h_dim=32, tau=1.0):
        super(TransformationScorer, self).__init__()
        self.z_dim = z_dim
        self.num_trans = num_trans
        self.h_dim = h_dim
        self.tau = tau

        # Feature interaction dimension: [z_0 || z_k || z_0 * z_k] -> 3 * z_dim
        in_dim = 3 * z_dim

        # Differentiable 2-layer scoring MLP with LayerNorm and GELU
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, zs, z_0=None, verbose=False):
        """
        Forward pass to compute scalar scores and normalized weights.
        """
        if z_0 is None:
            # zs contains anchor at index 0 and K transformations at indices 1..K
            assert zs.dim() == 3, f"Expected 3D tensor (B, K+1, z_dim), got shape {zs.shape}"
            z_0 = zs[:, 0]          # Shape: (B, z_dim)
            z_trans = zs[:, 1:]     # Shape: (B, K, z_dim)
        else:
            z_trans = zs            # Shape: (B, K, z_dim)

        B, K, D = z_trans.shape
        assert D == self.z_dim, f"Dimension mismatch: expected z_dim={self.z_dim}, got {D}"

        # Expand anchor z_0 to match K transformations: (B, K, z_dim)
        z_0_expanded = z_0.unsqueeze(1).expand(-1, K, -1)

        # Bilinear element-wise interaction feature
        interaction = z_0_expanded * z_trans  # (B, K, z_dim)

        # Concatenate features along feature dimension: (B, K, 3*z_dim)
        feat = torch.cat([z_0_expanded, z_trans, interaction], dim=-1)

        # Compute raw scalar score per transformation: (B, K, 1) -> (B, K)
        raw_scores = self.net(feat).squeeze(-1)

        # Differentiable temperature-scaled Softmax normalization across the K transformations
        weights = F.softmax(raw_scores / self.tau, dim=-1)

        if verbose:
            self.print_summary(raw_scores, weights)

        return raw_scores, weights

    def format_summary(self, raw_scores, weights):
        """
        Format scores and weights for debugging and logging.
        """
        # Average across batch dimension
        mean_scores = raw_scores.mean(dim=0).detach().cpu()
        mean_weights = weights.mean(dim=0).detach().cpu()

        K = len(mean_scores)
        lines = []
        lines.append("=" * 60)
        lines.append(f"Adaptive Transformation Scoring Summary (K={K}):")
        lines.append("-" * 60)
        for i in range(K):
            lines.append(f"Transformation {i+1:2d} -> score: {mean_scores[i]:+8.4f} -> weight: {mean_weights[i]:.4f}")
        lines.append("-" * 60)
        lines.append(f"Weight Sum across K={K}: {mean_weights.sum().item():.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def print_summary(self, raw_scores, weights):
        print(self.format_summary(raw_scores, weights))
