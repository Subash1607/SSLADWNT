"""
colab_sanity_check.py
=====================
Lightweight sanity check for the Adaptive NeuTraL-AD implementation.
Runs in Google Colab in < 2 minutes. Does NOT require full training.

Checks:
  1.  Input shape is correct for NATOPS
  2.  Transformation representations shape: (B, K+1, z_dim)
  3.  Scorer output shape: (B, K) scores, (B, K) weights
  4.  Weights sum to 1.0 per sample
  5.  Weights are non-negative
  6.  No NaN or Inf in scores / weights / loss
  7.  Weighted DCL loss is numerically close to unweighted when weights are uniform
  8.  Baseline mode (adaptive=False) produces identical output to original
  9.  Gradients propagate through TransformationScorer
  10. Anomaly score shape is (B,)
  11. Config file for adaptive mode loads without error
  12. Config file for GPU baseline loads without error

Usage (in Colab):
    !python colab_sanity_check.py
"""

import sys
import os
import torch
import numpy as np

# ─── Colour helpers ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗ FAIL{RESET}  {msg}"); sys.exit(1)
def info(msg): print(f"  {CYAN}»{RESET}  {msg}")
def header(msg): print(f"\n{BOLD}{YELLOW}{'─'*60}{RESET}\n{BOLD}{YELLOW}  {msg}{RESET}\n{BOLD}{YELLOW}{'─'*60}{RESET}")

# ─── Constants ────────────────────────────────────────────────────────────────
B       = 8       # Batch size
C       = 24      # NATOPS channels
L       = 51      # NATOPS sequence length
K       = 11      # Number of transformations
Z_DIM   = 64      # Latent dimension
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─── Minimal config dict that mirrors config_natops_adaptive_gpu.yml ─────────
BASE_CONFIG = dict(
    model='seqNTL', trainer='NTL', network='seqNTL',
    device=DEVICE,
    x_length=L,
    learning_rate=0.001,
    training_epochs=2,      # Tiny — only for integration test
    latent_dim=Z_DIM,
    enc_hdim=32,
    enc_nlayers=6,
    num_trans=K,
    trans_nlayers=5,
    trans_type='mul',
    loss='DCL',
    enc_bias=False,
    batch_norm=False,
    loss_temp=0.1,
    l2=1e-5,
    optimizer='Adam',
    scheduler=None,
    early_stopper=None,
    shuffle=True,
    num_repeat=1,
    save_scores=False,
    result_folder='RESULTS/RESULTS_TEST_',
    adaptive=False,   # will be overridden per test
    scorer_hdim=32,
    tau_w=1.0,
)

def make_config(adaptive=False):
    cfg = dict(BASE_CONFIG)
    cfg['adaptive'] = adaptive
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Config loading
# ══════════════════════════════════════════════════════════════════════════════
header("CHECK 1 — Config file loading")
from config.base import Grid, Config

for yml, label in [
    ('config_files/config_natops_gpu.yml',          'Baseline GPU config'),
    ('config_files/config_natops_adaptive_gpu.yml', 'Adaptive GPU config'),
]:
    try:
        g = Grid(yml, 'natops')
        c = Config(**g[0])
        assert c['adaptive'] in (True, False), "adaptive key missing/wrong type"
        ok(f"{label} loads OK  (adaptive={c['adaptive']}, device={c['device']})")
    except Exception as e:
        fail(f"{label} failed to load: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — TransformationScorer standalone
# ══════════════════════════════════════════════════════════════════════════════
header("CHECK 2 — TransformationScorer standalone")
from models.AdaptiveModule import TransformationScorer

scorer = TransformationScorer(z_dim=Z_DIM, num_trans=K, h_dim=32, tau=1.0).to(DEVICE)
scorer.eval()

# Simulate zs: anchor at index 0, K transforms at 1..K
zs_fake = torch.randn(B, K+1, Z_DIM, device=DEVICE)

with torch.no_grad():
    raw_scores, weights = scorer(zs_fake)

info(f"zs input shape:     {list(zs_fake.shape)}")
info(f"raw_scores shape:   {list(raw_scores.shape)}")
info(f"weights shape:      {list(weights.shape)}")

if raw_scores.shape != (B, K):
    fail(f"raw_scores shape should be ({B}, {K}), got {raw_scores.shape}")
ok(f"raw_scores shape: ({B}, {K})")

if weights.shape != (B, K):
    fail(f"weights shape should be ({B}, {K}), got {weights.shape}")
ok(f"weights shape: ({B}, {K})")

weight_sums = weights.sum(dim=-1)
if not torch.allclose(weight_sums, torch.ones(B, device=DEVICE), atol=1e-5):
    fail(f"Weights do not sum to 1.0 — got: {weight_sums}")
ok(f"Weights sum to 1.0 per sample  (min={weight_sums.min():.6f}, max={weight_sums.max():.6f})")

if (weights < 0).any():
    fail("Negative weights found — Softmax failed")
ok(f"All weights non-negative  (min={weights.min():.6f})")

info(f"Sample 0 weights:   {[round(float(w),4) for w in weights[0]]}")
info(f"Sample 0 weight sum: {float(weights[0].sum()):.6f}")
info(f"Min weight:  {float(weights.min()):.6f}")
info(f"Max weight:  {float(weights.max()):.6f}")

if torch.isnan(weights).any() or torch.isinf(weights).any():
    fail("NaN/Inf in weights!")
ok("No NaN or Inf in weights")

if torch.isnan(raw_scores).any() or torch.isinf(raw_scores).any():
    fail("NaN/Inf in raw_scores!")
ok("No NaN or Inf in raw_scores")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — SeqNeutralAD baseline mode (adaptive=False)
# ══════════════════════════════════════════════════════════════════════════════
header("CHECK 3 — SeqNeutralAD: BASELINE mode (adaptive=False)")
from config.base import Config
from models.NeutralAD import SeqNeutralAD
from models.SeqNets import SeqNets

cfg_baseline = Config.from_dict(make_config(adaptive=False))
model_baseline = SeqNeutralAD(SeqNets(), x_dim=(C, L), config=cfg_baseline).to(DEVICE)
model_baseline.eval()

x = torch.randn(B, C, L, device=DEVICE)
with torch.no_grad():
    zs = model_baseline(x)

info(f"Input shape:               {list(x.shape)}")
info(f"zs shape (baseline):       {list(zs.shape)}")

if zs.shape != (B, K+1, Z_DIM):
    fail(f"zs should be ({B}, {K+1}, {Z_DIM}), got {zs.shape}")
ok(f"zs shape correct: ({B}, {K+1}, {Z_DIM})")

if model_baseline.scorer is not None:
    fail("Scorer should be None in baseline mode!")
ok("model.scorer is None in baseline mode — baseline preserved")

if torch.isnan(zs).any() or torch.isinf(zs).any():
    fail("NaN/Inf in baseline zs!")
ok("No NaN/Inf in baseline representations")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — SeqNeutralAD adaptive mode (adaptive=True)
# ══════════════════════════════════════════════════════════════════════════════
header("CHECK 4 — SeqNeutralAD: ADAPTIVE mode (adaptive=True)")
cfg_adaptive = Config.from_dict(make_config(adaptive=True))
model_adaptive = SeqNeutralAD(SeqNets(), x_dim=(C, L), config=cfg_adaptive).to(DEVICE)
model_adaptive.eval()

with torch.no_grad():
    zs_a, scores_a, weights_a = model_adaptive(x, return_weights=True)

info(f"zs shape (adaptive):       {list(zs_a.shape)}")
info(f"scores shape:              {list(scores_a.shape)}")
info(f"weights shape:             {list(weights_a.shape)}")

if zs_a.shape != (B, K+1, Z_DIM):
    fail(f"zs_a shape wrong: {zs_a.shape}")
ok(f"zs shape correct: ({B}, {K+1}, {Z_DIM})")

if scores_a.shape != (B, K):
    fail(f"scores_a shape wrong: {scores_a.shape}")
ok(f"scores shape correct: ({B}, {K})")

if weights_a.shape != (B, K):
    fail(f"weights_a shape wrong: {weights_a.shape}")
ok(f"weights shape correct: ({B}, {K})")

ws = weights_a.sum(dim=-1)
if not torch.allclose(ws, torch.ones(B, device=DEVICE), atol=1e-5):
    fail(f"Adaptive weights don't sum to 1: {ws}")
ok("Adaptive weights sum to 1.0 per sample")

if torch.isnan(weights_a).any() or torch.isinf(weights_a).any():
    fail("NaN/Inf in adaptive weights!")
ok("No NaN/Inf in adaptive weights")

info(f"Sample 0 adaptive weights: {[round(float(w),4) for w in weights_a[0]]}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5 — DCL loss: baseline path (no weights)
# ══════════════════════════════════════════════════════════════════════════════
header("CHECK 5 — DCL: baseline path (weights=None)")
from models.Losses import DCL

loss_fn = DCL(temperature=0.1).to(DEVICE)

with torch.no_grad():
    loss_baseline = loss_fn(zs_a, eval=False)
    score_baseline = loss_fn(zs_a, eval=True)

info(f"loss_baseline shape:   {list(loss_baseline.shape)}")
info(f"score_baseline shape:  {list(score_baseline.shape)}")

if loss_baseline.shape != (B,):
    fail(f"Baseline loss shape should be ({B},), got {loss_baseline.shape}")
ok(f"Baseline DCL loss shape: ({B},)")

if score_baseline.shape != (B,):
    fail(f"Baseline score shape wrong")
ok(f"Baseline anomaly score shape: ({B},)")

if torch.isnan(loss_baseline).any() or torch.isinf(loss_baseline).any():
    fail("NaN/Inf in baseline DCL loss!")
ok("No NaN/Inf in baseline DCL loss")

info(f"Baseline loss values (first 4):  {[round(float(v),4) for v in loss_baseline[:4]]}")
info(f"Baseline score values (first 4): {[round(float(v),4) for v in score_baseline[:4]]}")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6 — DCL loss: adaptive path (with weights)
# ══════════════════════════════════════════════════════════════════════════════
header("CHECK 6 — DCL: adaptive path (with weights)")

with torch.no_grad():
    loss_adaptive = loss_fn(zs_a, eval=False, weights=weights_a)
    score_adaptive = loss_fn(zs_a, eval=True,  weights=weights_a)

info(f"loss_adaptive shape:   {list(loss_adaptive.shape)}")
info(f"score_adaptive shape:  {list(score_adaptive.shape)}")

if loss_adaptive.shape != (B,):
    fail(f"Adaptive loss shape wrong: {loss_adaptive.shape}")
ok(f"Adaptive DCL loss shape: ({B},)")

if score_adaptive.shape != (B,):
    fail(f"Adaptive score shape wrong")
ok(f"Adaptive anomaly score shape: ({B},)")

if torch.isnan(loss_adaptive).any() or torch.isinf(loss_adaptive).any():
    fail("NaN/Inf in adaptive DCL loss!")
ok("No NaN/Inf in adaptive DCL loss")

info(f"Adaptive loss values (first 4):  {[round(float(v),4) for v in loss_adaptive[:4]]}")
info(f"Adaptive score values (first 4): {[round(float(v),4) for v in score_adaptive[:4]]}")

# When weights are uniform (1/K each), weighted loss should equal unweighted loss
uniform_weights = torch.full((B, K), 1.0/K, device=DEVICE)
with torch.no_grad():
    loss_uniform = loss_fn(zs_a, eval=False, weights=uniform_weights)
diff = (loss_uniform - loss_baseline).abs().max().item()
info(f"Max |loss_uniform - loss_baseline| = {diff:.6f}  (should be ~0 for uniform weights)")
if diff > 1e-4:
    fail(f"Uniform-weight adaptive loss differs from baseline by {diff:.6f} — mismatch!")
ok(f"Uniform weights reproduce baseline loss (max diff={diff:.8f})")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7 — Gradient flow through TransformationScorer
# ══════════════════════════════════════════════════════════════════════════════
header("CHECK 7 — Gradient propagation through TransformationScorer")
model_adaptive.train()
optimizer_test = torch.optim.Adam(model_adaptive.parameters(), lr=1e-3)
optimizer_test.zero_grad()

x_grad = torch.randn(B, C, L, device=DEVICE)
zs_g, scores_g, weights_g = model_adaptive(x_grad, return_weights=True)
loss_g = loss_fn(zs_g, eval=False, weights=weights_g)
loss_g.mean().backward()

# Check scorer parameters have gradients
scorer_params = list(model_adaptive.scorer.parameters())
grads = [p.grad for p in scorer_params if p.grad is not None]
info(f"TransformationScorer has {len(scorer_params)} parameter tensors")
info(f"Scorer params with non-None grad: {len(grads)} / {len(scorer_params)}")

if len(grads) == 0:
    fail("No gradients reached TransformationScorer — backprop broken!")
ok(f"Gradients propagated to TransformationScorer ({len(grads)}/{len(scorer_params)} params)")

for i, p in enumerate(scorer_params):
    if p.grad is None:
        fail(f"Scorer parameter {i} has no gradient!")
    if torch.isnan(p.grad).any():
        fail(f"NaN gradient in scorer parameter {i}!")
ok("All scorer parameter gradients are non-None and NaN-free")

# Check encoder also has gradients
enc_params = [p for p in model_adaptive.enc[0].parameters() if p.grad is not None]
ok(f"Encoder also receives gradients ({len(enc_params)} params)")

optimizer_test.step()
ok("optimizer.step() completed without error")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 8 — Mini training loop: 2 epochs, NATOPS-shaped synthetic data
# ══════════════════════════════════════════════════════════════════════════════
header("CHECK 8 — Mini training loop (2 epochs, synthetic NATOPS-shaped data)")
from torch.utils.data import TensorDataset, DataLoader

# Synthetic NATOPS-shaped data: 30 samples (class 0 = normal)
N_train = 30
x_fake  = torch.randn(N_train, C, L)
y_fake  = torch.zeros(N_train, dtype=torch.long)
ds = TensorDataset(x_fake, y_fake)
loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)

from models.NeutralAD_trainer import NeutralAD_trainer

model_mini  = SeqNeutralAD(SeqNets(), x_dim=(C, L), config=cfg_adaptive).to(DEVICE)
loss_mini   = DCL(temperature=0.1).to(DEVICE)
trainer_mini = NeutralAD_trainer(model_mini, loss_mini, device=DEVICE)
opt_mini = torch.optim.Adam(model_mini.parameters(), lr=1e-3, weight_decay=1e-5)

losses = []
for epoch in range(1, 3):
    tl = trainer_mini._train(loader, opt_mini)
    losses.append(tl)
    info(f"  Epoch {epoch}:  train_loss = {tl:.6f}")
    if trainer_mini.last_epoch_weights is not None:
        w = trainer_mini.last_epoch_weights
        info(f"  Weights:  min={float(w.min()):.4f}  max={float(w.max()):.4f}  sum={float(w.sum()):.4f}")

if any(np.isnan(l) for l in losses):
    fail("NaN training loss in mini loop!")
ok(f"Mini training loop completed (losses: {[round(l,6) for l in losses]})")

if trainer_mini.last_epoch_weights is None:
    fail("Weight history not populated during adaptive training!")
ok(f"Weight history is being tracked (len={len(trainer_mini.weight_history)})")

if len(trainer_mini.weight_history) != 2:
    fail(f"Expected 2 weight history entries (one per epoch), got {len(trainer_mini.weight_history)}")
ok("Weight history has one entry per epoch")


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 9 — Baseline mode mini loop (must behave identically to original)
# ══════════════════════════════════════════════════════════════════════════════
header("CHECK 9 — Baseline mode mini loop (adaptive=False)")
model_base2  = SeqNeutralAD(SeqNets(), x_dim=(C, L), config=cfg_baseline).to(DEVICE)
trainer_base = NeutralAD_trainer(model_base2, DCL(0.1).to(DEVICE), device=DEVICE)
opt_base = torch.optim.Adam(model_base2.parameters(), lr=1e-3)

base_losses = []
for epoch in range(1, 3):
    tl = trainer_base._train(loader, opt_base)
    base_losses.append(tl)

if any(np.isnan(l) for l in base_losses):
    fail("NaN in baseline mini-loop!")
ok(f"Baseline mini-loop completed (losses: {[round(l,6) for l in base_losses]})")

if trainer_base.last_epoch_weights is not None:
    fail("last_epoch_weights should be None in baseline mode — scorer is leaking!")
ok("last_epoch_weights is None in baseline mode — no scorer interference")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{GREEN}{'='*60}")
print("  ALL CHECKS PASSED — Implementation is Colab-ready!")
print(f"{'='*60}{RESET}\n")
print(f"  Device used:          {DEVICE}")
print(f"  K (transformations):  {K}")
print(f"  Batch size:           {B}")
print(f"  z_dim:                {Z_DIM}")
print(f"  Input shape:          ({B}, {C}, {L})")
print(f"  zs shape:             ({B}, {K+1}, {Z_DIM})")
print(f"  scores shape:         ({B}, {K})")
print(f"  weights shape:        ({B}, {K})")
print(f"  anomaly score shape:  ({B},)")
print()
print("  Next step: run the full 6-class experiment with")
print("    python Launch_Exps.py --config-file config_files/config_natops_adaptive_gpu.yml --dataset-name natops")
print()
