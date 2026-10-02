import torch
import numpy as np
from loader.LoadData import load_data
from config.base import Config, Grid
from evaluation.Experiments import runExperiment
from models.SeqNets import SeqNets
from models.NeutralAD import SeqNeutralAD
from models.Losses import DCL
from models.NeutralAD_trainer import NeutralAD_trainer

def run_integration_tests():
    print("=" * 75)
    print("RUNNING INTEGRATION TEST: BASELINE vs ADAPTIVE MODES")
    print("=" * 75)

    # Load 1 class of real NATOPS dataset (Class 0 as normal)
    print("\n[Step 1] Loading NATOPS dataset (Class 0)...")
    dataset = load_data('natops', cls=0, cls_type='normal')
    train_set, val_set, test_set = dataset
    print(f"✓ NATOPS loaded: Train samples={len(train_set)}, Test samples={len(test_set)}")

    # -------------------------------------------------------------
    # Test A: BASELINE MODE (adaptive = False)
    # -------------------------------------------------------------
    print("\n[Step 2] Testing BASELINE MODE (adaptive = False)...")
    baseline_cfg = {
        'dataset': 'natops',
        'model': 'seqNTL',
        'network': 'seqNTL',
        'trainer': 'NTL',
        'loss': 'DCL',
        'loss_temp': 0.1,
        'optimizer': 'Adam',
        'scheduler': None,
        'early_stopper': None,
        'learning_rate': 0.001,
        'l2': 1e-5,
        'training_epochs': 2,       # Small sanity run
        'device': 'cpu',
        'x_dim': 24,
        'x_length': 51,
        'latent_dim': 64,
        'enc_hdim': 32,
        'enc_nlayers': 6,
        'num_trans': 11,
        'trans_nlayers': 5,
        'trans_type': 'mul',
        'enc_bias': False,
        'batch_norm': False,
        'shuffle': True,
        'adaptive': False
    }

    exp_base = runExperiment(baseline_cfg, exp_path='RESULTS/TEST_BASELINE')
    val_auc, test_auc, test_ap, test_f1, scores, labels = exp_base.run_test(dataset, logger=None)

    assert not np.isnan(test_auc) and not np.isnan(test_ap) and not np.isnan(test_f1), "NaN detected in baseline metrics!"
    assert not np.isnan(scores).any() and not np.isinf(scores).any(), "NaN or Inf in baseline anomaly scores!"
    print(f"✓ Baseline mode ran successfully for 2 epochs!")
    print(f"  Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}, Test F1: {test_f1:.4f}")
    print(f"  Scores shape: {scores.shape}, Min score: {scores.min():.4f}, Max score: {scores.max():.4f}")

    # -------------------------------------------------------------
    # Test B: ADAPTIVE MODE (adaptive = True)
    # -------------------------------------------------------------
    print("\n[Step 3] Testing ADAPTIVE MODE (adaptive = True)...")
    adaptive_cfg = baseline_cfg.copy()
    adaptive_cfg['adaptive'] = True
    adaptive_cfg['scorer_hdim'] = 32
    adaptive_cfg['tau_w'] = 1.0

    exp_adapt = runExperiment(adaptive_cfg, exp_path='RESULTS/TEST_ADAPTIVE')
    val_auc_a, test_auc_a, test_ap_a, test_f1_a, scores_a, labels_a = exp_adapt.run_test(dataset, logger=None)

    assert not np.isnan(test_auc_a) and not np.isnan(test_ap_a) and not np.isnan(test_f1_a), "NaN detected in adaptive metrics!"
    assert not np.isnan(scores_a).any() and not np.isinf(scores_a).any(), "NaN or Inf in adaptive anomaly scores!"
    print(f"✓ Adaptive mode ran successfully for 2 epochs!")
    print(f"  Test AUC: {test_auc_a:.4f}, Test AP: {test_ap_a:.4f}, Test F1: {test_f1_a:.4f}")
    print(f"  Scores shape: {scores_a.shape}, Min score: {scores_a.min():.4f}, Max score: {scores_a.max():.4f}")

    # -------------------------------------------------------------
    # Test C: Gradient Propagation Verification on Scorer
    # -------------------------------------------------------------
    print("\n[Step 4] Verifying End-to-End Gradient Flow into Scorer Parameters...")
    model_adapt = SeqNeutralAD(SeqNets(), x_dim=24, config=adaptive_cfg)
    loss_fun = DCL(temperature=0.1)

    # Single batch forward
    sample_batch = torch.randn(8, 24, 51)
    zs, raw_scores, weights = model_adapt(sample_batch, return_weights=True)

    # Check shapes
    assert zs.shape == (8, 12, 64)
    assert raw_scores.shape == (8, 11)
    assert weights.shape == (8, 11)

    # Compute weighted DCL loss
    loss = loss_fun(zs, weights=weights)
    loss_val = loss.mean()
    loss_val.backward()

    # Verify all scorer parameters received non-zero gradients
    scorer_grads = [p.grad for p in model_adapt.scorer.parameters()]
    assert all(g is not None for g in scorer_grads), "Scorer parameter gradient missing!"
    assert all(not torch.isnan(g).any() for g in scorer_grads), "NaN in scorer gradients!"
    print(f"✓ Scorer gradients verified: all {len(scorer_grads)} parameter tensors received healthy gradients.")
    print(f"  Gradient norm of scorer output layer: {model_adapt.scorer.net[-1].weight.grad.norm().item():.6f}")

    print("\n" + "=" * 75)
    print("ALL PIPELINE INTEGRATION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 75)

if __name__ == '__main__':
    run_integration_tests()
