import torch
from models.AdaptiveModule import TransformationScorer
from models.SeqNets import SeqNets
from models.NeutralAD import SeqNeutralAD

def run_tests():
    print("=" * 70)
    print("STARTING TEST SUITE: Transformation Importance Scoring Mechanism")
    print("=" * 70)

    # -------------------------------------------------------------
    # Test 1: Standalone TransformationScorer Verification (K=11)
    # -------------------------------------------------------------
    print("\n--- [Test 1] Standalone TransformationScorer with K=11 ---")
    B = 4
    K = 11
    z_dim = 64
    scorer = TransformationScorer(z_dim=z_dim, num_trans=K, h_dim=32, tau=1.0)

    # Simulate encoder output: (B, K+1, z_dim) where index 0 is anchor, 1..11 are transforms
    zs = torch.randn(B, K + 1, z_dim, requires_grad=True)
    raw_scores, weights = scorer(zs, verbose=True)

    # Dimension checks
    assert raw_scores.shape == (B, K), f"Expected raw_scores shape ({B}, {K}), got {raw_scores.shape}"
    assert weights.shape == (B, K), f"Expected weights shape ({B}, {K}), got {weights.shape}"
    print(f"✓ Output shapes verified: raw_scores: {raw_scores.shape}, weights: {weights.shape}")

    # Validity checks: weights in [0, 1] and sum to 1.0 per sample
    assert (weights >= 0.0).all() and (weights <= 1.0).all(), "Weights must be in range [0, 1]"
    weight_sums = weights.sum(dim=-1)
    print(f"✓ Weight sums per batch sample: {weight_sums.detach().cpu().numpy()}")
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), "Weights do not sum to 1.0"
    print("✓ All weights are strictly non-negative and sum to 1.0 per sample!")

    # Differentiability check (gradient backpropagation)
    dummy_loss = (weights * raw_scores).sum()
    dummy_loss.backward()
    assert zs.grad is not None and not torch.isnan(zs.grad).any(), "Gradient backpropagation into zs failed!"
    has_param_grads = all(p.grad is not None for p in scorer.parameters())
    assert has_param_grads, "Not all scorer parameters received gradients!"
    print("✓ Differentiability verified: Gradients successfully backpropagated through scorer to inputs and parameters.")

    # -------------------------------------------------------------
    # Test 2: Configurable K Check (e.g., K=5, K=15)
    # -------------------------------------------------------------
    print("\n--- [Test 2] Configurable K Support (K=5 and K=15) ---")
    for test_k in [5, 15]:
        scorer_k = TransformationScorer(z_dim=z_dim, num_trans=test_k, h_dim=32)
        zs_k = torch.randn(2, test_k + 1, z_dim)
        scores_k, weights_k = scorer_k(zs_k)
        assert scores_k.shape == (2, test_k)
        assert weights_k.shape == (2, test_k)
        assert torch.allclose(weights_k.sum(dim=-1), torch.ones(2), atol=1e-5)
        print(f"✓ Successfully verified with configurable K={test_k}: scores shape {scores_k.shape}, weights sum={weights_k.sum(dim=-1).detach().cpu().numpy()}")

    # -------------------------------------------------------------
    # Test 3: Integration with SeqNeutralAD (Baseline vs Adaptive)
    # -------------------------------------------------------------
    print("\n--- [Test 3] Integration with SeqNeutralAD (Baseline vs Adaptive Mode) ---")
    base_config = {
        'enc_nlayers': 6,
        'enc_hdim': 32,
        'latent_dim': 64,
        'x_length': 51,
        'trans_nlayers': 5,
        'num_trans': 11,
        'batch_norm': False,
        'enc_bias': False,
        'trans_type': 'mul',
        'device': 'cpu',
        'adaptive': False
    }

    # Baseline Mode
    model_baseline = SeqNeutralAD(SeqNets(), x_dim=24, config=base_config)
    x_sample = torch.randn(2, 24, 51)
    zs_baseline = model_baseline(x_sample)
    assert zs_baseline.shape == (2, 12, 64), f"Baseline output shape mismatch: {zs_baseline.shape}"
    assert model_baseline.scorer is None, "Scorer should be None in baseline mode!"
    print("✓ Baseline mode intact: forward(x) returns standard zs without modifying behavior.")

    # Adaptive Mode
    adapt_config = base_config.copy()
    adapt_config['adaptive'] = True
    adapt_config['scorer_hdim'] = 32
    adapt_config['tau_w'] = 1.0

    model_adaptive = SeqNeutralAD(SeqNets(), x_dim=24, config=adapt_config)
    assert model_adaptive.scorer is not None, "Scorer should be initialized in adaptive mode!"

    # Standard forward call in adaptive mode should still return zs for seamless training compatibility
    zs_std = model_adaptive(x_sample)
    assert zs_std.shape == (2, 12, 64)
    print("✓ Adaptive mode backward compatibility: standard forward(x) returns zs unchanged.")

    # Forward call requesting weights
    zs_ret, scores_ret, weights_ret = model_adaptive(x_sample, return_weights=True)
    assert zs_ret.shape == (2, 12, 64)
    assert scores_ret.shape == (2, 11)
    assert weights_ret.shape == (2, 11)
    assert torch.allclose(weights_ret.sum(dim=-1), torch.ones(2), atol=1e-5)
    print("✓ Adaptive mode with return_weights=True: returns (zs, scores, weights) matching exact dimensions.")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! The scoring mechanism is fully operational.")
    print("=" * 70)

if __name__ == '__main__':
    run_tests()
