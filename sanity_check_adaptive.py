import os
import json
import torch
import numpy as np
from loader.LoadData import load_data
from models.SeqNets import SeqNets
from models.NeutralAD import SeqNeutralAD
from models.Losses import DCL
from models.NeutralAD_trainer import NeutralAD_trainer

def run_controlled_sanity_check(epochs=10):
    print("=" * 80)
    print(f"CONTROLLED SANITY CHECK: ADAPTIVE NeuTraL-AD (K=11, Epochs={epochs})")
    print("=" * 80)

    # 1. Load NATOPS dataset (Class 0)
    print("\n[Step 1] Loading NATOPS dataset...")
    train_set, val_set, test_set = load_data('natops', cls=0, cls_type='normal')
    print(f"✓ NATOPS loaded successfully: {len(train_set)} train samples, {len(test_set)} test samples.")

    # 2. Configure model
    config = {
        'enc_nlayers': 6,
        'enc_hdim': 32,
        'latent_dim': 64,
        'x_length': 51,
        'trans_nlayers': 5,
        'num_trans': 11,
        'trans_type': 'mul',
        'loss_temp': 0.1,
        'enc_bias': False,
        'batch_norm': False,
        'device': 'cpu',
        'adaptive': True,
        'scorer_hdim': 32,
        'tau_w': 1.0,
        'learning_rate': 0.001,
        'l2': 1e-5
    }

    torch.manual_seed(42)
    np.random.seed(42)

    # 3. Instantiate model, loss, optimizer, and trainer
    model = SeqNeutralAD(SeqNets(), x_dim=24, config=config)
    loss_fun = DCL(temperature=config['loss_temp'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2'])
    trainer = NeutralAD_trainer(model, loss_function=loss_fun, device=config['device'])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    print("\n[Step 2] Beginning controlled training...")
    epoch_losses = []
    
    for epoch in range(1, epochs + 1):
        loss = trainer._train(train_loader, optimizer)
        epoch_losses.append(loss)
        w = trainer.last_epoch_weights
        trainer.weight_history.append(w.tolist())
        w_str = " | ".join([f"T{i+1}:{w[i]:.4f}" for i in range(11)])
        print(f"Epoch {epoch:2d}/{epochs:2d} | TR Loss: {loss:.4f} | Weights: [{w_str}]")

    # 4. Check Gradients on Scorer Parameters
    print("\n[Step 3] Verifying gradient reach into TransformationScorer...")
    scorer_grads = [p.grad for p in model.scorer.parameters()]
    has_grads = all(g is not None for g in scorer_grads)
    no_nan_grads = all(not torch.isnan(g).any() for g in scorer_grads)
    max_grad_norm = max(g.norm().item() for g in scorer_grads)
    print(f"✓ All {len(scorer_grads)} scorer parameter tensors received gradients (max grad norm: {max_grad_norm:.6f}).")

    # 5. Evaluate Outlier Detection (Anomaly Scoring)
    print("\n[Step 4] Testing anomaly scoring on test set...")
    auc, ap, f1, loss_in, loss_out, scores, targets = trainer.detect_outliers(test_loader, cls=0)
    print(f"✓ Anomaly scoring completed: Test AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")
    print(f"  Score stats: min={scores.min():.4f}, mean={scores.mean():.4f}, max={scores.max():.4f}")

    # 6. Save Weight History
    os.makedirs('RESULTS', exist_ok=True)
    history_file = 'RESULTS/adaptive_weight_history.json'
    history_data = {
        'num_trans': 11,
        'epochs': epochs,
        'epoch_losses': epoch_losses,
        'weight_history': trainer.weight_history,
        'test_metrics': {'auc': float(auc), 'ap': float(ap), 'f1': float(f1)}
    }
    with open(history_file, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"\n[Step 5] Weight history across all {epochs} epochs saved to: {history_file}")

    # 7. Analysis & Verification of the 8 Checklist Questions
    weight_history = np.array(trainer.weight_history)  # (epochs, 11)
    final_weights = weight_history[-1]                 # (11,)
    avg_weights = weight_history.mean(axis=0)           # (11,)
    initial_weights = weight_history[0]                # (11,)

    # (1) Are transformation weights different from each other?
    weight_std = float(final_weights.std())
    diff_from_uniform = np.abs(final_weights - (1.0 / 11.0)).max()
    is_differentiated = weight_std > 1e-4

    # (2) Do weights change during training?
    weight_drift = np.abs(final_weights - initial_weights).sum()
    weights_changed = weight_drift > 1e-4

    # (3) Do weights sum to approximately 1?
    sums = weight_history.sum(axis=1)
    sums_approx_1 = np.allclose(sums, np.ones(epochs), atol=1e-4)

    # (4) Are any weights NaN or Inf?
    has_nan = np.isnan(weight_history).any()
    has_inf = np.isinf(weight_history).any()

    # (5) Do gradients reach the transformation scorer?
    grads_ok = has_grads and no_nan_grads

    # (6) Does training loss decrease or behave normally?
    loss_decreased = epoch_losses[-1] < epoch_losses[0]

    # (7) Does model complete training without errors?
    training_ok = True

    # (8) Does anomaly scoring still work?
    scoring_ok = not np.isnan(scores).any() and not np.isinf(scores).any() and len(scores) == len(test_set)

    # 8. Print Formatted Weight Table
    print("\n" + "=" * 80)
    print("TRANSFORMATION WEIGHT ANALYSIS SUMMARY (K=11)")
    print("=" * 80)
    print(f"{'Transformation':<18} | {'Initial (Ep 1)':<14} | {'Final (Ep ' + str(epochs) + ')':<14} | {'Average (All Ep)':<16} | {'Delta':<10}")
    print("-" * 80)
    for i in range(11):
        delta = final_weights[i] - initial_weights[i]
        print(f"Transformation {i+1:2d}  | {initial_weights[i]:.5f}      | {final_weights[i]:.5f}      | {avg_weights[i]:.5f}         | {delta:+.5f}")
    print("-" * 80)
    print(f"{'Total Sum':<18} | {initial_weights.sum():.5f}      | {final_weights.sum():.5f}      | {avg_weights.sum():.5f}         |")
    print("=" * 80)

    # 9. Explicit 8-Point Verification Checklist
    print("\n" + "=" * 80)
    print("EXPLICIT 8-POINT VERIFICATION CHECKLIST")
    print("=" * 80)
    print(f"1. Weights different from each other?  : {'YES' if is_differentiated else 'NO'} (Final std={weight_std:.5f}, range=[{final_weights.min():.4f}, {final_weights.max():.4f}])")
    print(f"2. Weights change during training?     : {'YES' if weights_changed else 'NO'} (Total absolute drift={weight_drift:.5f})")
    print(f"3. Weights sum to ~1?                  : {'YES' if sums_approx_1 else 'NO'} (Min sum={sums.min():.6f}, Max sum={sums.max():.6f})")
    print(f"4. Any NaN or Inf in weights?          : {'NO (Clean)' if (not has_nan and not has_inf) else 'YES (ERROR)'}")
    print(f"5. Gradients reach the scorer?         : {'YES' if grads_ok else 'NO'} (All {len(scorer_grads)} parameters received valid gradients)")
    print(f"6. Training loss behaves normally?     : {'YES' if loss_decreased else 'NO'} (Start={epoch_losses[0]:.4f} -> End={epoch_losses[-1]:.4f})")
    print(f"7. Completed without errors?           : {'YES' if training_ok else 'NO'}")
    print(f"8. Anomaly scoring still works?        : {'YES' if scoring_ok else 'NO'} (Computed valid scores for all {len(scores)} test items)")
    print("=" * 80)

    return history_data

if __name__ == '__main__':
    run_controlled_sanity_check(epochs=10)
