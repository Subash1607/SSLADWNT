# NeuTraL-AD Baseline Evaluation: NATOPS Dataset

This document records the exact configuration, architecture mapping, and benchmark results for the original **fixed-$K$ NeuTraL-AD** baseline on the complete 6-class NATOPS dataset. This serves as the benchmark against which the future **Adaptive NeuTraL-AD** extension will be evaluated.

---

## 1. Baseline Configuration

The baseline is evaluated using the original NeuTraL-AD architecture and hyperparameters with fixed number of transformations ($K=11$):

| Parameter | Baseline Setting | Description |
| :--- | :--- | :--- |
| **Dataset** | NATOPS (Naval Air Training and Operating Procedures Standardization) | 6 gesture classes, 24 sensor channels, sequence length 51 |
| **Problem Formulation** | One-vs-Rest Anomaly Detection | 1 normal gesture class vs 5 anomalous gesture classes |
| **Classes Evaluated** | 6 (Classes 0 to 5) | Full dataset reproduction |
| **Model Type** | `SeqNeutralAD` | 1D-CNN based NeuTraL-AD for multivariate time series |
| **Number of Transformations ($K$)** | `11` | Fixed set of neural transformations (no adaptive selection/weighting) |
| **Transformation Type** | `mul` | Sigmoid-gated multiplicative transformation: $T_i(x) = \sigma(f_i(x)) \odot x$ |
| **Transformation Layers** | `5` | 1D Conv + InstanceNorm + Residual blocks |
| **Encoder Layers** | `6` | 1D Residual Convolution blocks with downsampling |
| **Encoder Hidden Dim** | `32` | Base hidden channels |
| **Latent Representation Dim ($z$)**| `64` | Embedding dimension of $z_0, z_1, \dots, z_K$ |
| **Loss Function** | `DCL` (Deterministic Contrastive Loss) | Temperature $\tau = 0.1$ |
| **Training Epochs** | `100` | Full training schedule |
| **Optimizer** | Adam | Learning rate: `0.001`, Weight decay ($L_2$): `1e-5` |
| **LR Scheduler** | StepLR | Step size: `100`, Gamma: `0.5` |
| **Early Stopper** | Patience | Patience: `100` (on training loss) |
| **Batch Size** | Auto ($N_{\text{train}} / 4 \approx 7$) | Original NeuTraL-AD default for small datasets |
| **Execution Hardware** | CPU | `torch.device('cpu')` |

---

## 2. Codebase Implementation Mapping

The core components of the baseline are implemented across the following files:

| Component | File & Class / Function | Description |
| :--- | :--- | :--- |
| **Neural Transformations** | [`models/SeqNets.py`](file:///c:/Users/SUBASH/Desktop/Project/NeuTraL-AD/models/SeqNets.py): `SeqTransformNet`, `SeqNets._make_nets` | Instantiates $K=11$ independent 1D residual convolutional networks (`self.trans = nn.ModuleList([SeqTransformNet(...) for _ in range(num_trans)])`). |
| **Transformation Representations** | [`models/NeutralAD.py`](file:///c:/Users/SUBASH/Desktop/Project/NeuTraL-AD/models/NeutralAD.py): `SeqNeutralAD.forward`<br>[`models/SeqNets.py`](file:///c:/Users/SUBASH/Desktop/Project/NeuTraL-AD/models/SeqNets.py): `SeqEncoder` | Applies each transformation $T_i(x)$, stacks original and transformed sequences $[x, T_1(x), \dots, T_K(x)]$, and maps them through `SeqEncoder` to produce latent matrix $z \in \mathbb{R}^{B \times (K+1) \times d_z}$. |
| **Deterministic Contrastive Loss (DCL)** | [`models/Losses.py`](file:///c:/Users/SUBASH/Desktop/Project/NeuTraL-AD/models/Losses.py): `DCL.forward` | Computes cosine-similarity contrastive loss pulling transformed embeddings $z_1, \dots, z_K$ close to original $z_0$ while repelling different transformation pairs from each other. |
| **Anomaly Scoring** | [`models/Losses.py`](file:///c:/Users/SUBASH/Desktop/Project/NeuTraL-AD/models/Losses.py): `DCL.forward(z, eval=True)`<br>[`models/NeutralAD_trainer.py`](file:///c:/Users/SUBASH/Desktop/Project/NeuTraL-AD/models/NeutralAD_trainer.py): `NeutralAD_trainer.detect_outliers` | Test sample anomaly score is defined as the sum of transformation divergence under the DCL loss: $\mathcal{S}(x) = \sum_{k=1}^K \ell_{\text{DCL}}^{(k)}(x)$. Higher score indicates anomaly. |
| **Training & Evaluation Loop** | [`models/NeutralAD_trainer.py`](file:///c:/Users/SUBASH/Desktop/Project/NeuTraL-AD/models/NeutralAD_trainer.py): `_train`, `train`, `detect_outliers` | Manages epoch loop, loss backpropagation, evaluation on validation/test sets, and metric calculations (AUROC, AP, F1). |
| **NATOPS Configuration** | [`config_files/config_natops.yml`](file:///c:/Users/SUBASH/Desktop/Project/NeuTraL-AD/config_files/config_natops.yml)<br>[`config/Dataset_Class.py`](file:///c:/Users/SUBASH/Desktop/Project/NeuTraL-AD/config/Dataset_Class.py): `class natops` | Specifies hyperparameters (100 epochs, $K=11$, DCL, CPU) and defines dataset metadata (`num_cls = 6`). |
| **Data Ingestion & Splitting** | [`loader/LoadData.py`](file:///c:/Users/SUBASH/Desktop/Project/NeuTraL-AD/loader/LoadData.py): `load_data`, `split_in_out` | Loads NATOPS `.npy` arrays, filters normal class samples for training, and constructs binary labeled test sets. |

---

## 3. NATOPS Baseline Results (Full 6 Classes, 100 Epochs)

The baseline model was evaluated across all 6 NATOPS gesture classes in the one-vs-rest anomaly detection setup:

| Normal Class | Gesture Index | AUROC | Average Precision (AP) | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| Class 1 | Gesture 0 | 0.9331 | 0.9853 | 0.9400 |
| Class 2 | Gesture 1 | 0.8522 | 0.9698 | 0.8933 |
| Class 3 | Gesture 2 | 0.9076 | 0.9815 | 0.9000 |
| Class 4 | Gesture 3 | 1.0000 | 1.0000 | 1.0000 |
| Class 5 | Gesture 4 | 0.9996 | 0.9999 | 0.9933 |
| Class 6 | Gesture 5 | 0.9987 | 0.9997 | 0.9867 |
| **Overall Mean** | — | **0.9485** (94.85%) | **0.9894** (98.94%) | **0.9522** (95.22%) |

### Observations:
1. **Easy Classes:** Classes 4, 5, and 6 achieve near-perfect separation ($\ge 99.8\%$ AUROC, $100\%$ AP on Class 4), demonstrating that gesture kinematics for these movements are distinctly separated from others in the learned representation space.
2. **Challenging Classes:** Class 2 (AUROC 85.22%) and Class 3 (AUROC 90.76%) exhibit some overlap with other classes, which represents the primary area where an **Adaptive Transformation** mechanism (e.g., pruning redundant transformations or weighting high-discriminability transformations) can provide measurable gains.

---

## 4. Current Status: Baseline vs. Extension

- [x] **Fixed-$K$ NeuTraL-AD Baseline:** Verified and documented.
- [ ] **Adaptive Neural Transformation Extension:** Not yet implemented.
  - Future work will introduce transformation importance scoring, adaptive weighting in DCL, and transformation pruning/selection mechanisms without altering this baseline documentation.
