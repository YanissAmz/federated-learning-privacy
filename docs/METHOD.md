# Method — what each piece does and why

This doc walks through the math behind the four moving parts of this repo:

1. FedAvg with optional Central DP on state-dict diffs
2. iDLG label inference + image reconstruction
3. ε → σ calibration via zCDP composition
4. Why the resulting privacy/utility curves look the way they do

Read it as a companion to the code in `src/`, not as a replacement for the original papers.

---

## 1 · FedAvg + Central DP

### Vanilla FedAvg

For round `t` with `N` clients, server holds global parameters `θ_t`. Each client `k`:

1. Receives `θ_t`, copies to local `θ_t^k`.
2. Trains locally for `E` epochs with batch size `B` and lr `η`.
3. Returns the post-train state `θ_{t+1}^k`.

Server averages and updates:

```
θ_{t+1}  =  (1/N) Σ_k θ_{t+1}^k
        =  θ_t + (1/N) Σ_k (θ_{t+1}^k - θ_t)
        =  θ_t + (1/N) Σ_k Δ^k
```

where `Δ^k` is client k's update (state-dict diff). Implementation: `FLServer.aggregate()` in `src/fl/server.py`.

### Adding Central DP

To make the server's released `θ_{t+1}` (ε, δ)-DP with respect to any single client's data:

1. Each client's diff `Δ^k` is **L2-clipped** to bound sensitivity:
   ```
   Δ^k_clipped = Δ^k · min(1, C / ‖Δ^k‖_2)
   ```
2. Server computes `(1/N) Σ_k Δ^k_clipped`.
3. Server adds **one** Gaussian noise term:
   ```
   noise  ~  N(0, (Cσ/N)² · I_D)
   θ_{t+1} = θ_t + (1/N) Σ_k Δ^k_clipped  +  noise
   ```

Sensitivity argument: the L2 distance between the released sums (with vs. without one client) is at most `2C` (one client adding their max-norm clipped update vs. the same client absent). After dividing by `N`, sensitivity of the released average is `2C/N` — but since clipping bounds *each* client at `C` rather than the joint sum, we use **single-client sensitivity = `C`** for the σ calibration. Standard DP-FedAvg convention.

The σ used here is per-coordinate noise std for the Gaussian mechanism; total noise vector L2 norm in expectation is `√D · σ`.

**Why the repo's variant adds noise once on the average**: with per-client noise σ' the variance of the averaged update is `(σ')² / N` (independent Gaussians), so to match the formal guarantee one would need `σ' = σ · √N`, i.e. noise √N× larger per client. Adding it once at the end avoids this √N inflation. See commit history for the bug fix.

---

## 2 · iDLG (improved Deep Leakage from Gradients)

### The original DLG attack (Zhu et al., NeurIPS 2019)

Given gradients `g* = ∇_θ L(f(x*), y*)` shared by a victim, attacker:

1. Initializes random dummy `x̂` and `ŷ`.
2. Optimizes `(x̂, ŷ)` to minimize the **gradient-matching loss**:
   ```
   L_DLG  =  Σ_l ‖∇_θ L(f(x̂), ŷ)_l - g*_l‖_2²    +    λ · TV(x̂)
   ```
3. After ~300 L-BFGS iterations, `x̂ ≈ x*`.

### iDLG's improvement (Zhao et al., 2020)

Observation: for cross-entropy loss with a final linear layer, the gradient of the logit for the **true class** is `softmax(z)[y*] − 1 < 0`, while for any other class it is `softmax(z)[c] > 0`. The same sign pattern appears on the **bias** of the last linear layer:

```python
def infer_label_from_gradients(grads, num_classes):
    # find the last 1-D bias of size num_classes
    for g in reversed(grads):
        if g.ndim == 1 and g.shape[0] == num_classes:
            return g.argmin().item()  # the unique negative index
```

Once `ŷ` is fixed (one-hot from the inferred label), the optimizer has half as many parameters to fit, and the reconstruction is sharper. iDLG also avoids the soft-label pathology where DLG's `ŷ` drifts.

Implementation: `src/attacks/dlg.py:infer_label_from_gradients` and the `variant="idlg"` branch of `DLGAttack.attack()`.

### When iDLG fails

- Batch size > 1: gradients are sums; per-sample identity is lost.
- Activation functions without strong gradient signal (e.g. ReLU dead zones).
- DP-protected gradients: noise ‖N(0, σ²I_D)‖_2 dominates the signal `‖g*‖_2 ≤ C` when `σ √D ≫ C`.

---

## 3 · ε → σ via zCDP composition

### Per-release Gaussian mechanism

Given a query `f` with global L2 sensitivity `Δ`, releasing `f(D) + N(0, σ²I)` is `(α, αΔ²/(2σ²))`-RDP for any `α > 1`. For `Δ = 1` (sensitivity normalized by clipping):

```
ρ(α)  =  α / (2σ²)        — Rényi DP order α, parameter ρ
```

For `T` independent Gaussian releases at fixed σ:

```
ρ_total(α) = T · α / (2σ²)
```

### RDP → (ε, δ)-DP conversion

For any `α > 1`:

```
ε(α) = ρ_total(α) + ln(1/δ) / (α − 1)
```

The tightest ε is found by minimizing over `α`. Setting derivative to zero:

```
T / (2σ²) = ln(1/δ) / (α − 1)²
α* = 1 + σ √(2 ln(1/δ) / T)
```

Plugging back in (and using `u = √ρ` substitution for clean inversion):

```
let  u = -√(ln(1/δ)) + √(ln(1/δ) + ε)
     ρ = u²
     σ = √(T / (2ρ))
```

This is what `noise_multiplier_from_epsilon(epsilon, delta, num_rounds)` returns. For ε=8, δ=1e-5, T=20:

```
ln(1/δ) = ln(1e5) ≈ 11.51
u       = -3.39 + √(19.51) ≈ 1.03
ρ       ≈ 1.06
σ       = √(20 / 2.12) ≈ 3.07
```

vs. simple-composition σ ≈ 13.6 — a **4×** improvement on the privacy/utility frontier.

---

## 4 · Why naive Central DP collapses on small federations

The core inequality (per round, after averaging over `N` clients):

```
‖noise‖_2          ≈  √D · C · σ / N
‖signal‖_2         ≤  C
SNR per coordinate  =  1 / (σ √D)
```

For SimpleCNN on CIFAR-10:
- D ≈ 1.1 M → √D ≈ 1050
- N = 5
- C = 1.0 (max grad norm)

| Target ε | σ (RDP, T=20) | σ_avg = σ/N | Noise L2 norm | SNR per coord |
|---|---|---|---|---|
| 64 | 0.60 | 0.12 | ≈ 126 | 1 / 630 |
| 16 | 1.71 | 0.34 | ≈ 360 | 1 / 1800 |
| 4 | 5.80 | 1.16 | ≈ 1220 | 1 / 6090 |
| 1 | 21.9 | 4.38 | ≈ 4600 | 1 / 23 000 |

In all rows, the per-coord SNR is in `1e-3` to `1e-5`. SGD over 20 rounds doesn't recover from this — the model does an essentially random walk in parameter space. Empirically the test accuracy stays near initialization (10% on CIFAR-10 = random guess for 10 classes).

### How production systems escape this curse

1. **Subsampling amplification**: each round, only a small fraction of clients participate. (ε, δ)-DP after subsampling at rate `q` is amplified to roughly (`qε`, `qδ`)-DP per round → smaller σ for same total budget.
2. **DP-SGD per-sample clipping**: sensitivity is `1` per **sample**, not per client. With `B` samples per batch, signal scales with √B but noise stays at σ — net SNR improves with √B.
3. **Many clients**: `N = 10⁶` means `‖noise‖ / ‖signal‖` shrinks by `√(10⁶/5)` ≈ 450× compared to here.
4. **Sparsified or low-rank updates**: only release a `k`-dim projection (k ≪ D), reducing noise norm by `√(k/D)`.

This repo doesn't implement any of those — it implements the textbook primitive faithfully, and the experiments make visible *why* you'd want them in production. That's the pedagogical point.

---

## References

- McMahan, Moore, Ramage, Hampson, Arcas. **Communication-Efficient Learning of Deep Networks from Decentralized Data** (FedAvg). AISTATS 2017. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
- Zhu, Liu, Han. **Deep Leakage from Gradients**. NeurIPS 2019. [arXiv:1906.08935](https://arxiv.org/abs/1906.08935)
- Zhao, Mopuri, Bilen. **iDLG: Improved Deep Leakage from Gradients**. 2020. [arXiv:2001.02610](https://arxiv.org/abs/2001.02610)
- McMahan, Ramage, Talwar, Zhang. **Learning Differentially Private Recurrent Language Models** (DP-FedAvg). ICLR 2018. [arXiv:1710.06963](https://arxiv.org/abs/1710.06963)
- Bun, Steinke. **Concentrated Differential Privacy** (zCDP composition). TCC 2016. [arXiv:1605.02065](https://arxiv.org/abs/1605.02065)
- Mironov. **Rényi Differential Privacy**. CSF 2017. [arXiv:1702.07476](https://arxiv.org/abs/1702.07476)
- Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang. **Deep Learning with Differential Privacy** (DP-SGD + moments accountant). CCS 2016. [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)
- Yin, Mallya, Vahdat, Alvarez, Kautz, Molchanov. **See through Gradients: Image Batch Recovery via GradInversion**. CVPR 2021. [arXiv:2104.07586](https://arxiv.org/abs/2104.07586)
