# Changelog

## v0.3.0 — integrity attacks, robust aggregation, defense-aware adversary

Adds the **active threat model** to the repo: malicious clients submit
poisoned updates to bias the global model (utility attack) or to
**deanonymize** an honest target client by collusion (privacy attack via
poisoning, the v0.3 angle worth shipping). I first prototyped the
gradient-suppression idea in TFF on a clinical-data FL coursework; v0.3 is
the PyTorch port plus the robust aggregations that defeat it, and a
defense-aware adversary that motivates non-trivial defenses.

### Added

- `src/attacks/byzantine.py` — five attacks:
  - `SignFlipAttack` (Δ → -Δ) — utility baseline.
  - `ConstantAttack` — fixed huge value, defeated trivially by norm bound.
  - `GradientSuppressionAttack` — N-1 colluding clients submit pairwise-
    cancelling extreme updates; the residual is dominated by one *target*
    honest client. Used to deanonymize a specific client, not just degrade
    accuracy.
  - `StealthSuppression` — defense-aware variant that respects an estimated
    honest-norm envelope, designed to slip past coordinate-wise median /
    trimmed-mean / Krum.
  - `BackdoorAttack` — stub reserved for v0.4 (Bagdasaryan+ 2020).
- `src/defenses/robust.py` — four aggregators + a norm filter:
  - `aggregate_mean` — vanilla FedAvg, reference behavior.
  - `aggregate_median` — coordinate-wise median (Yin+ 2018).
  - `aggregate_trimmed_mean` — drop top/bottom k% per coordinate.
  - `aggregate_krum` — pick the client closest to its peers (Blanchard+ 2017).
  - `filter_by_update_norm` — pre-filter for updates exceeding an L2 bound.
- `FLServer.aggregate(aggregator='mean'|'median'|'trimmed_mean'|'krum')`
  with full DP composition (DP and robust aggregator are independent).
- `FLServer.train_round(byzantine_attacks={i: attack})` for malicious
  client injection.
- `scripts/byzantine.py` — full attack × aggregator matrix CLI sweeping
  4 attacks × 4 aggregators × N rounds, with `--quick` smoke mode and
  `--attacks`/`--aggregators` subset filters.
- `tests/test_byzantine.py` — 14 tests, including E2E "median recovers
  ≥30% of mean's error under 2 sign-flippers" gate.
- Gradio Tab 5 — accuracy + cosine-similarity heatmaps over the matrix.

### Changed

- `FLServer.aggregate()` API: previously fixed to mean; now takes an
  `aggregator` argument. Default behavior (`aggregator="mean"`) matches v0.2.
- README and ROADMAP updated with the v0.3 results matrix.

### Known limitations

- The cosine-similarity metric `cos(target_diff, global_diff)` saturates
  near 1 in early FL rounds when all clients agree on direction. A more
  sensitive deanonymization metric (residual after subtracting non-target
  honest contribution) is queued for v0.3.1.

## v0.2.0 — privacy story is now real

### Added

- **Real `FLServer.evaluate()`** on a held-out testset (CIFAR-10 by default).
  Replaced the `_evaluate_placeholder()` stub that always returned 0.0.
- **Central DP for FedAvg** in `src/defenses/dp.py`:
  `state_dict_diff`, `clip_state_diff`, `add_noise_to_diff`,
  `apply_dp_to_state_diff`, and a zCDP-tight `noise_multiplier_from_epsilon`.
  Server adds Gaussian noise *once* on the averaged update (not per-client) to
  match the formal sensitivity argument.
- **iDLG attack variant** (Zhao+ 2020) in `src/attacks/dlg.py`: infers the true
  label from the sign pattern of the last linear layer's bias gradient, then
  optimizes the input only. Sharper reconstructions than soft-label DLG.
- **CLI scripts** under `scripts/`:
  - `train.py` — one FL run with optional `--epsilon` for Central DP.
  - `attack.py` — single iDLG attack with `--defense {none, dp}`, saves
    target/recon PNG + optimization-trajectory GIF + metrics JSON.
  - `evaluate.py` — full privacy/utility matrix (4 DP budgets × 3 samples) +
    aggregated `summary.json`.
  - `make_figures.py` — `accuracy_curves.png` + `reconstructions_grid.png` for
    the README.
  - `update_readme.py` — patches the README tables in place from the JSONs.
- **Gradio demo** (`src/demo/app.py`) with four tabs:
  1. FL training curves under different DP budgets.
  2. Live iDLG attack — no defense, watch the reconstruction emerge.
  3. Live iDLG attack — with DP, watch the reconstruction collapse to noise.
  4. Privacy/utility tradeoff plot.
- **Two model architectures** (`src/fl/model.py:build_model`): `SimpleCNN`
  (~1.1 M params, where Central DP collapses) and `TinyMLP` (~31 K params,
  where Central DP gives a usable curve). Selectable via `--model`.
- **End-to-end privacy regression test** (`tests/test_e2e_privacy.py`):
  asserts that DP on the gradients drops the iDLG reconstruction PSNR by
  ≥ 3 dB compared to the undefended attack on the same target.
- **`docs/METHOD.md`** — derivations of the FedAvg+DP integration, iDLG label
  inference, zCDP composition, and an honest analysis of why naive Central DP
  collapses on small federations (Gaussian-mechanism curse of dimensionality).
- **`docs/ROADMAP.md`** — milestones for v0.3 (Byzantine attacks + robust
  aggregation), v0.4 (DP-SGD + Opacus), v0.5 (non-IID + Flower), v0.6
  (GradInversion + Inverting Gradients + R-GAP).

### Changed

- README rewritten to:
  - Describe what is *actually* implemented and reproducible.
  - Show the empirical accuracy curves under DP (with the honest finding that
    naive Central DP collapses on this scale).
  - Document the zCDP-tight σ derivation explicitly.
  - Acknowledge the Gaussian-mechanism dimensionality curse and point at
    DP-SGD / Opacus as the production path.
- Test suite: 11 tests (was 7), all passing on CPU under 60 s.
- Dependencies: added `imageio` and `pillow` for the attack GIF rendering.

### Removed

- Empty nested directories `src/attacks/attack/` and
  `src/defenses/defense/` (dead structure).

### Fixed

- `aggregate()` now adds Gaussian noise once on the averaged update, rather
  than independently per client. Per-client noise inflated the effective
  variance by √N relative to the formal DP guarantee.

## v0.1.0 — initial scaffold (March 2026)

- FedAvg server, FLClient, SimpleCNN, IID + non-IID partition.
- DLG attack with soft-label optimization.
- Gradient-level DP primitive (`clip_gradients`, `add_noise`, `apply_dp`).
- Tests for the DP primitive and SimpleCNN forward pass.
- README pitch describing the intended demo (which v0.2 actually delivers).
