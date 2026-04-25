# Roadmap

## v0.2 (current) — privacy attacks and defenses

- [x] FedAvg with optional Central DP on state-dict diffs
- [x] iDLG gradient inversion attack (with label inference)
- [x] zCDP-tight ε → σ calibration
- [x] CLI for train / attack / evaluate / make_figures / update_readme
- [x] Gradio demo with live iDLG reconstruction
- [x] End-to-end "privacy story" regression test
- [x] CIFAR-10 + SimpleCNN matrix (the "DP collapses on small federations" finding)
- [x] Plan B: TinyMLP matrix to show DP can give a graceful curve at smaller D

## v0.3 — integrity attacks and robust aggregation

The current repo only handles **passive privacy attacks** (the attacker reads
gradients). The complementary threat is **active integrity attacks** (the
attacker submits malicious updates to bias or destroy the global model).

Planned additions:

- [ ] `src/attacks/byzantine.py` — three poisoning attacks:
  - **Sign-flip**: malicious clients return -Δ instead of +Δ.
  - **Constant attack**: malicious clients return a fixed huge value (gradient
    suppression — inspired by the original-author's TFF demo using
    `np.full_like(w, -1e8)` to suppress honest clients' contributions).
  - **Backdoor / model-replacement** (Bagdasaryan+ 2020): malicious clients
    train on a backdoor task and return the resulting Δ scaled to overwrite.
- [ ] `src/defenses/robust.py` — robust server aggregations:
  - **Coordinate-wise median** (Yin+ 2018).
  - **Trimmed mean**: drop top/bottom k% per coordinate before averaging.
  - **Krum** (Blanchard+ 2017): pick the client whose update is closest to its
    `n - f - 2` nearest neighbors.
  - **FoolsGold** (Fung+ 2018) — for sybil-style colluding poisoners.
- [ ] FLServer: `aggregator: str = "mean" | "median" | "trimmed_mean" | "krum"`.
- [ ] CLI: `scripts/byzantine.py` — train with `f` malicious clients out of
  `N`, sweep aggregator choice, plot accuracy under attack.
- [ ] Demo Tab 5 — visualize: 5 honest + 2 sign-flippers, accuracy curve under
  mean (collapses) vs median (recovers).

## v0.4 — DP-SGD and subsampling amplification

Today's Central DP collapses on a 1.1 M-param model with 5 clients. The
production fix is per-sample DP-SGD with subsampling amplification, which
this repo would benefit from to demonstrate the practical privacy/utility
frontier.

- [ ] Per-sample DP-SGD via Opacus integration on the client side
  (`src/fl/client.py:DPSGDClient`).
- [ ] RDP/moments-accountant for sub-sampled Gaussian (replaces the simple
  zCDP closed form for this path).
- [ ] Side-by-side comparison: Central DP vs DP-SGD on the same model.

## v0.5 — non-IID + cross-silo realism

- [ ] Dirichlet-α non-IID partition default (currently exposed via `--non-iid`).
- [ ] Per-client class imbalance metrics in the metrics dump.
- [ ] Optional Flower (`flwr`) integration so the federation actually runs as
  separate Python processes / containers, not a simulator loop.

## v0.6 — bigger attacks

- [ ] **GradInversion** (Yin+ 2021) — works at batch size > 1.
- [ ] **Inverting Gradients** (Geiping+ 2020) — uses cosine similarity loss.
- [ ] **R-GAP** (Zhu+ 2021) — closed-form gradient inversion for ResNet-style
  models.

---

## Acknowledgements

The Byzantine attack direction (v0.3) was prompted by a reading of an
academic FL project (TFF / Keras) that demonstrated gradient suppression by
seeding non-target clients with `np.full_like(w, -1e8)`. The reduction to
"a sufficiently extreme initialization is equivalent to a Byzantine update"
shaped the planned attack catalog above.
