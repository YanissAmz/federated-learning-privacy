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

## v0.3 — integrity attacks, robust aggregation, defense-aware adversary

The current repo only handles **passive privacy attacks** (the attacker reads
gradients). The complementary threat is **active integrity attacks** (the
attacker submits malicious updates to bias the global model — or, more
interestingly, to **deanonymize** a target client).

> **Origin.** The gradient-suppression attack below grew out of a federated
> learning project I built earlier in TFF / Keras on MIMIC-style clinical
> data (binary mortality prediction). There I observed that seeding all
> non-target clients with extreme negative weights collapses the FedAvg
> average so cleanly that the global model effectively traces only the one
> target client — a hybrid Byzantine-as-deanonymization attack rarely
> framed that way in the literature. v0.3 ports that idea to PyTorch +
> CIFAR-10 and pairs it with the robust aggregations that defeat it.

Planned additions:

- [ ] `src/attacks/byzantine.py` — four poisoning attacks:
  - **Sign-flip**: malicious clients return -Δ instead of +Δ.
  - **Constant attack**: malicious clients return a fixed huge value.
  - **Gradient suppression**: N-1 malicious clients submit updates with
    extreme negative values (e.g. weights initialized to ~-1e8) to saturate
    the FedAvg average so the global model traces only one target client's
    contribution. The hybrid poisoning + privacy framing — using Byzantine
    behavior to deanonymize a specific honest client rather than just to
    break utility — is the angle worth shipping.
  - **Backdoor / model-replacement** (Bagdasaryan+ 2020).
- [ ] `src/defenses/robust.py` — robust server aggregations:
  - **Coordinate-wise median** (Yin+ 2018).
  - **Trimmed mean**: drop top/bottom k% per coordinate before averaging.
  - **Krum** (Blanchard+ 2017): pick the client whose update is closest to
    its `n - f - 2` nearest neighbors.
  - **Norm-bounded clipping**: reject updates with L2 norm > threshold.
- [ ] **Defense-aware adversary**: `StealthSuppression` that respects an
  estimated norm clip and stays close to honest-client distribution to slip
  past median / trimmed-mean / Krum. Demonstrates that naive defenses fail
  against an adaptive attacker, and motivates Krum-with-norm-bound or
  certified defenses (Trimmed-Krum, FoolsGold).
- [ ] FLServer: `aggregator: str = "mean" | "median" | "trimmed_mean" | "krum"`.
- [ ] CLI: `scripts/byzantine.py` — train with `f` malicious clients out of
  `N`, sweep attack × aggregator, plot accuracy + cosine-similarity
  (target → global) matrix.
- [ ] Demo Tab 5: K malicious slider × attack type × aggregator, side-by-side
  accuracy curves.
- [ ] E2E test: with K=2 sign-flippers + median aggregator, accuracy must
  stay ≥ 90% of the no-attack baseline.

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

