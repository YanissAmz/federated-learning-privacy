"""Robust server-side aggregation against Byzantine clients.

Each function takes a list of client state-dicts and returns a single
aggregated state-dict. They are interchangeable replacements for the plain
`mean` aggregator inside `FLServer.aggregate()`.

References:
    Yin, Chen, Kannan, Bartlett — "Byzantine-robust distributed learning
        via robust aggregation rules" (ICML 2018) — coordinate-wise median
        and trimmed mean.
    Blanchard, El Mhamdi, Guerraoui, Stainer — "Machine Learning with
        Adversaries: Byzantine Tolerant Gradient Descent" (NIPS 2017) — Krum.
"""

from __future__ import annotations

import torch

# ----- baseline (sanity check) ----------------------------------------------


def aggregate_mean(client_states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Plain coordinate-wise arithmetic mean. Reference for behavior tests."""
    if not client_states:
        raise ValueError("client_states must be non-empty")
    out: dict[str, torch.Tensor] = {}
    for key in client_states[0]:
        stacked = torch.stack([s[key].float() for s in client_states])
        out[key] = stacked.mean(dim=0)
    return out


# ----- coordinate-wise median (Yin+ 2018) -----------------------------------


def aggregate_median(client_states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Per-coordinate median across clients.

    Robust to up to (N-1)/2 Byzantine clients per Yin+ 2018, with a slight
    accuracy cost vs mean on honest data.
    """
    if not client_states:
        raise ValueError("client_states must be non-empty")
    out: dict[str, torch.Tensor] = {}
    for key in client_states[0]:
        stacked = torch.stack([s[key].float() for s in client_states])
        out[key] = stacked.median(dim=0).values
    return out


# ----- trimmed mean ---------------------------------------------------------


def aggregate_trimmed_mean(
    client_states: list[dict[str, torch.Tensor]],
    trim_ratio: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Drop the top and bottom `trim_ratio` of values per coordinate, then mean.

    With `trim_ratio=0` this reduces to plain mean. Robust to up to
    `floor(trim_ratio * N)` Byzantine clients per side.
    """
    if not client_states:
        raise ValueError("client_states must be non-empty")
    if not 0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    n = len(client_states)
    k = int(trim_ratio * n)
    out: dict[str, torch.Tensor] = {}
    for key in client_states[0]:
        stacked = torch.stack([s[key].float() for s in client_states])
        sorted_vals, _ = stacked.sort(dim=0)
        if k > 0:
            sorted_vals = sorted_vals[k : n - k]
        out[key] = sorted_vals.mean(dim=0)
    return out


# ----- Krum (Blanchard+ 2017) -----------------------------------------------


def _flatten_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate all values into a single 1-D tensor."""
    return torch.cat([v.flatten().float() for v in state.values()])


def _krum_score(idx: int, flats: list[torch.Tensor], n_keep: int) -> float:
    """Sum of squared L2 distances from `flats[idx]` to its `n_keep` nearest peers."""
    distances = sorted(
        (flats[idx] - flats[j]).pow(2).sum().item() for j in range(len(flats)) if j != idx
    )
    return sum(distances[:n_keep])


def aggregate_krum(
    client_states: list[dict[str, torch.Tensor]],
    n_byzantine: int = 1,
) -> dict[str, torch.Tensor]:
    """Krum: pick the single client whose update is closest to its peers.

    Specifically: for each client `i`, compute the sum of squared distances
    to its `N − f − 2` nearest *other* clients, then return the state of the
    client that minimizes that sum. Tolerates up to `f` Byzantine clients
    among `N` total, provided `N ≥ 2f + 3`.

    Returns a copy of one client's state-dict, not an aggregation — that's
    fundamental to Krum.
    """
    n = len(client_states)
    if n == 0:
        raise ValueError("client_states must be non-empty")
    if n < 2 * n_byzantine + 3:
        # Krum's tolerance bound — degrades gracefully by clamping n_byzantine
        n_byzantine = max(0, (n - 3) // 2)
    n_keep = max(0, n - n_byzantine - 2)
    if n_keep == 0:
        return {k: v.float().clone() for k, v in client_states[0].items()}

    flats = [_flatten_state(s) for s in client_states]
    scores = [_krum_score(i, flats, n_keep) for i in range(n)]
    chosen = min(range(n), key=lambda i: scores[i])
    return {k: v.float().clone() for k, v in client_states[chosen].items()}


# ----- norm-bound pre-filter (cheap defense, easy to fool) ------------------


def filter_by_update_norm(
    global_state: dict[str, torch.Tensor],
    client_states: list[dict[str, torch.Tensor]],
    max_norm: float,
) -> list[dict[str, torch.Tensor]]:
    """Drop client states whose Δ has L2 norm > max_norm.

    Cheap, complementary defense: combine with median or Krum to harden
    against constant-attack and gradient-suppression adversaries that
    rely on extreme magnitudes.
    """
    kept = []
    for s in client_states:
        diff = {k: s[k].float() - global_state[k].float() for k in global_state}
        norm = _flatten_state(diff).norm(2).item()
        if norm <= max_norm:
            kept.append(s)
    return kept if kept else [client_states[0]]  # never aggregate empty


# ----- factory --------------------------------------------------------------


def build_aggregator(
    name: str,
    *,
    trim_ratio: float = 0.2,
    n_byzantine: int = 1,
):
    """Return a `(client_states) -> aggregated_state` callable by name."""
    if name == "mean":
        return aggregate_mean
    if name == "median":
        return aggregate_median
    if name == "trimmed_mean":
        return lambda states: aggregate_trimmed_mean(states, trim_ratio=trim_ratio)
    if name == "krum":
        return lambda states: aggregate_krum(states, n_byzantine=n_byzantine)
    raise ValueError(f"unknown aggregator: {name!r}")
