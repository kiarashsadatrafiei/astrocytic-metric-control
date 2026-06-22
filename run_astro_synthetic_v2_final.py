
"""
Astrocytic precision-phase coupling simulation — Version 2
Mechanistic synthetic ECoG-like regime analysis with environmental stress,
nonstationary astrocytic control fields, time-lagged Hermitian loop circulation,
transition-boundary instability, and adaptive stability/flexibility readouts.

Run:
    python run_astro_synthetic_v2.py

Outputs:
    outputs_v2/subject_level_metrics_v2.csv
    outputs_v2/regime_summary_v2.csv
    outputs_v2/*.png
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class SimConfig:
    seed: int = 20260618

    n_subjects_per_regime: int = 32
    n_windows: int = 180
    n_channels: int = 48
    n_bands: int = 4
    n_states: int = 3

    block_len: int = 15
    amplitude_matched: bool = True
    max_transition_search: int = 28

    output_dir: str = "outputs_v2"


REGIMES = {
    "balanced": {
        "pi_base": 0.55,
        "theta_base": 0.55,
        "chi_base": 0.05,
        "delay": 0,
        "recovery": 0.85,
        "same_state_bias": 0.55,
        "stress_sensitivity": 0.35,
        "boundary_fragility": 0.10,
        "reactive_noise": 0.05,
        "burst_prob": 0.015,
    },
    "hypo_support": {
        "pi_base": -0.60,
        "theta_base": -0.55,
        "chi_base": 0.42,
        "delay": 0,
        "recovery": 0.45,
        "same_state_bias": 0.35,
        "stress_sensitivity": 0.55,
        "boundary_fragility": 0.30,
        "reactive_noise": 0.12,
        "burst_prob": 0.025,
    },
    "over_stabilization": {
        "pi_base": 0.85,
        "theta_base": 0.80,
        "chi_base": 0.52,
        "delay": 0,
        "recovery": 0.70,
        "same_state_bias": 1.35,
        "stress_sensitivity": 0.25,
        "boundary_fragility": 0.15,
        "reactive_noise": 0.06,
        "burst_prob": 0.012,
    },
    "mistimed_support": {
        "pi_base": 0.55,
        "theta_base": 0.55,
        "chi_base": 0.18,
        "delay": 6,
        "recovery": 0.70,
        "same_state_bias": 0.55,
        "stress_sensitivity": 0.40,
        "boundary_fragility": 0.75,
        "reactive_noise": 0.08,
        "burst_prob": 0.018,
    },
    "reactive_noise": {
        "pi_base": 0.05,
        "theta_base": 0.00,
        "chi_base": 0.32,
        "delay": 2,
        "recovery": 0.38,
        "same_state_bias": 0.65,
        "stress_sensitivity": 0.95,
        "boundary_fragility": 0.55,
        "reactive_noise": 0.42,
        "burst_prob": 0.080,
    },
}


STATE_NAMES = {
    0: "task",
    1: "distractor",
    2: "low_demand",
}


def wrap_phase(x: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * x))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def make_cue_schedule(cfg: SimConfig) -> np.ndarray:
    """
    External task environment.
    0 = task
    1 = distractor
    2 = low-demand/rest
    """
    motif = [2, 0, 0, 1, 0, 2, 0, 1, 2, 0, 0, 2]
    cue = []
    while len(cue) < cfg.n_windows:
        for state in motif:
            cue.extend([state] * cfg.block_len)
            if len(cue) >= cfg.n_windows:
                break
    return np.asarray(cue[: cfg.n_windows], dtype=int)


def delayed(x: np.ndarray, d: int) -> np.ndarray:
    if d <= 0:
        return x.copy()
    y = np.zeros_like(x)
    y[d:] = x[:-d]
    y[:d] = x[0]
    return y


def cue_boundary_mask(cue: np.ndarray, width: int = 4) -> np.ndarray:
    mask = np.zeros_like(cue, dtype=bool)
    changes = np.where(cue[1:] != cue[:-1])[0] + 1
    for idx in changes:
        lo = idx
        hi = min(len(cue), idx + width)
        mask[lo:hi] = True
    return mask


def make_environmental_stress(
    cfg: SimConfig,
    rng: np.random.Generator,
    burst_prob: float,
) -> dict:
    """
    Slowly varying environmental stress with occasional bursts.
    This creates naturalistic nonstationarity rather than clean regime blocks.
    """
    n = cfg.n_windows
    slow = np.zeros(n)
    for t in range(1, n):
        slow[t] = 0.96 * slow[t - 1] + 0.055 * rng.normal()

    drift = 0.20 * np.sin(np.linspace(0, 3.5 * np.pi, n) + rng.uniform(0, 2 * np.pi))

    bursts = np.zeros(n)
    for t in range(1, n):
        if rng.random() < burst_prob:
            width = int(rng.integers(4, 14))
            amp = float(rng.uniform(0.45, 1.25) * rng.choice([-1, 1]))
            end = min(n, t + width)
            kernel = np.exp(-np.arange(end - t) / max(width / 2.0, 1.0))
            bursts[t:end] += amp * kernel

    stress = slow + drift + bursts
    stress = (stress - np.median(stress)) / (1.4826 * np.median(np.abs(stress - np.median(stress))) + 1e-8)
    stress = np.clip(stress, -2.5, 2.5)

    return {
        "slow": slow,
        "drift": drift,
        "bursts": bursts,
        "stress": stress,
    }


def simulate_astro_field(
    cfg: SimConfig,
    regime_params: dict,
    cue: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """
    Upstream astrocytic control variables:
    pi       = Fisher precision support
    theta    = Hermitian timing support
    chi      = Finsler transition-cost pressure
    q        = metabolic/ionic debt
    stress   = environmental nonstationarity
    """
    n = cfg.n_windows
    demand = (cue != 2).astype(float)
    boundary = cue_boundary_mask(cue, width=5).astype(float)
    demand_d = delayed(demand, regime_params["delay"])

    env = make_environmental_stress(cfg, rng, burst_prob=regime_params["burst_prob"])
    stress = env["stress"]

    q = np.zeros(n)
    for t in range(1, n):
        q[t] = (
            0.93 * q[t - 1]
            + 0.16 * demand[t]
            + 0.06 * max(stress[t], 0.0)
            - 0.11 * regime_params["recovery"] * (1.0 - demand[t])
        )
        q[t] = max(q[t], 0.0)

    ar = np.zeros((n, 3))
    for t in range(1, n):
        ar[t] = 0.80 * ar[t - 1] + regime_params["reactive_noise"] * rng.normal(size=3)

    sens = regime_params["stress_sensitivity"]
    frag = regime_params["boundary_fragility"]

    pi = (
        regime_params["pi_base"]
        + 0.25 * demand_d
        - 0.18 * q
        - 0.18 * sens * np.maximum(stress, 0.0)
        - 0.20 * frag * boundary
        + ar[:, 0]
    )

    theta = (
        regime_params["theta_base"]
        + 0.25 * demand_d
        - 0.15 * q
        - 0.22 * sens * np.maximum(stress, 0.0)
        - 0.35 * frag * boundary
        + ar[:, 1]
    )

    chi = (
        regime_params["chi_base"]
        + 0.42 * q
        + 0.18 * sens * np.maximum(stress, 0.0)
        + 0.20 * frag * boundary
        - 0.09 * regime_params["recovery"] * demand_d
        + ar[:, 2]
    )

    return {
        "pi": pi,
        "theta": theta,
        "chi": chi,
        "q": q,
        "demand": demand,
        "boundary": boundary.astype(bool),
        "stress": stress,
        "stress_slow": env["slow"],
        "stress_bursts": env["bursts"],
    }


def make_state_centers(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Moderate state separation to avoid ceiling decoding.
    """
    n_features = cfg.n_channels * cfg.n_bands
    raw = rng.normal(0, 1, size=(cfg.n_states, n_features))

    # Orthogonalize approximately.
    q, _ = np.linalg.qr(raw.T)
    centers = q[:, : cfg.n_states].T

    centers -= centers.mean(axis=1, keepdims=True)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8

    # Lower separation than v1.
    centers *= math.sqrt(n_features) * 0.62
    return centers


def simulate_latent_states(
    cfg: SimConfig,
    cue: np.ndarray,
    astro: dict,
    regime_params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    n = cfg.n_windows
    states = np.zeros(n, dtype=int)
    states[0] = cue[0]

    base_barrier = np.array([
        [0.00, 1.00, 0.75],
        [1.05, 0.00, 0.85],
        [1.20, 0.90, 0.00],
    ])

    direction_cost = np.array([
        [0.00, 0.20, 0.12],
        [0.35, 0.00, 0.18],
        [0.72, 0.28, 0.00],
    ])

    boundary = astro["boundary"].astype(float)

    for t in range(1, n):
        current = states[t - 1]
        target = cue[t]

        barrier = base_barrier[current] + astro["chi"][t] * direction_cost[current]
        logits = -barrier

        # Cue attraction depends on precision and timing support.
        support = 0.55 * astro["pi"][t] + 0.45 * astro["theta"][t]
        cue_strength = 1.05 + 0.28 * np.tanh(support)

        # At boundaries, mistiming/reactive fragility reduces immediate cue uptake.
        cue_strength -= regime_params["boundary_fragility"] * boundary[t] * 0.45

        logits[target] += cue_strength
        logits[current] += 0.60 + regime_params["same_state_bias"]

        # Fatigue biases toward low-demand basins.
        if astro["q"][t] > 0.85:
            logits[2] += 0.38 * astro["q"][t]

        # Low precision increases distractor capture.
        if astro["pi"][t] < -0.25:
            logits[1] += 0.25 * abs(astro["pi"][t])

        p = softmax(logits)
        states[t] = rng.choice(cfg.n_states, p=p)

    return states


def simulate_ecog_features(
    cfg: SimConfig,
    states: np.ndarray,
    cue: np.ndarray,
    astro: dict,
    centers: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    ECoG-like band features. Low precision causes:
    - higher observation dispersion
    - partial collapse of state centers toward the global center
    - low-rank shared noise under environmental stress
    """
    n_features = cfg.n_channels * cfg.n_bands
    n = cfg.n_windows

    global_center = centers.mean(axis=0)
    subject_loading = rng.normal(0, 0.18, size=n_features)

    low_rank_axes = rng.normal(0, 1, size=(3, n_features))
    low_rank_axes /= np.linalg.norm(low_rank_axes, axis=1, keepdims=True) + 1e-8

    X = np.zeros((n, n_features))

    for t in range(n):
        precision = astro["pi"][t]
        timing = astro["theta"][t]
        stress = max(astro["stress"][t], 0.0)

        blur = float(sigmoid(-precision)) * 0.45
        effective_center = (1.0 - blur) * centers[states[t]] + blur * global_center

        # Low precision and stress increase dispersion.
        noise_sd = 1.10 * np.exp(-0.35 * precision) + 0.10 * stress

        independent_noise = noise_sd * rng.normal(0, 1, size=n_features)

        # Shared low-rank components mimic environmental/common-mode stress.
        coeff = rng.normal(0, 1, size=3) * (0.18 + 0.10 * stress + 0.08 * max(-timing, 0.0))
        shared_noise = coeff @ low_rank_axes * math.sqrt(n_features)

        # Boundary disturbance under mistiming/reactive states.
        boundary_shock = 0.0
        if astro["boundary"][t]:
            boundary_shock = rng.normal(0, 0.25 + 0.15 * stress, size=n_features)

        X[t] = effective_center + subject_loading + independent_noise + shared_noise + boundary_shock

    if cfg.amplitude_matched:
        # Match overall RMS to reduce trivial amplitude confounds while preserving geometry.
        rms = np.sqrt(np.mean(X ** 2))
        X = X / (rms + 1e-8)

    return X


def make_phase_network(cfg: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = cfg.n_channels
    W = np.zeros((n, n))

    for i in range(n):
        W[i, (i - 1) % n] = 1.0
        W[i, (i + 1) % n] = 1.0
        W[i, (i + 4) % n] = 0.5

    random_edges = int(1.2 * n)
    for _ in range(random_edges):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            W[i, j] = max(W[i, j], 0.55)
            W[j, i] = max(W[j, i], 0.55)

    W = W / (W.sum(axis=1, keepdims=True) + 1e-8)

    # Directed preferred phase lags.
    phi = rng.uniform(-0.35, 0.35, size=(n, n))
    phi = phi - phi.T

    return W, phi


def simulate_phase_signals(
    cfg: SimConfig,
    astro: dict,
    W: np.ndarray,
    phi: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = cfg.n_windows
    c = cfg.n_channels

    phase = np.zeros((n, c))
    phase[0] = rng.uniform(-np.pi, np.pi, size=c)

    omega = rng.normal(0.20, 0.025, size=c)

    for t in range(1, n):
        previous = phase[t - 1]
        diff = previous[None, :] - previous[:, None] - phi

        coupling = np.sum(W * np.sin(diff), axis=1)

        support = astro["theta"][t]
        stress = max(astro["stress"][t], 0.0)
        boundary = 1.0 if astro["boundary"][t] else 0.0

        kappa = 0.12 * np.exp(0.35 * support)
        diffusion = 0.34 * np.exp(-0.55 * support) + 0.06 * stress + 0.12 * boundary * max(-support, 0.0)

        phase[t] = wrap_phase(previous + omega + kappa * coupling + diffusion * rng.normal(0, 1, size=c))

    # Synthetic high-frequency amplitude modulated by low-frequency phase.
    # The modulation is stronger when Hermitian timing support is preserved
    # and degrades under stress or low timing support.
    driver = 0
    offsets = rng.uniform(-np.pi, np.pi, size=c)
    pac_strength = 0.08 + 0.32 * sigmoid(astro["theta"])
    gamma_noise = 0.16 + 0.06 * np.maximum(astro["stress"], 0.0) + 0.10 * np.maximum(-astro["theta"], 0.0)

    gamma_amp = 1.0 + pac_strength[:, None] * np.sin(phase[:, [driver]] + offsets[None, :])
    gamma_amp += gamma_noise[:, None] * rng.normal(0, 1, size=(n, c))
    gamma_amp = np.clip(gamma_amp, 0.05, None)

    return phase, gamma_amp


def phase_metrics(
    phase: np.ndarray,
    gamma_amp: np.ndarray,
    phi: np.ndarray,
    cue: np.ndarray,
) -> dict:
    n, c = phase.shape
    boundary = cue_boundary_mask(cue, width=5)
    boundary_idx = np.where(boundary[1:])[0] + 1

    edges = [(i, (i + 1) % c) for i in range(c)]

    plv_values = []
    for u, v in edges:
        plv_values.append(np.abs(np.mean(np.exp(1j * (phase[:, u] - phase[:, v] - phi[u, v])))))
    mean_plv = float(np.mean(plv_values))

    slip_rates = []
    boundary_slips = []
    for u, v in edges:
        d = wrap_phase(phase[:, u] - phase[:, v] - phi[u, v])
        dd = np.abs(wrap_phase(np.diff(d)))
        slip = dd > 0.65
        slip_rates.append(np.mean(slip))
        if len(boundary_idx) > 0:
            valid = boundary_idx[boundary_idx - 1 < len(slip)]
            boundary_slips.append(np.mean(slip[valid - 1]) if len(valid) else np.nan)

    phase_slip_rate = float(np.nanmean(slip_rates))
    boundary_phase_slip_rate = float(np.nanmean(boundary_slips))

    cycles = []
    step = max(c // 16, 2)
    for start in range(0, c - 2, step):
        cycles.append((start, start + 1, start + 2))
    cycles = cycles[:16]

    loop_terms = []
    boundary_loop_terms = []

    # Time-lagged directed circulation avoids telescoping cancellation.
    for a, b, d in cycles:
        residual = wrap_phase(
            (phase[1:, b] - phase[:-1, a] - phi[a, b])
            + (phase[1:, d] - phase[:-1, b] - phi[b, d])
            + (phase[1:, a] - phase[:-1, d] - phi[d, a])
        )
        sq = residual ** 2
        loop_terms.append(np.mean(sq))

        if len(boundary_idx) > 0:
            valid = boundary_idx[boundary_idx - 1 < len(sq)]
            boundary_loop_terms.append(np.mean(sq[valid - 1]) if len(valid) else np.nan)

    loop_instability = float(np.nanmean(loop_terms))
    boundary_loop_instability = float(np.nanmean(boundary_loop_terms))

    pac_values = []
    n_bins = 18
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    driver_nodes = [0, 3, 7, 11]
    target_nodes = [12, 16, 20, 24]

    for u, v in zip(driver_nodes, target_nodes):
        ph = phase[:, u]
        amp = gamma_amp[:, v]
        binned = np.digitize(ph, bins) - 1
        mean_amp = np.zeros(n_bins)

        for k in range(n_bins):
            if np.any(binned == k):
                mean_amp[k] = amp[binned == k].mean()
            else:
                mean_amp[k] = 1e-8

        p = mean_amp / (mean_amp.sum() + 1e-8)
        uniform = np.ones(n_bins) / n_bins
        mi = np.sum(p * np.log((p + 1e-8) / uniform)) / np.log(n_bins)
        pac_values.append(mi)

    pac = float(np.mean(pac_values))

    return {
        "plv": mean_plv,
        "pac": pac,
        "phase_slip_rate": phase_slip_rate,
        "boundary_phase_slip_rate": boundary_phase_slip_rate,
        "loop_instability": loop_instability,
        "boundary_loop_instability": boundary_loop_instability,
    }


def run_decoding_metrics(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """
    Fast cross-validated centroid decoder with a fixed random projection.
    The projection is label-independent and declared by seed. This keeps the
    synthetic analysis fast while avoiding regime-label leakage.
    """
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2 or counts.min() < 8:
        return {
            "decode_acc": np.nan,
            "balanced_decode_acc": np.nan,
            "posterior_entropy": np.nan,
            "silhouette": np.nan,
        }

    rng = np.random.default_rng(seed)
    n_comp = min(14, X.shape[1])
    R = rng.normal(0, 1, size=(X.shape[1], n_comp))
    R /= np.linalg.norm(R, axis=0, keepdims=True) + 1e-8

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros_like(y)
    proba = np.zeros((len(y), len(labels)), dtype=float)

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])

        Ztr = Xtr @ R
        Zte = Xte @ R

        pooled_var = np.var(Ztr, axis=0) + 1e-6
        centroids = []
        priors = []

        for lab in labels:
            z_lab = Ztr[y[train_idx] == lab]
            centroids.append(z_lab.mean(axis=0))
            priors.append(len(z_lab) / len(Ztr))

        centroids = np.vstack(centroids)
        priors = np.asarray(priors)

        scores = []
        for c_idx in range(len(labels)):
            diff = Zte - centroids[c_idx]
            score = -0.5 * np.sum((diff ** 2) / pooled_var, axis=1) + np.log(priors[c_idx] + 1e-12)
            scores.append(score)

        scores = np.vstack(scores).T
        fold_proba = softmax(scores, axis=1)
        proba[test_idx] = fold_proba
        pred[test_idx] = labels[np.argmax(fold_proba, axis=1)]

    acc = accuracy_score(y, pred)
    bacc = balanced_accuracy_score(y, pred)

    entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
    entropy = entropy / np.log(proba.shape[1])

    Z = StandardScaler().fit_transform(X) @ R
    try:
        sil = silhouette_score(Z, y)
    except Exception:
        sil = np.nan

    return {
        "decode_acc": float(acc),
        "balanced_decode_acc": float(bacc),
        "posterior_entropy": float(np.mean(entropy)),
        "silhouette": float(sil),
    }


def transition_metrics(states: np.ndarray, cue: np.ndarray, cfg: SimConfig) -> dict:
    changes = np.where(cue[1:] != cue[:-1])[0] + 1
    latencies = []
    rest_to_task = []
    task_to_rest = []
    failures = []
    perseveration = []

    max_search = cfg.max_transition_search

    for idx in changes:
        target = cue[idx]
        prev = cue[idx - 1]

        latency = max_search
        reached = False

        for t in range(idx, min(len(states), idx + max_search)):
            if states[t] == target:
                latency = t - idx
                reached = True
                break

        latencies.append(latency)
        failures.append(0 if reached else 1)

        # How long the previous state persists after a cue switch.
        p_len = 0
        for t in range(idx, min(len(states), idx + max_search)):
            if states[t] == prev:
                p_len += 1
            else:
                break
        perseveration.append(p_len)

        if prev == 2 and target == 0:
            rest_to_task.append(latency)
        if prev == 0 and target == 2:
            task_to_rest.append(latency)

    def mean_or_nan(x):
        return float(np.mean(x)) if len(x) else np.nan

    dwell_task = []
    current_len = 0
    for s in states:
        if s == 0:
            current_len += 1
        else:
            if current_len > 0:
                dwell_task.append(current_len)
            current_len = 0
    if current_len > 0:
        dwell_task.append(current_len)

    hysteresis = mean_or_nan(rest_to_task) - mean_or_nan(task_to_rest)

    return {
        "transition_latency": mean_or_nan(latencies),
        "rest_to_task_latency": mean_or_nan(rest_to_task),
        "task_to_rest_latency": mean_or_nan(task_to_rest),
        "hysteresis": float(hysteresis) if not np.isnan(hysteresis) else np.nan,
        "task_dwell": mean_or_nan(dwell_task),
        "update_failure_rate": mean_or_nan(failures),
        "perseveration_after_switch": mean_or_nan(perseveration),
    }


def simulate_subject(
    cfg: SimConfig,
    regime_name: str,
    regime_params: dict,
    subject_index: int,
    centers: np.ndarray,
    W: np.ndarray,
    phi: np.ndarray,
) -> tuple[dict, dict | None]:
    regime_order = list(REGIMES.keys())
    rng = np.random.default_rng(cfg.seed + 10000 * regime_order.index(regime_name) + subject_index)

    cue = make_cue_schedule(cfg)
    astro = simulate_astro_field(cfg, regime_params, cue, rng)
    states = simulate_latent_states(cfg, cue, astro, regime_params, rng)
    X = simulate_ecog_features(cfg, states, cue, astro, centers, rng)
    phase, gamma_amp = simulate_phase_signals(cfg, astro, W, phi, rng)

    dec = run_decoding_metrics(X, states, seed=cfg.seed + subject_index)
    ph = phase_metrics(phase, gamma_amp, phi, cue)
    tr = transition_metrics(states, cue, cfg)

    row = {
        "regime": regime_name,
        "subject": subject_index,
        "mean_pi": float(np.mean(astro["pi"])),
        "mean_theta": float(np.mean(astro["theta"])),
        "mean_chi": float(np.mean(astro["chi"])),
        "mean_debt_q": float(np.mean(astro["q"])),
        "mean_stress": float(np.mean(astro["stress"])),
        "stress_sd": float(np.std(astro["stress"])),
    }
    row.update(dec)
    row.update(ph)
    row.update(tr)

    example = None
    if subject_index == 0:
        example = {
            "regime": regime_name,
            "X": X,
            "states": states,
            "cue": cue,
            "astro": astro,
            "phase": phase,
        }

    return row, example


def add_integrated_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hysteresis_abs"] = out["hysteresis"].abs()

    positive_stability = [
        "balanced_decode_acc",
        "silhouette",
        "plv",
        "pac",
    ]

    instability = [
        "posterior_entropy",
        "phase_slip_rate",
        "boundary_phase_slip_rate",
        "loop_instability",
        "boundary_loop_instability",
    ]

    transition_burden = [
        "transition_latency",
        "update_failure_rate",
        "perseveration_after_switch",
        "hysteresis_abs",
    ]

    for col in positive_stability + instability + transition_burden:
        mu = out[col].mean()
        sd = out[col].std(ddof=0) + 1e-8
        out[f"z_{col}"] = (out[col] - mu) / sd

    out["stability_support_index"] = out[[f"z_{c}" for c in positive_stability]].mean(axis=1)
    out["instability_index"] = out[[f"z_{c}" for c in instability]].mean(axis=1)
    out["transition_burden_index"] = out[[f"z_{c}" for c in transition_burden]].mean(axis=1)

    out["precision_phase_margin"] = out["stability_support_index"] - out["instability_index"]
    out["adaptive_margin"] = out["precision_phase_margin"] - 1.00 * out["transition_burden_index"]
    out["flexibility_index"] = -out["transition_burden_index"]

    crit = out.loc[out["regime"] == "balanced", "adaptive_margin"].quantile(0.20)
    out["above_adaptive_crit"] = out["adaptive_margin"] > crit

    return out


def save_summary(df: pd.DataFrame, outdir: Path) -> None:
    df.to_csv(outdir / "subject_level_metrics_v2.csv", index=False)

    summary = df.groupby("regime").agg(["mean", "std", "sem"])
    summary.to_csv(outdir / "regime_summary_v2.csv")

    key_cols = [
        "balanced_decode_acc",
        "posterior_entropy",
        "plv",
        "pac",
        "phase_slip_rate",
        "boundary_phase_slip_rate",
        "loop_instability",
        "boundary_loop_instability",
        "transition_latency",
        "update_failure_rate",
        "perseveration_after_switch",
        "hysteresis_abs",
        "precision_phase_margin",
        "adaptive_margin",
        "flexibility_index",
    ]

    compact = df.groupby("regime")[key_cols].mean().loc[list(REGIMES.keys())]
    compact.to_csv(outdir / "compact_regime_means_v2.csv")


def plot_fingerprint_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    metrics = [
        "balanced_decode_acc",
        "posterior_entropy",
        "plv",
        "pac",
        "phase_slip_rate",
        "boundary_phase_slip_rate",
        "loop_instability",
        "boundary_loop_instability",
        "transition_latency",
        "update_failure_rate",
        "perseveration_after_switch",
        "adaptive_margin",
        "flexibility_index",
    ]

    d = df.copy()
    for c in metrics:
        d[c] = (d[c] - d[c].mean()) / (d[c].std(ddof=0) + 1e-8)

    table = d.groupby("regime")[metrics].mean().loc[list(REGIMES.keys())]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    im = ax.imshow(table.values, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_title("Astrocytic regime fingerprints across extracted neural-state metrics")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Z-scored group mean")
    fig.tight_layout()
    fig.savefig(outdir / "fig_v2_regime_metric_fingerprints.png", dpi=300)
    plt.close(fig)


def plot_margin_and_flexibility(df: pd.DataFrame, outdir: Path) -> None:
    order = list(REGIMES.keys())
    summary = df.groupby("regime")[["precision_phase_margin", "adaptive_margin", "flexibility_index"]].agg(["mean", "sem"]).loc[order]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(order))
    width = 0.27

    ax.bar(x - width, summary[("precision_phase_margin", "mean")], width, yerr=summary[("precision_phase_margin", "sem")], capsize=3, label="Precision-phase margin")
    ax.bar(x, summary[("adaptive_margin", "mean")], width, yerr=summary[("adaptive_margin", "sem")], capsize=3, label="Adaptive margin")
    ax.bar(x + width, summary[("flexibility_index", "mean")], width, yerr=summary[("flexibility_index", "sem")], capsize=3, label="Flexibility index")

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Index value")
    ax.set_title("Stability and flexibility signatures across astrocytic regimes")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_v2_margin_flexibility_by_regime.png", dpi=300)
    plt.close(fig)


def plot_state_space_panels(examples: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(17, 3.8), sharex=False, sharey=False)

    for ax, regime in zip(axes, REGIMES.keys()):
        ex = examples[regime]
        Xp = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(ex["X"]))
        states = ex["states"]

        for s in range(3):
            idx = states == s
            ax.scatter(Xp[idx, 0], Xp[idx, 1], s=12, alpha=0.60, label=STATE_NAMES[s])

        ax.set_title(regime)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    axes[-1].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Regime-specific ECoG-like state-space structure", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig_v2_state_space_panels.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_control_traces(examples: dict, outdir: Path) -> None:
    selected = ["balanced", "hypo_support", "mistimed_support", "reactive_noise"]
    fig, axes = plt.subplots(len(selected), 1, figsize=(11, 8.5), sharex=True)

    for ax, regime in zip(axes, selected):
        astro = examples[regime]["astro"]
        cue = examples[regime]["cue"]

        ax.plot(astro["pi"], label="pi: Fisher precision support", linewidth=1.4)
        ax.plot(astro["theta"], label="theta: Hermitian timing support", linewidth=1.4)
        ax.plot(astro["chi"], label="chi: Finsler cost pressure", linewidth=1.4)
        ax.plot(0.35 * astro["stress"], label="scaled environmental stress", linewidth=1.0, alpha=0.65)

        changes = np.where(cue[1:] != cue[:-1])[0] + 1
        for c in changes:
            ax.axvline(c, linewidth=0.5, alpha=0.18)

        ax.set_title(regime)
        ax.set_ylabel("Control value")
        ax.legend(frameon=False, fontsize=8, ncol=4)

    axes[-1].set_xlabel("Window")
    fig.tight_layout()
    fig.savefig(outdir / "fig_v2_upstream_control_traces.png", dpi=300)
    plt.close(fig)


def plot_transition_instability(df: pd.DataFrame, outdir: Path) -> None:
    order = list(REGIMES.keys())
    metrics = ["boundary_phase_slip_rate", "boundary_loop_instability", "transition_latency", "update_failure_rate"]

    table = df.groupby("regime")[metrics].mean().loc[order]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))

    for ax, metric in zip(axes, metrics):
        vals = table[metric].values
        sem = df.groupby("regime")[metric].sem().loc[order].values
        ax.bar(order, vals, yerr=sem, capsize=3)
        ax.set_title(metric)
        ax.set_xticklabels(order, rotation=45, ha="right")

    fig.suptitle("Transition-boundary instability and switching burden", y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "fig_v2_transition_boundary_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = SimConfig()
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "simulation_config_v2.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    rng_global = np.random.default_rng(cfg.seed)
    centers = make_state_centers(cfg, rng_global)
    W, phi = make_phase_network(cfg, rng_global)

    rows = []
    examples = {}

    for regime_name, params in REGIMES.items():
        for subject_index in tqdm(range(cfg.n_subjects_per_regime), desc=f"Simulating {regime_name}"):
            row, example = simulate_subject(
                cfg=cfg,
                regime_name=regime_name,
                regime_params=params,
                subject_index=subject_index,
                centers=centers,
                W=W,
                phi=phi,
            )
            rows.append(row)
            if example is not None:
                examples[regime_name] = example

    df = pd.DataFrame(rows)
    df = add_integrated_indices(df)

    save_summary(df, outdir)

    plot_fingerprint_heatmap(df, outdir)
    plot_margin_and_flexibility(df, outdir)
    plot_state_space_panels(examples, outdir)
    plot_control_traces(examples, outdir)
    plot_transition_instability(df, outdir)

    print("\nDone.")
    print(f"Outputs saved to: {outdir.resolve()}")
    print("\nKey files:")
    print(" - subject_level_metrics_v2.csv")
    print(" - regime_summary_v2.csv")
    print(" - compact_regime_means_v2.csv")
    print(" - fig_v2_regime_metric_fingerprints.png")
    print(" - fig_v2_margin_flexibility_by_regime.png")
    print(" - fig_v2_state_space_panels.png")
    print(" - fig_v2_upstream_control_traces.png")
    print(" - fig_v2_transition_boundary_metrics.png")


if __name__ == "__main__":
    main()
