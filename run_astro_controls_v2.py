
"""
Astrocytic precision-phase coupling controls — Version 2

This script completes the synthetic regime analysis by adding:
1. state-label shuffle control
2. phase-scramble control
3. regime-label shuffle control
4. Fisher/Hermitian/Finsler/integrated ablation
5. bootstrap confidence intervals
6. seed-level robustness summary

Run from the same folder that contains:
    run_astro_synthetic_v2_final.py

PowerShell:
    python run_astro_controls_v2.py

Outputs are saved to:
    outputs_v2_controls/
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from tqdm import tqdm

import run_astro_synthetic_v2_final as sim


warnings.filterwarnings("ignore")


CONTROL_DIR = Path("outputs_v2_controls")
CONTROL_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_GROUPS = {
    "fisher_only": [
        "balanced_decode_acc",
        "posterior_entropy",
        "silhouette",
    ],
    "hermitian_only": [
        "plv",
        "pac",
        "phase_slip_rate",
        "boundary_phase_slip_rate",
        "loop_instability",
        "boundary_loop_instability",
    ],
    "finsler_only": [
        "transition_latency",
        "update_failure_rate",
        "perseveration_after_switch",
        "hysteresis_abs",
        "flexibility_index",
    ],
    "integrated_only": [
        "precision_phase_margin",
        "adaptive_margin",
        "flexibility_index",
    ],
    "full_metric_set": [
        "balanced_decode_acc",
        "posterior_entropy",
        "silhouette",
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
    ],
}


KEY_BOOTSTRAP_METRICS = [
    "balanced_decode_acc",
    "posterior_entropy",
    "plv",
    "phase_slip_rate",
    "boundary_phase_slip_rate",
    "loop_instability",
    "boundary_loop_instability",
    "transition_latency",
    "update_failure_rate",
    "perseveration_after_switch",
    "precision_phase_margin",
    "adaptive_margin",
    "flexibility_index",
]


def make_global_objects(cfg: sim.SimConfig):
    rng_global = np.random.default_rng(cfg.seed)
    centers = sim.make_state_centers(cfg, rng_global)
    W, phi = sim.make_phase_network(cfg, rng_global)
    return centers, W, phi


def generate_raw_subject(
    cfg: sim.SimConfig,
    regime_name: str,
    subject_index: int,
    centers: np.ndarray,
    W: np.ndarray,
    phi: np.ndarray,
):
    params = sim.REGIMES[regime_name]
    regime_order = list(sim.REGIMES.keys())
    rng = np.random.default_rng(cfg.seed + 10000 * regime_order.index(regime_name) + subject_index)

    cue = sim.make_cue_schedule(cfg)
    astro = sim.simulate_astro_field(cfg, params, cue, rng)
    states = sim.simulate_latent_states(cfg, cue, astro, params, rng)
    X = sim.simulate_ecog_features(cfg, states, cue, astro, centers, rng)
    phase, gamma_amp = sim.simulate_phase_signals(cfg, astro, W, phi, rng)

    return {
        "cue": cue,
        "astro": astro,
        "states": states,
        "X": X,
        "phase": phase,
        "gamma_amp": gamma_amp,
        "phi": phi,
    }


def circular_shift_phase_by_channel(phase: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Preserves each channel's phase distribution and temporal autocorrelation
    while disrupting cross-channel alignment and directed loop timing.
    """
    out = np.empty_like(phase)
    n, c = phase.shape
    for ch in range(c):
        shift = int(rng.integers(1, n - 1))
        out[:, ch] = np.roll(phase[:, ch], shift)
    return out


def run_state_label_and_phase_controls(cfg: sim.SimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    centers, W, phi = make_global_objects(cfg)

    state_rows = []
    phase_rows = []

    for regime_name in sim.REGIMES.keys():
        for subject_index in tqdm(range(cfg.n_subjects_per_regime), desc=f"Controls {regime_name}"):
            raw = generate_raw_subject(cfg, regime_name, subject_index, centers, W, phi)

            rng = np.random.default_rng(cfg.seed + 500000 + 10000 * list(sim.REGIMES.keys()).index(regime_name) + subject_index)

            full_dec = sim.run_decoding_metrics(raw["X"], raw["states"], seed=cfg.seed + subject_index)
            shuffled_states = rng.permutation(raw["states"])
            shuffle_dec = sim.run_decoding_metrics(raw["X"], shuffled_states, seed=cfg.seed + subject_index + 991)

            state_rows.append({
                "regime": regime_name,
                "subject": subject_index,
                "full_decode_acc": full_dec["balanced_decode_acc"],
                "shuffle_decode_acc": shuffle_dec["balanced_decode_acc"],
                "full_entropy": full_dec["posterior_entropy"],
                "shuffle_entropy": shuffle_dec["posterior_entropy"],
                "full_silhouette": full_dec["silhouette"],
                "shuffle_silhouette": shuffle_dec["silhouette"],
            })

            full_phase = sim.phase_metrics(raw["phase"], raw["gamma_amp"], raw["phi"], raw["cue"])
            phase_scrambled = circular_shift_phase_by_channel(raw["phase"], rng)
            scrambled_phase = sim.phase_metrics(phase_scrambled, raw["gamma_amp"], raw["phi"], raw["cue"])

            row = {
                "regime": regime_name,
                "subject": subject_index,
            }

            for key in [
                "plv",
                "pac",
                "phase_slip_rate",
                "boundary_phase_slip_rate",
                "loop_instability",
                "boundary_loop_instability",
            ]:
                row[f"full_{key}"] = full_phase[key]
                row[f"scrambled_{key}"] = scrambled_phase[key]
                row[f"delta_{key}"] = full_phase[key] - scrambled_phase[key]

            phase_rows.append(row)

    state_df = pd.DataFrame(state_rows)
    phase_df = pd.DataFrame(phase_rows)

    state_df.to_csv(CONTROL_DIR / "state_label_shuffle_control.csv", index=False)
    phase_df.to_csv(CONTROL_DIR / "phase_scramble_control.csv", index=False)

    return state_df, phase_df


def cv_centroid_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    n_splits: int = 5,
) -> float:
    labels = np.unique(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    pred = np.empty_like(y, dtype=object)

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])

        centroids = []
        priors = []
        pooled_var = np.var(Xtr, axis=0) + 1e-6

        for lab in labels:
            z = Xtr[y[train_idx] == lab]
            centroids.append(z.mean(axis=0))
            priors.append(len(z) / len(Xtr))

        centroids = np.vstack(centroids)
        priors = np.asarray(priors)

        scores = []
        for i, lab in enumerate(labels):
            diff = Xte - centroids[i]
            score = -0.5 * np.sum((diff ** 2) / pooled_var, axis=1) + np.log(priors[i] + 1e-12)
            scores.append(score)

        scores = np.vstack(scores).T
        pred[test_idx] = labels[np.argmax(scores, axis=1)]

    return float(np.mean(pred == y))


def permutation_accuracy_pvalue(
    X: np.ndarray,
    y: np.ndarray,
    observed_acc: float,
    seed: int = 0,
    n_perm: int = 300,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        null[i] = cv_centroid_accuracy(X, y_perm, seed=seed + i + 101)
    p = (1 + np.sum(null >= observed_acc)) / (n_perm + 1)
    return float(null.mean()), float(np.percentile(null, 95)), float(p)


def prepare_subject_metrics() -> pd.DataFrame:
    """
    Uses the existing main simulation output if available.
    Otherwise regenerates it using the imported main simulation functions.
    """
    candidate = Path("outputs_v2") / "subject_level_metrics_v2.csv"
    if candidate.exists():
        df = pd.read_csv(candidate)
    else:
        cfg = sim.SimConfig()
        centers, W, phi = make_global_objects(cfg)
        rows = []
        for regime_name, params in sim.REGIMES.items():
            for subject_index in range(cfg.n_subjects_per_regime):
                row, _ = sim.simulate_subject(cfg, regime_name, params, subject_index, centers, W, phi)
                rows.append(row)
        df = sim.add_integrated_indices(pd.DataFrame(rows))

    if "hysteresis_abs" not in df.columns:
        df["hysteresis_abs"] = df["hysteresis"].abs()

    return df


def run_ablation_and_regime_shuffle(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = df["regime"].values.astype(object)

    for group_name, cols in FEATURE_GROUPS.items():
        available_cols = [c for c in cols if c in df.columns]
        X = df[available_cols].values.astype(float)

        observed_acc = cv_centroid_accuracy(X, y, seed=2026)
        null_mean, null_95, p_perm = permutation_accuracy_pvalue(
            X, y, observed_acc=observed_acc, seed=3000 + len(rows), n_perm=300
        )

        rows.append({
            "feature_group": group_name,
            "n_features": len(available_cols),
            "observed_regime_accuracy": observed_acc,
            "chance_level": 1.0 / len(np.unique(y)),
            "permutation_null_mean": null_mean,
            "permutation_null_95th": null_95,
            "permutation_p_value": p_perm,
        })

    out = pd.DataFrame(rows)
    out.to_csv(CONTROL_DIR / "ablation_regime_classification.csv", index=False)
    return out


def bootstrap_ci_by_regime(
    df: pd.DataFrame,
    metrics: list[str],
    n_boot: int = 1000,
    seed: int = 20260618,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for regime in sim.REGIMES.keys():
        sub = df[df["regime"] == regime]
        n = len(sub)
        for metric in metrics:
            if metric not in sub.columns:
                continue

            values = sub[metric].dropna().values.astype(float)
            if len(values) < 3:
                continue

            boot = np.zeros(n_boot)
            for b in range(n_boot):
                sample = rng.choice(values, size=len(values), replace=True)
                boot[b] = np.mean(sample)

            rows.append({
                "regime": regime,
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "n": int(n),
            })

    out = pd.DataFrame(rows)
    out.to_csv(CONTROL_DIR / "bootstrap_ci_by_regime.csv", index=False)
    return out


def run_seed_robustness(
    base_seed: int = 20260618,
    n_seeds: int = 8,
    n_subjects_per_regime: int = 8,
) -> pd.DataFrame:
    rows = []

    for s in tqdm(range(n_seeds), desc="Seed robustness"):
        cfg = sim.SimConfig(
            seed=base_seed + s,
            n_subjects_per_regime=n_subjects_per_regime,
            output_dir="unused_seed_robustness",
        )

        centers, W, phi = make_global_objects(cfg)
        run_rows = []

        for regime_name, params in sim.REGIMES.items():
            for subject_index in range(cfg.n_subjects_per_regime):
                row, _ = sim.simulate_subject(cfg, regime_name, params, subject_index, centers, W, phi)
                run_rows.append(row)

        run_df = sim.add_integrated_indices(pd.DataFrame(run_rows))
        run_df["hysteresis_abs"] = run_df["hysteresis"].abs()

        grouped = run_df.groupby("regime")[[
            "precision_phase_margin",
            "adaptive_margin",
            "flexibility_index",
            "balanced_decode_acc",
            "phase_slip_rate",
            "transition_latency",
        ]].mean()

        for regime in sim.REGIMES.keys():
            rows.append({
                "seed": base_seed + s,
                "regime": regime,
                "precision_phase_margin": float(grouped.loc[regime, "precision_phase_margin"]),
                "adaptive_margin": float(grouped.loc[regime, "adaptive_margin"]),
                "flexibility_index": float(grouped.loc[regime, "flexibility_index"]),
                "balanced_decode_acc": float(grouped.loc[regime, "balanced_decode_acc"]),
                "phase_slip_rate": float(grouped.loc[regime, "phase_slip_rate"]),
                "transition_latency": float(grouped.loc[regime, "transition_latency"]),
            })

    out = pd.DataFrame(rows)
    out.to_csv(CONTROL_DIR / "seed_robustness_summary.csv", index=False)
    return out


def plot_state_label_control(state_df: pd.DataFrame) -> None:
    summary = state_df.groupby("regime")[["full_decode_acc", "shuffle_decode_acc"]].agg(["mean", "sem"]).loc[list(sim.REGIMES.keys())]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(sim.REGIMES))
    width = 0.35

    ax.bar(x - width / 2, summary[("full_decode_acc", "mean")], width, yerr=summary[("full_decode_acc", "sem")], capsize=3, label="Full labels")
    ax.bar(x + width / 2, summary[("shuffle_decode_acc", "mean")], width, yerr=summary[("shuffle_decode_acc", "sem")], capsize=3, label="Shuffled state labels")
    ax.axhline(1 / 3, linewidth=1, linestyle="--", label="Three-state chance")

    ax.set_xticks(x)
    ax.set_xticklabels(list(sim.REGIMES.keys()), rotation=30, ha="right")
    ax.set_ylabel("Cross-validated state decoding accuracy")
    ax.set_title("State-label shuffle control")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CONTROL_DIR / "fig_control_state_label_shuffle.png", dpi=300)
    plt.close(fig)


def plot_phase_scramble_control(phase_df: pd.DataFrame) -> None:
    metrics = [
        "plv",
        "pac",
        "phase_slip_rate",
        "boundary_phase_slip_rate",
        "loop_instability",
        "boundary_loop_instability",
    ]

    rows = []
    for metric in metrics:
        rows.append({
            "metric": metric,
            "full_mean": phase_df[f"full_{metric}"].mean(),
            "full_sem": phase_df[f"full_{metric}"].sem(),
            "scrambled_mean": phase_df[f"scrambled_{metric}"].mean(),
            "scrambled_sem": phase_df[f"scrambled_{metric}"].sem(),
        })
    table = pd.DataFrame(rows)
    table.to_csv(CONTROL_DIR / "phase_scramble_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, table["full_mean"], width, yerr=table["full_sem"], capsize=3, label="Full phase")
    ax.bar(x + width / 2, table["scrambled_mean"], width, yerr=table["scrambled_sem"], capsize=3, label="Channel-shifted phase")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Phase-scramble control")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CONTROL_DIR / "fig_control_phase_scramble.png", dpi=300)
    plt.close(fig)


def plot_ablation(ablation_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))

    x = np.arange(len(ablation_df))
    ax.bar(x, ablation_df["observed_regime_accuracy"], label="Observed")
    ax.scatter(x, ablation_df["permutation_null_mean"], marker="o", label="Permutation null mean")
    ax.scatter(x, ablation_df["permutation_null_95th"], marker="x", label="Permutation null 95th")
    ax.axhline(0.20, linestyle="--", linewidth=1, label="Five-regime chance")

    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df["feature_group"], rotation=30, ha="right")
    ax.set_ylabel("Regime classification accuracy")
    ax.set_title("Layer ablation and regime-label shuffle control")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(CONTROL_DIR / "fig_ablation_regime_classification.png", dpi=300)
    plt.close(fig)


def plot_seed_robustness(seed_df: pd.DataFrame) -> None:
    metrics = ["adaptive_margin", "precision_phase_margin", "flexibility_index"]
    summary = seed_df.groupby("regime")[metrics].agg(["mean", "sem"]).loc[list(sim.REGIMES.keys())]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for ax, metric in zip(axes, metrics):
        vals = summary[(metric, "mean")]
        sems = summary[(metric, "sem")]
        x = np.arange(len(vals))
        ax.bar(x, vals, yerr=sems, capsize=3)
        ax.axhline(0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(vals.index, rotation=35, ha="right")
        ax.set_title(metric)

    fig.suptitle("Seed-level robustness of regime ordering", y=1.03)
    fig.tight_layout()
    fig.savefig(CONTROL_DIR / "fig_seed_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = sim.SimConfig()

    with open(CONTROL_DIR / "controls_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_config": cfg.__dict__,
            "controls": [
                "state_label_shuffle",
                "phase_scramble_by_independent_channel_circular_shift",
                "regime_label_shuffle_permutation",
                "layer_ablation",
                "bootstrap_confidence_intervals",
                "seed_robustness",
            ],
            "n_permutations_regime_shuffle": 300,
            "n_bootstrap": 1000,
            "n_seed_robustness_runs": 8,
            "n_subjects_per_regime_per_seed": 8,
        }, f, indent=2)

    print("\n[1/6] Running state-label shuffle and phase-scramble controls...")
    state_df, phase_df = run_state_label_and_phase_controls(cfg)

    print("\n[2/6] Loading main subject-level metrics...")
    main_df = prepare_subject_metrics()

    print("\n[3/6] Running ablation and regime-label shuffle control...")
    ablation_df = run_ablation_and_regime_shuffle(main_df)

    print("\n[4/6] Computing bootstrap confidence intervals...")
    boot_df = bootstrap_ci_by_regime(main_df, KEY_BOOTSTRAP_METRICS, n_boot=1000)

    print("\n[5/6] Running seed-level robustness...")
    seed_df = run_seed_robustness(n_seeds=8, n_subjects_per_regime=8)

    print("\n[6/6] Saving figures...")
    plot_state_label_control(state_df)
    plot_phase_scramble_control(phase_df)
    plot_ablation(ablation_df)
    plot_seed_robustness(seed_df)

    print("\nDone.")
    print(f"Control outputs saved to: {CONTROL_DIR.resolve()}")
    print("\nKey files:")
    print(" - state_label_shuffle_control.csv")
    print(" - phase_scramble_control.csv")
    print(" - phase_scramble_summary.csv")
    print(" - ablation_regime_classification.csv")
    print(" - bootstrap_ci_by_regime.csv")
    print(" - seed_robustness_summary.csv")
    print(" - fig_control_state_label_shuffle.png")
    print(" - fig_control_phase_scramble.png")
    print(" - fig_ablation_regime_classification.png")
    print(" - fig_seed_robustness.png")


if __name__ == "__main__":
    main()
