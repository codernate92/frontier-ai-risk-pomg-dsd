"""
Generate vector figures for the POMG-DSD experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv"
FIG_DIR = ROOT / "paper_figures"
FIG_DIR.mkdir(exist_ok=True)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0:
        return series * 0.0
    return (series - series.mean()) / std


def plot_calibration_overlay() -> None:
    traj = pd.read_csv(CSV_DIR / "pomg_calibrated_trajectory.csv")
    with (ROOT / "known_exploited_vulnerabilities.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    from datetime import datetime, timedelta

    rows = payload["vulnerabilities"]
    fmt = "%Y-%m-%d"
    added = [datetime.strptime(row["dateAdded"], fmt) for row in rows]
    due = [datetime.strptime(row["dueDate"], fmt) for row in rows]
    end = max(added)
    start = end - timedelta(weeks=52)
    weekly_additions = []
    weekly_backlog = []
    for i in range(53):
        t = start + timedelta(weeks=i)
        weekly_additions.append(sum(1 for a in added if t <= a < t + timedelta(weeks=1)))
        weekly_backlog.append(sum(1 for a, d in zip(added, due) if a <= t < d))
    anchor = pd.DataFrame(
        {
            "week": range(53),
            "weekly_additions": weekly_additions,
            "weekly_backlog": weekly_backlog,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].plot(anchor["week"], _zscore(anchor["weekly_additions"]), color="#1f77b4", lw=2, label="CISA KEV weekly additions")
    axes[0].plot(traj["week"], _zscore(traj["vuln_discovery_rate"]), color="#d62728", lw=2, ls="--", label="Model vulnerability discovery")
    axes[0].set_title("KEV Anchor Fit: Discovery Flow")
    axes[0].set_xlabel("Week")
    axes[0].set_ylabel("z-score")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(anchor["week"], _zscore(anchor["weekly_backlog"]), color="#2ca02c", lw=2, label="CISA KEV remediation backlog")
    axes[1].plot(traj["week"], _zscore(traj["vuln_backlog"]), color="#9467bd", lw=2, ls="--", label="Model vulnerability backlog")
    axes[1].set_title("KEV Anchor Fit: Backlog Stock")
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("z-score")
    axes[1].legend(frameon=False, fontsize=8)

    fig.savefig(FIG_DIR / "pomg_calibration_overlay.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training_curves() -> None:
    std = pd.read_csv(CSV_DIR / "pomg_standard_training.csv")
    prop = pd.read_csv(CSV_DIR / "pomg_proposed_training.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    axes[0].plot(std["iteration"], std["defender_return"], color="#4c78a8", lw=2, label="Standard PPO")
    axes[0].plot(prop["iteration"], prop["defender_return"], color="#f58518", lw=2, label="POMG-DSD")
    axes[0].set_title("Defender Return During Self-Play")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean episodic defender return")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(std["iteration"], std["min_cbf_margin"], color="#4c78a8", lw=2, label="Standard PPO")
    axes[1].plot(prop["iteration"], prop["min_cbf_margin"], color="#f58518", lw=2, label="POMG-DSD")
    axes[1].set_title("Minimum Safety Margin During Training")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Mean minimum CBF margin")
    axes[1].legend(frameon=False, fontsize=8)

    fig.savefig(FIG_DIR / "pomg_training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison() -> None:
    comp = pd.read_csv(CSV_DIR / "pomg_method_comparison.csv")
    methods = comp["method"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)
    x = range(len(methods))
    colors = ["#7f7f7f", "#4c78a8", "#f58518"]

    axes[0].bar(x, comp["peak_cbrn_mean"], yerr=comp["peak_cbrn_sd"], color=colors, capsize=4)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(["Heuristic", "Std. PPO", "POMG-DSD"], rotation=0)
    axes[0].set_title("Peak CBRN Misuse Under Reference Adversary")
    axes[0].set_ylabel("Normalized peak misuse")

    alloc_cols = ["alloc_cyber", "alloc_cbrn", "alloc_epistemic", "alloc_governance", "alloc_autonomy"]
    alloc_labels = ["Cyber", "CBRN", "Epistemic", "Gov.", "Autonomy"]
    alloc_colors = ["#4c78a8", "#72b7b2", "#e45756", "#54a24b", "#b279a2"]
    bottom = pd.Series([0.0] * len(comp))
    for col, label, color in zip(alloc_cols, alloc_labels, alloc_colors):
        axes[1].bar(x, comp[col], bottom=bottom, label=label, color=color)
        bottom += comp[col]
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(["Heuristic", "Std. PPO", "POMG-DSD"], rotation=0)
    axes[1].set_title("Defender Budget Mix")
    axes[1].set_ylabel("Mean action mass")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)

    fig.savefig(FIG_DIR / "pomg_method_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_lower_tier_amplifiers() -> None:
    dose = pd.read_csv(CSV_DIR / "pomg_lower_tier_dose_response.csv")
    palette = {
        "lambda_disinfo": "#e45756",
        "lambda_bias": "#4c78a8",
        "lambda_econ": "#54a24b",
    }
    labels = {
        "lambda_disinfo": "Disinformation",
        "lambda_bias": "Algorithmic bias",
        "lambda_econ": "Labor displacement",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
    for amplifier, subset in dose.groupby("amplifier"):
        subset = subset.sort_values("lambda")
        axes[0].plot(
            subset["lambda"],
            subset["peak_cbrn_mean"],
            lw=2,
            color=palette[amplifier],
            label=labels[amplifier],
        )
        axes[1].plot(
            subset["lambda"],
            subset["peak_overload_mean"],
            lw=2,
            color=palette[amplifier],
            label=labels[amplifier],
        )
    axes[0].set_title("Lower-Tier Harm Sweep: Peak CBRN Misuse")
    axes[0].set_xlabel(r"Amplifier strength $\lambda$")
    axes[0].set_ylabel("Mean peak misuse")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("Shared Substrate Stress: Institutional Overload")
    axes[1].set_xlabel(r"Amplifier strength $\lambda$")
    axes[1].set_ylabel("Mean peak overload")
    axes[1].legend(frameon=False, fontsize=8)

    fig.savefig(FIG_DIR / "pomg_lower_tier_amplifiers.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_adaptation_stress() -> None:
    hist = pd.read_csv(CSV_DIR / "pomg_adaptation_training.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)

    axes[0].plot(hist["steps_seen"], hist["adversary_return"], color="#d62728", lw=2)
    axes[0].set_title("Frozen-Defender Adversarial Adaptation")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Mean adversary return")

    axes[1].plot(hist["steps_seen"], hist["peak_loss_control"], color="#9467bd", lw=2, label="Peak loss of control")
    axes[1].plot(hist["steps_seen"], hist["min_cbf_margin"], color="#2ca02c", lw=2, label="Min CBF margin")
    axes[1].set_title("Adversary Stress During Adaptation")
    axes[1].set_xlabel("Environment steps")
    axes[1].legend(frameon=False, fontsize=8)

    fig.savefig(FIG_DIR / "pomg_adaptation_stress.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cbf_ablation() -> None:
    traj = pd.read_csv(CSV_DIR / "pomg_cbf_ablation_trajectory.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), constrained_layout=True)
    palette = {"cbf_on": "#4c78a8", "cbf_disabled_at_26": "#e45756"}
    labels = {"cbf_on": "CBF on", "cbf_disabled_at_26": "CBF disabled at week 26"}

    for scenario, subset in traj.groupby("scenario"):
        subset = subset.sort_values("week")
        axes[0].plot(subset["week"], subset["hazard_index"], color=palette[scenario], lw=2, label=labels[scenario])
        axes[1].plot(subset["week"], subset["loss_of_control"], color=palette[scenario], lw=2, label=labels[scenario])

    axes[0].set_title("Stress Rollout Hazard Index")
    axes[0].set_xlabel("Week")
    axes[0].set_ylabel("Hazard index")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title("Stress Rollout: Irreversible Loss of Control")
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("Loss-of-control stock")
    axes[1].legend(frameon=False, fontsize=8)

    fig.savefig(FIG_DIR / "pomg_cbf_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_boundary_conditions() -> None:
    terminal = pd.read_csv(CSV_DIR / "pomg_terminal_derivatives.csv")
    infinite_cap = pd.read_csv(CSV_DIR / "pomg_infinite_capability_ablation.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    x = range(len(terminal))
    cyber = terminal["terminal_d_cyber_mean"]
    cbrn = terminal["terminal_d_cbrn_mean"]
    cyber_sd = terminal["terminal_d_cyber_sd"]
    cbrn_sd = terminal["terminal_d_cbrn_sd"]

    axes[0].bar([v - 0.18 for v in x], cyber, width=0.34, yerr=cyber_sd, color="#4c78a8", capsize=4, label=r"$\dot{S}_3(T)$")
    axes[0].bar([v + 0.18 for v in x], cbrn, width=0.34, yerr=cbrn_sd, color="#e45756", capsize=4, label=r"$\dot{S}_3^C(T)$")
    axes[0].axhline(0.0, color="black", lw=1)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(["Heuristic", "Std. PPO", "POMG-DSD"], rotation=0)
    axes[0].set_title("Terminal Derivatives at Week 52")
    axes[0].set_ylabel("Mean terminal derivative")
    axes[0].legend(frameon=False, fontsize=8)

    x2 = range(len(infinite_cap))
    axes[1].bar(x2, infinite_cap["violation_week_mean"], color=["#4c78a8", "#f58518"], width=0.55, label="Mean first violation week")
    axes[1].errorbar(
        x2,
        infinite_cap["violation_week_mean"],
        yerr=infinite_cap["violation_week_sd"],
        fmt="none",
        ecolor="black",
        capsize=4,
        lw=1,
    )
    axes[1].plot(
        list(x2),
        104 * infinite_cap["mean_governance_alloc"],
        color="#54a24b",
        marker="o",
        lw=2,
        label=r"$104 \times$ mean governance allocation",
    )
    axes[1].set_xticks(list(x2))
    axes[1].set_xticklabels(["Std. PPO", "POMG-DSD"], rotation=0)
    axes[1].set_title(r"Unbounded Capability Ablation ($K_{\mathrm{cap}} \to \infty$)")
    axes[1].set_ylabel("Weeks / scaled governance allocation")
    axes[1].legend(frameon=False, fontsize=8)

    fig.savefig(FIG_DIR / "pomg_boundary_conditions.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    plot_calibration_overlay()
    plot_training_curves()
    plot_method_comparison()
    plot_lower_tier_amplifiers()
    plot_adaptation_stress()
    plot_cbf_ablation()
    plot_boundary_conditions()
    print("Wrote POMG-DSD figures to", FIG_DIR)


if __name__ == "__main__":
    main()
