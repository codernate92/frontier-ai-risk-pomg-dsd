"""
Standalone runner for the mid-rollout CBF shutdown stress test.
"""
from __future__ import annotations

import json
from pathlib import Path

from pomg_dsd import (
    DifferentiableSupergraphDynamics,
    RolloutConfig,
    rollout_policy_trajectory,
    set_seed,
    train_adversary_against_frozen_defender,
    train_self_play,
    write_training_history,
)
from run_pomg_dsd import save_dict_csv


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv"
CSV_DIR.mkdir(exist_ok=True)


def main() -> None:
    set_seed(31)
    dynamics = DifferentiableSupergraphDynamics(device="cpu")
    dynamics.fit_low_confidence_signs(steps=90, lr=1.5e-2, horizon_weeks=52)

    train_cfg = RolloutConfig(horizon_steps=12, dt=1.0, batch_size=8, ppo_epochs=3, lr=3e-4, entropy_coeff=0.01, value_coeff=0.5)
    long_cfg = RolloutConfig(horizon_steps=52, dt=1.0, batch_size=1, ppo_epochs=1, lr=3e-4)

    defender, _reference_adv, _hist = train_self_play(
        dynamics=dynamics,
        config=train_cfg,
        iterations=22,
        partial_observability=True,
        use_cbf=True,
        domain_randomization=True,
        seed=19,
    )
    adapted_adv, adaptation_hist = train_adversary_against_frozen_defender(
        dynamics=dynamics,
        defender=defender,
        config=train_cfg,
        total_steps=10000,
        partial_observability=True,
        use_cbf=True,
        domain_randomization=True,
        seed=29,
    )
    write_training_history(CSV_DIR / "pomg_adaptation_training.csv", adaptation_hist)

    rows = []
    best_gap = -1.0e9
    best_on = []
    best_off = []
    best_on_summary = {}
    best_off_summary = {}
    for sample_id in range(24):
        fixed_latent = dynamics.sample_latent(
            batch_size=1,
            domain_randomization=True,
            amplifier_overrides={"lambda_disinfo": 1.0, "lambda_bias": 1.0, "lambda_econ": 1.0},
        )
        on_rows, on_summary = rollout_policy_trajectory(
            dynamics=dynamics,
            config=long_cfg,
            defender=defender,
            adversary=adapted_adv,
            partial_observability=True,
            use_cbf=True,
            domain_randomization=True,
            preset_latent=fixed_latent,
        )
        off_rows, off_summary = rollout_policy_trajectory(
            dynamics=dynamics,
            config=long_cfg,
            defender=defender,
            adversary=adapted_adv,
            partial_observability=True,
            use_cbf=True,
            domain_randomization=True,
            cbf_disable_step=26,
            preset_latent=fixed_latent,
        )
        gap = (
            (off_summary["final_loss_of_control"] - on_summary["final_loss_of_control"])
            + (off_summary["peak_loss_control"] - on_summary["peak_loss_control"])
            + (off_summary["max_hazard_index"] - on_summary["max_hazard_index"])
        )
        rows.append(
            {
                "sample_id": sample_id,
                "cbf_on_final_loss_of_control": on_summary["final_loss_of_control"],
                "cbf_off_final_loss_of_control": off_summary["final_loss_of_control"],
                "cbf_on_peak_loss_control": on_summary["peak_loss_control"],
                "cbf_off_peak_loss_control": off_summary["peak_loss_control"],
                "cbf_on_max_hazard_index": on_summary["max_hazard_index"],
                "cbf_off_max_hazard_index": off_summary["max_hazard_index"],
                "gap_score": gap,
            }
        )
        if gap > best_gap:
            best_gap = gap
            best_on = on_rows
            best_off = off_rows
            best_on_summary = on_summary
            best_off_summary = off_summary

    save_dict_csv(CSV_DIR / "pomg_cbf_ablation_search.csv", rows)
    save_dict_csv(
        CSV_DIR / "pomg_cbf_ablation_trajectory.csv",
        [{"scenario": "cbf_on", **row} for row in best_on] + [{"scenario": "cbf_disabled_at_26", **row} for row in best_off],
    )
    save_dict_csv(
        CSV_DIR / "pomg_cbf_ablation_summary.csv",
        [{"scenario": "cbf_on", **best_on_summary}, {"scenario": "cbf_disabled_at_26", **best_off_summary}],
    )
    payload = {"search": rows, "cbf_on": best_on_summary, "cbf_disabled_at_26": best_off_summary}
    with (ROOT / "pomg_cbf_ablation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
