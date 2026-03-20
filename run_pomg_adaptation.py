"""
Standalone runner for the frozen-defender adversarial adaptation experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

from pomg_dsd import (
    DifferentiableSupergraphDynamics,
    RolloutConfig,
    evaluate_pair,
    set_seed,
    train_adversary_against_frozen_defender,
    train_self_play,
    write_training_history,
)


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv"
CSV_DIR.mkdir(exist_ok=True)


def main() -> None:
    set_seed(29)
    dynamics = DifferentiableSupergraphDynamics(device="cpu")
    dynamics.fit_low_confidence_signs(steps=90, lr=1.5e-2, horizon_weeks=52)

    train_cfg = RolloutConfig(horizon_steps=12, dt=1.0, batch_size=8, ppo_epochs=3, lr=3e-4, entropy_coeff=0.01, value_coeff=0.5)
    eval_cfg = RolloutConfig(horizon_steps=18, dt=1.0, batch_size=16, ppo_epochs=1, lr=3e-4)

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
    evaluation = evaluate_pair(
        dynamics=dynamics,
        defender=defender,
        adversary=adapted_adv,
        config=eval_cfg,
        episodes=64,
        partial_observability=True,
        use_cbf=True,
        domain_randomization=True,
        heuristic_mode=False,
    )

    write_training_history(CSV_DIR / "pomg_adaptation_training.csv", adaptation_hist)
    write_training_history(CSV_DIR / "pomg_adaptation_eval.csv", [{"iteration": 0.0, **evaluation}])
    payload = {"final_training": adaptation_hist[-1], "evaluation": evaluation}
    with (ROOT / "pomg_adaptation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
