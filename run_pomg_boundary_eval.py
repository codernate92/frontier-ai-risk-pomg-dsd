"""
Standalone runner for terminal-derivative and infinite-capability evaluations.
"""
from __future__ import annotations

import json
from pathlib import Path

from pomg_dsd import (
    DifferentiableSupergraphDynamics,
    RolloutConfig,
    evaluate_infinite_capability_ablation,
    evaluate_terminal_derivatives,
    set_seed,
    train_self_play,
)
from run_pomg_dsd import save_dict_csv


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv"
CSV_DIR.mkdir(exist_ok=True)


def main() -> None:
    set_seed(37)
    dynamics = DifferentiableSupergraphDynamics(device="cpu")
    dynamics.fit_low_confidence_signs(steps=90, lr=1.5e-2, horizon_weeks=52)

    train_cfg = RolloutConfig(horizon_steps=12, dt=1.0, batch_size=8, ppo_epochs=3, lr=3e-4, entropy_coeff=0.01, value_coeff=0.5)
    boundary_cfg = RolloutConfig(horizon_steps=52, dt=1.0, batch_size=1, ppo_epochs=1, lr=3e-4)

    standard_def, _standard_adv, _standard_hist = train_self_play(
        dynamics=dynamics,
        config=train_cfg,
        iterations=8,
        partial_observability=False,
        use_cbf=False,
        domain_randomization=False,
        seed=11,
    )
    proposed_def, proposed_adv, _proposed_hist = train_self_play(
        dynamics=dynamics,
        config=train_cfg,
        iterations=12,
        partial_observability=True,
        use_cbf=True,
        domain_randomization=True,
        seed=19,
    )

    terminal_rows = [
        {
            "method": "Heuristic static patching",
            **evaluate_terminal_derivatives(
                dynamics=dynamics,
                config=boundary_cfg,
                defender=None,
                adversary=proposed_adv,
                partial_observability=True,
                use_cbf=False,
                domain_randomization=True,
                episodes=12,
                heuristic_mode=True,
            ),
        },
        {
            "method": "Standard PPO",
            **evaluate_terminal_derivatives(
                dynamics=dynamics,
                config=boundary_cfg,
                defender=standard_def,
                adversary=proposed_adv,
                partial_observability=False,
                use_cbf=False,
                domain_randomization=True,
                episodes=12,
                heuristic_mode=False,
            ),
        },
        {
            "method": "POMG-DSD + CBF + domain randomization",
            **evaluate_terminal_derivatives(
                dynamics=dynamics,
                config=boundary_cfg,
                defender=proposed_def,
                adversary=proposed_adv,
                partial_observability=True,
                use_cbf=True,
                domain_randomization=True,
                episodes=12,
                heuristic_mode=False,
            ),
        },
    ]

    unbounded = dynamics.clone()
    unbounded.set_capability_ceiling(10.0)
    infinite_rows = [
        {
            "method": "Standard PPO",
            **evaluate_infinite_capability_ablation(
                dynamics=unbounded,
                config=boundary_cfg,
                defender=standard_def,
                adversary=proposed_adv,
                partial_observability=False,
                use_cbf=False,
                domain_randomization=True,
                episodes=8,
                horizon_steps=78,
                heuristic_mode=False,
            ),
        },
        {
            "method": "POMG-DSD + CBF + domain randomization",
            **evaluate_infinite_capability_ablation(
                dynamics=unbounded,
                config=boundary_cfg,
                defender=proposed_def,
                adversary=proposed_adv,
                partial_observability=True,
                use_cbf=True,
                domain_randomization=True,
                episodes=8,
                horizon_steps=78,
                heuristic_mode=False,
            ),
        },
    ]

    save_dict_csv(CSV_DIR / "pomg_terminal_derivatives.csv", terminal_rows)
    save_dict_csv(CSV_DIR / "pomg_infinite_capability_ablation.csv", infinite_rows)
    payload = {"terminal_derivatives": terminal_rows, "infinite_capability": infinite_rows}
    with (ROOT / "pomg_boundary_results.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
