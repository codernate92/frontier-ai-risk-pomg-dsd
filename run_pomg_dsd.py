"""
Run calibration, self-play training, and evaluation for the POMG-DSD stack.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from graphs import get_all_graphs
from pomg_dsd import (
    DifferentiableSupergraphDynamics,
    RolloutConfig,
    evaluate_pair,
    evaluate_amplifier_grid,
    evaluate_infinite_capability_ablation,
    evaluate_terminal_derivatives,
    rollout_policy_trajectory,
    set_seed,
    train_adversary_against_frozen_defender,
    train_self_play,
    write_training_history,
)


ROOT = Path(__file__).resolve().parent
CSV_DIR = ROOT / "csv"
CSV_DIR.mkdir(exist_ok=True)


def save_dict_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_edge_updates(dynamics: DifferentiableSupergraphDynamics, top_k: int = 20) -> list[dict[str, float | str]]:
    weights = dynamics.edge_weights().detach().cpu()
    priors = dynamics.edge_prior.detach().cpu()
    learn_mask = dynamics.learn_mask.detach().cpu()
    rows = []
    for idx, ((src, dst), learned, prior, learnable) in enumerate(
        zip(dynamics.edge_names, weights.tolist(), priors.tolist(), learn_mask.tolist())
    ):
        if not learnable:
            continue
        rows.append(
            {
                "edge": f"{src}->{dst}",
                "prior_weight": prior,
                "learned_weight": learned,
                "delta": learned - prior,
                "flipped": int((learned >= 0) != (prior >= 0)),
            }
        )
    rows.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    return rows[:top_k]


def count_paths_bfs(graph: nx.DiGraph, targets: list[str], max_depth: int = 8) -> dict[str, int]:
    path_counts = {target: 0 for target in targets}
    sources = [node for node in graph.nodes() if node not in targets and graph.out_degree(node) > 0]
    sources = sorted(sources, key=lambda node: -graph.out_degree(node))[:30]
    for target in targets:
        count = 0
        for source in sources:
            try:
                for _ in nx.all_simple_paths(graph, source, target, cutoff=max_depth):
                    count += 1
                    if count > 500:
                        break
            except (nx.NetworkXError, nx.NodeNotFound):
                continue
            if count > 500:
                break
        path_counts[target] = count
    return path_counts


def make_silo_graph(graph: nx.DiGraph) -> nx.DiGraph:
    graph_copy = graph.copy()
    to_remove = [(u, v) for u, v, data in graph_copy.edges(data=True) if data.get("cross_domain", False)]
    graph_copy.remove_edges_from(to_remove)
    return graph_copy


def make_direct_only_graph(graph: nx.DiGraph, shared_nodes: list[str]) -> nx.DiGraph:
    graph_copy = graph.copy()
    for node in shared_nodes:
        if graph_copy.has_node(node):
            graph_copy.remove_node(node)
    return graph_copy


def compute_structural_convergence() -> dict[str, object]:
    graphs = get_all_graphs()
    supergraph = graphs["supergraph"]
    endpoints = graphs["endpoint_nodes"]
    shared_nodes = graphs["shared_nodes"]

    full_paths = count_paths_bfs(supergraph, endpoints, max_depth=8)
    silo_graph = make_silo_graph(supergraph)
    silo_paths = count_paths_bfs(silo_graph, endpoints, max_depth=8)
    direct_graph = make_direct_only_graph(supergraph, shared_nodes)
    direct_paths = count_paths_bfs(direct_graph, endpoints, max_depth=8)

    rows: list[dict[str, float | str]] = []
    total_full = 0
    total_silo = 0
    total_direct = 0
    for endpoint in endpoints:
        full_count = full_paths[endpoint]
        silo_count = silo_paths[endpoint]
        direct_count = direct_paths[endpoint]
        rows.append(
            {
                "endpoint": endpoint,
                "full_graph": full_count,
                "silo": silo_count,
                "direct_only": direct_count,
                "convergence_pct": 100.0 * (1.0 - silo_count / max(full_count, 1)),
            }
        )
        total_full += full_count
        total_silo += silo_count
        total_direct += direct_count

    def total_bounded_paths(graph: nx.DiGraph) -> int:
        return sum(count_paths_bfs(graph, endpoints, max_depth=8).values())

    def block_key(graph: nx.DiGraph, u: str, v: str, data: dict[str, object]) -> tuple[object, ...]:
        return (
            graph.nodes[u].get("domain"),
            graph.nodes[v].get("domain"),
            data.get("sign"),
            data.get("cross_domain"),
        )

    def block_preserving_shuffle(graph: nx.DiGraph, rng: np.random.Generator, swap_frac: float = 0.35) -> nx.DiGraph:
        shuffled = graph.copy()
        buckets: dict[tuple[object, ...], list[tuple[str, str]]] = defaultdict(list)
        for u, v, data in shuffled.edges(data=True):
            buckets[block_key(shuffled, u, v, data)].append((u, v))

        for bucket_edges in buckets.values():
            if len(bucket_edges) < 2:
                continue
            target_swaps = max(1, int(len(bucket_edges) * swap_frac))
            swaps = 0
            attempts = 0
            while swaps < target_swaps and attempts < target_swaps * 60:
                attempts += 1
                idx = rng.choice(len(bucket_edges), size=2, replace=False)
                (u, v), (x, y) = bucket_edges[idx[0]], bucket_edges[idx[1]]
                if len({u, v, x, y}) < 4 or shuffled.has_edge(u, y) or shuffled.has_edge(x, v):
                    continue
                uv = shuffled[u][v].copy()
                xy = shuffled[x][y].copy()
                shuffled.remove_edge(u, v)
                shuffled.remove_edge(x, y)
                shuffled.add_edge(u, y, **uv)
                shuffled.add_edge(x, v, **xy)
                bucket_edges[idx[0]] = (u, y)
                bucket_edges[idx[1]] = (x, v)
                swaps += 1
        return shuffled

    rng = np.random.default_rng(2026)
    block_null_ratios: list[float] = []
    convergence_multiplier = total_full / max(total_silo, 1)
    for _ in range(200):
        permuted = block_preserving_shuffle(supergraph, rng)
        permuted_full = total_bounded_paths(permuted)
        permuted_silo = total_bounded_paths(make_silo_graph(permuted))
        if permuted_silo > 0:
            block_null_ratios.append(permuted_full / permuted_silo)

    save_dict_csv(CSV_DIR / "pomg_exp6_convergence.csv", rows + [{
        "endpoint": "Total",
        "full_graph": total_full,
        "silo": total_silo,
        "direct_only": total_direct,
        "convergence_pct": 100.0 * (1.0 - total_silo / max(total_full, 1)),
    }])
    save_dict_csv(
        CSV_DIR / "pomg_exp6_block_null.csv",
        [{"ratio": ratio} for ratio in block_null_ratios],
    )
    return {
        "node_count": supergraph.number_of_nodes(),
        "edge_count": supergraph.number_of_edges(),
        "endpoint_count": len(endpoints),
        "endpoint_names": endpoints,
        "rows": rows,
        "total_full": total_full,
        "total_silo": total_silo,
        "total_direct": total_direct,
        "convergence_multiplier": convergence_multiplier,
        "block_null_mean_ratio": float(np.mean(block_null_ratios)),
        "block_null_median_ratio": float(np.median(block_null_ratios)),
        "block_null_p_value": float(np.mean(np.array(block_null_ratios) >= convergence_multiplier)),
    }


def calibrated_trajectory_rows(dynamics: DifferentiableSupergraphDynamics, weeks: int = 52) -> list[dict[str, float]]:
    latent = dynamics.sample_latent(batch_size=1, domain_randomization=False)
    traj = dynamics.rollout_open_loop(horizon_steps=weeks, dt=1.0, latent=latent).detach().cpu()
    node = dynamics.node_index
    rows = []
    for t in range(traj.shape[0]):
        rows.append(
            {
                "week": float(t),
                "vuln_discovery_rate": float(traj[t, node["Vuln_Discovery_Rate"]]),
                "vuln_backlog": float(traj[t, node["Vuln_Backlog"]]),
                "incident_burden": float(traj[t, node["Incident_Burden"]]),
                "misuse_opportunity": float(traj[t, node["Misuse_Opportunity"]]),
                "trust_erosion": float(traj[t, node["Trust_Erosion"]]),
                "institutional_overload": float(traj[t, node["Institutional_Overload"]]),
            }
        )
    return rows


def main() -> None:
    set_seed(7)
    device = "cpu"
    dynamics = DifferentiableSupergraphDynamics(device=device)

    print("=" * 72)
    print("  CALIBRATING LEARNABLE EDGE POLARITIES AGAINST CISA KEV")
    print("=" * 72)
    calibration = dynamics.fit_low_confidence_signs(steps=90, lr=1.5e-2, horizon_weeks=52)
    print(json.dumps(calibration, indent=2))
    save_dict_csv(CSV_DIR / "pomg_calibration.csv", [{k: float(v) for k, v in calibration.items()}])
    save_dict_csv(CSV_DIR / "pomg_edge_updates.csv", top_edge_updates(dynamics))
    save_dict_csv(CSV_DIR / "pomg_calibrated_trajectory.csv", calibrated_trajectory_rows(dynamics))

    print("=" * 72)
    print("  RECOMPUTING STRUCTURAL CONVERGENCE WITH THE EXTENDED ENDPOINT SET")
    print("=" * 72)
    structural = compute_structural_convergence()
    print(json.dumps({k: v for k, v in structural.items() if k not in {"rows", "endpoint_names"}}, indent=2))

    train_cfg = RolloutConfig(
        horizon_steps=12,
        dt=1.0,
        batch_size=8,
        ppo_epochs=3,
        lr=3e-4,
        entropy_coeff=0.01,
        value_coeff=0.5,
    )
    eval_cfg = RolloutConfig(
        horizon_steps=18,
        dt=1.0,
        batch_size=16,
        ppo_epochs=1,
        lr=3e-4,
    )
    amp_cfg = RolloutConfig(
        horizon_steps=26,
        dt=1.0,
        batch_size=16,
        ppo_epochs=1,
        lr=3e-4,
    )
    long_cfg = RolloutConfig(
        horizon_steps=52,
        dt=1.0,
        batch_size=1,
        ppo_epochs=1,
        lr=3e-4,
    )
    boundary_cfg = RolloutConfig(
        horizon_steps=52,
        dt=1.0,
        batch_size=1,
        ppo_epochs=1,
        lr=3e-4,
    )

    print("=" * 72)
    print("  TRAINING STANDARD FULL-STATE PPO SELF-PLAY")
    print("=" * 72)
    standard_def, standard_adv, standard_hist = train_self_play(
        dynamics=dynamics,
        config=train_cfg,
        iterations=16,
        partial_observability=False,
        use_cbf=False,
        domain_randomization=False,
        seed=11,
    )
    write_training_history(CSV_DIR / "pomg_standard_training.csv", standard_hist)

    print("=" * 72)
    print("  TRAINING PARTIAL-OBS SAFE POMG-DSD SELF-PLAY")
    print("=" * 72)
    proposed_def, proposed_adv, proposed_hist = train_self_play(
        dynamics=dynamics,
        config=train_cfg,
        iterations=22,
        partial_observability=True,
        use_cbf=True,
        domain_randomization=True,
        seed=19,
    )
    write_training_history(CSV_DIR / "pomg_proposed_training.csv", proposed_hist)

    print("=" * 72)
    print("  EVALUATING HEURISTIC, STANDARD PPO, AND POMG-DSD")
    print("=" * 72)
    heuristic_metrics = evaluate_pair(
        dynamics=dynamics,
        defender=None,
        adversary=proposed_adv,
        config=eval_cfg,
        episodes=64,
        partial_observability=True,
        use_cbf=False,
        domain_randomization=True,
        heuristic_mode=True,
    )
    standard_metrics = evaluate_pair(
        dynamics=dynamics,
        defender=standard_def,
        adversary=proposed_adv,
        config=eval_cfg,
        episodes=64,
        partial_observability=False,
        use_cbf=False,
        domain_randomization=True,
        heuristic_mode=False,
    )
    proposed_metrics = evaluate_pair(
        dynamics=dynamics,
        defender=proposed_def,
        adversary=proposed_adv,
        config=eval_cfg,
        episodes=64,
        partial_observability=True,
        use_cbf=True,
        domain_randomization=True,
        heuristic_mode=False,
    )

    summary_rows = [
        {"method": "Heuristic static patching", **heuristic_metrics},
        {"method": "Standard PPO", **standard_metrics},
        {"method": "POMG-DSD + CBF + domain randomization", **proposed_metrics},
    ]
    save_dict_csv(CSV_DIR / "pomg_method_comparison.csv", summary_rows)

    print("=" * 72)
    print("  TERMINAL DERIVATIVE EVALUATION AT T = 52 WEEKS")
    print("=" * 72)
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
                episodes=32,
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
                episodes=32,
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
                episodes=32,
                heuristic_mode=False,
            ),
        },
    ]
    save_dict_csv(CSV_DIR / "pomg_terminal_derivatives.csv", terminal_rows)

    print("=" * 72)
    print("  INFINITE-CAPABILITY ABLATION")
    print("=" * 72)
    unbounded = dynamics.clone()
    unbounded.set_capability_ceiling(1.0e6)
    infinite_cap_rows = [
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
                episodes=24,
                horizon_steps=104,
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
                episodes=24,
                horizon_steps=104,
                heuristic_mode=False,
            ),
        },
    ]
    save_dict_csv(CSV_DIR / "pomg_infinite_capability_ablation.csv", infinite_cap_rows)

    print("=" * 72)
    print("  MONTE CARLO LOWER-TIER AMPLIFIER SWEEP")
    print("=" * 72)
    lambda_values = [round(x * 0.1, 1) for x in range(11)]
    amplifier_rows = []
    for amplifier_name in ("lambda_disinfo", "lambda_bias", "lambda_econ"):
        amplifier_rows.extend(
            evaluate_amplifier_grid(
                dynamics=dynamics,
                config=amp_cfg,
                amplifier_name=amplifier_name,
                lambda_values=lambda_values,
                episodes=64,
                heuristic_mode=True,
            )
        )
    save_dict_csv(CSV_DIR / "pomg_lower_tier_dose_response.csv", amplifier_rows)
    amplifier_summary = []
    for amplifier_name in ("lambda_disinfo", "lambda_bias", "lambda_econ"):
        subset = [row for row in amplifier_rows if row["amplifier"] == amplifier_name]
        baseline = next(row for row in subset if float(row["lambda"]) == 0.0)
        high = next(row for row in subset if float(row["lambda"]) == 1.0)
        amplifier_summary.append(
            {
                "amplifier": amplifier_name,
                "peak_cyber_delta_pct": 100.0 * (high["peak_cyber_mean"] - baseline["peak_cyber_mean"]) / max(baseline["peak_cyber_mean"], 1e-6),
                "peak_cbrn_delta_pct": 100.0 * (high["peak_cbrn_mean"] - baseline["peak_cbrn_mean"]) / max(baseline["peak_cbrn_mean"], 1e-6),
                "trust_erosion_delta_pct": 100.0 * (high["peak_trust_erosion_mean"] - baseline["peak_trust_erosion_mean"]) / max(baseline["peak_trust_erosion_mean"], 1e-6),
                "overload_delta_pct": 100.0 * (high["peak_overload_mean"] - baseline["peak_overload_mean"]) / max(baseline["peak_overload_mean"], 1e-6),
            }
        )
    save_dict_csv(CSV_DIR / "pomg_lower_tier_summary.csv", amplifier_summary)

    print("=" * 72)
    print("  ADVERSARIAL ADAPTATION AGAINST FROZEN CBF DEFENDER (10,000 STEPS)")
    print("=" * 72)
    adapted_adv, adaptation_hist = train_adversary_against_frozen_defender(
        dynamics=dynamics,
        defender=proposed_def,
        config=train_cfg,
        total_steps=10000,
        partial_observability=True,
        use_cbf=True,
        domain_randomization=True,
        seed=29,
    )
    write_training_history(CSV_DIR / "pomg_adaptation_training.csv", adaptation_hist)
    adaptation_eval = evaluate_pair(
        dynamics=dynamics,
        defender=proposed_def,
        adversary=adapted_adv,
        config=eval_cfg,
        episodes=64,
        partial_observability=True,
        use_cbf=True,
        domain_randomization=True,
        heuristic_mode=False,
    )
    save_dict_csv(CSV_DIR / "pomg_adaptation_eval.csv", [{"method": "Frozen POMG-DSD vs adapted adversary", **adaptation_eval}])

    print("=" * 72)
    print("  MID-ROLLOUT CBF ABLATION STRESS TEST")
    print("=" * 72)
    ablation_search_rows = []
    best_gap = -1e9
    cbf_rows = []
    ablated_rows = []
    cbf_summary = {}
    ablated_summary = {}
    for sample_id in range(24):
        fixed_latent = dynamics.sample_latent(
            batch_size=1,
            domain_randomization=True,
            amplifier_overrides={"lambda_disinfo": 1.0, "lambda_bias": 1.0, "lambda_econ": 1.0},
        )
        on_rows, on_summary = rollout_policy_trajectory(
            dynamics=dynamics,
            config=long_cfg,
            defender=proposed_def,
            adversary=adapted_adv,
            partial_observability=True,
            use_cbf=True,
            domain_randomization=True,
            preset_latent=fixed_latent,
        )
        off_rows, off_summary = rollout_policy_trajectory(
            dynamics=dynamics,
            config=long_cfg,
            defender=proposed_def,
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
        ablation_search_rows.append(
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
            cbf_rows = on_rows
            ablated_rows = off_rows
            cbf_summary = on_summary
            ablated_summary = off_summary

    save_dict_csv(CSV_DIR / "pomg_cbf_ablation_search.csv", ablation_search_rows)
    save_dict_csv(
        CSV_DIR / "pomg_cbf_ablation_trajectory.csv",
        [{"scenario": "cbf_on", **row} for row in cbf_rows] + [{"scenario": "cbf_disabled_at_26", **row} for row in ablated_rows],
    )
    save_dict_csv(
        CSV_DIR / "pomg_cbf_ablation_summary.csv",
        [
            {"scenario": "cbf_on", **cbf_summary},
            {"scenario": "cbf_disabled_at_26", **ablated_summary},
        ],
    )

    results = {
        "calibration": calibration,
        "structural": structural,
        "standard_final": standard_hist[-1],
        "proposed_final": proposed_hist[-1],
        "comparison": {
            "heuristic": heuristic_metrics,
            "standard_ppo": standard_metrics,
            "pomg_dsd": proposed_metrics,
        },
        "boundary_conditions": {
            "terminal_derivatives": terminal_rows,
            "infinite_capability": infinite_cap_rows,
        },
        "lower_tier_amplifiers": {
            "dose_response": amplifier_rows,
            "summary": amplifier_summary,
        },
        "adaptation": {
            "final_training": adaptation_hist[-1],
            "evaluation": adaptation_eval,
        },
        "cbf_ablation": {
            "search": ablation_search_rows,
            "cbf_on": cbf_summary,
            "cbf_disabled_at_26": ablated_summary,
        },
    }
    with (ROOT / "pomg_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results["comparison"], indent=2))
    print("Results written to", ROOT / "pomg_results.json")


if __name__ == "__main__":
    main()
