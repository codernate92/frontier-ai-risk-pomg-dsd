"""
POMG-DSD environment and training utilities.

This module upgrades the original signed supergraph into a bounded, nonlinear,
full-state differentiable simulator that can be used as a partially observable
zero-sum control environment. The implementation is intentionally explicit:
all supergraph nodes participate in the transition dynamics, medium/low-
confidence edge polarities are learnable, and defender controls are wrapped
with a control-barrier-style safety check and an institutional-inertia limit.
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchdiffeq import odeint

from graphs import get_all_graphs


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _atanh_clamped(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))


def _zscore(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False)
    if std < 1e-6:
        return values * 0.0
    return (values - values.mean()) / std


def _simplex_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, min=eps)
    return x / torch.clamp(x.sum(dim=-1, keepdim=True), min=eps)


def _project_to_simplex_box(
    target: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Project onto the intersection of the simplex and a coordinate-wise box.

    This is a lightweight batch projection used to model actuator limits on
    defender budget reallocation. The feasible set is guaranteed for the
    current use case because `lower.sum() <= 1 <= upper.sum()` row-wise.
    """
    clipped = torch.minimum(torch.maximum(target, lower), upper)
    row_sum = clipped.sum(dim=-1, keepdim=True)

    over_mask = row_sum > 1.0 + eps
    if torch.any(over_mask):
        slack = torch.clamp(clipped - lower, min=0.0)
        slack_sum = torch.clamp(slack.sum(dim=-1, keepdim=True), min=eps)
        excess = torch.clamp(row_sum - 1.0, min=0.0)
        clipped = clipped - over_mask.float() * excess * slack / slack_sum

    row_sum = clipped.sum(dim=-1, keepdim=True)
    under_mask = row_sum < 1.0 - eps
    if torch.any(under_mask):
        headroom = torch.clamp(upper - clipped, min=0.0)
        headroom_sum = torch.clamp(headroom.sum(dim=-1, keepdim=True), min=eps)
        deficit = torch.clamp(1.0 - row_sum, min=0.0)
        clipped = clipped + under_mask.float() * deficit * headroom / headroom_sum

    clipped = torch.minimum(torch.maximum(clipped, lower), upper)
    return _simplex_normalize(clipped, eps=eps)


def load_kev_weekly_anchors(
    kev_path: Optional[Path] = None,
    horizon_weeks: int = 52,
    device: str | torch.device = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load a small weekly anchor set from the local CISA KEV feed."""
    if kev_path is None:
        kev_path = Path(__file__).with_name("known_exploited_vulnerabilities.json")
    with kev_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("vulnerabilities", [])
    if not rows:
        raise ValueError(f"No vulnerabilities found in {kev_path}")

    from datetime import datetime, timedelta

    fmt = "%Y-%m-%d"
    added = [datetime.strptime(row["dateAdded"], fmt) for row in rows]
    due = [datetime.strptime(row["dueDate"], fmt) for row in rows]
    end = max(added)
    start = end - timedelta(weeks=horizon_weeks)

    weekly_additions: List[float] = []
    weekly_backlog: List[float] = []
    week_times: List[float] = []
    for i in range(horizon_weeks + 1):
        t = start + timedelta(weeks=i)
        weekly_additions.append(
            float(sum(1 for a in added if t <= a < t + timedelta(weeks=1)))
        )
        weekly_backlog.append(float(sum(1 for a, d in zip(added, due) if a <= t < d)))
        week_times.append(float(i))

    return {
        "week": torch.tensor(week_times, dtype=torch.float32, device=device),
        "weekly_additions": torch.tensor(weekly_additions, dtype=torch.float32, device=device),
        "weekly_backlog": torch.tensor(weekly_backlog, dtype=torch.float32, device=device),
    }


def _contains_any(name: str, needles: Iterable[str]) -> bool:
    lname = name.lower()
    return any(needle in lname for needle in needles)


class DifferentiableSupergraphDynamics(nn.Module):
    """Nonlinear bounded ODE over the full supergraph."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__()
        graph_bundle = get_all_graphs()
        self.graph = graph_bundle["supergraph"]
        self.nodes = list(self.graph.nodes())
        self.node_index = {name: idx for idx, name in enumerate(self.nodes)}
        self.domains = [self.graph.nodes[node].get("domain", "shared") for node in self.nodes]
        self.n_nodes = len(self.nodes)

        self.device_name = str(device)
        self.device = torch.device(device)

        capacities: List[float] = []
        init_levels: List[float] = []
        rates: List[float] = []
        biases: List[float] = []
        tipping_thresholds: List[float] = []
        domain_mask = {"cyber": [], "cbrn": [], "deception": [], "autonomy": [], "governance": []}
        beneficial_mask: List[float] = []
        harmful_mask: List[float] = []
        endpoint_mask: List[float] = []
        obs_family: List[int] = []
        hazard_weights: List[float] = []

        for node, domain in zip(self.nodes, self.domains):
            cap, init, rate, bias, tipping, beneficial, obs_id, hazard = self._node_profile(node, domain)
            capacities.append(cap)
            init_levels.append(init)
            rates.append(rate)
            biases.append(bias)
            tipping_thresholds.append(tipping)
            beneficial_mask.append(1.0 if beneficial else 0.0)
            harmful_mask.append(0.0 if beneficial else 1.0)
            endpoint_mask.append(1.0 if domain == "endpoint" else 0.0)
            obs_family.append(obs_id)
            hazard_weights.append(hazard)
            if domain in domain_mask:
                domain_mask[domain].append(self.node_index[node])

        self.register_buffer("capacity", torch.tensor(capacities, dtype=torch.float32, device=self.device))
        self.register_buffer("init_state", torch.tensor(init_levels, dtype=torch.float32, device=self.device))
        self.register_buffer("base_rate", torch.tensor(rates, dtype=torch.float32, device=self.device))
        self.register_buffer("tipping_threshold", torch.tensor(tipping_thresholds, dtype=torch.float32, device=self.device))
        self.register_buffer("beneficial_mask", torch.tensor(beneficial_mask, dtype=torch.float32, device=self.device))
        self.register_buffer("harmful_mask", torch.tensor(harmful_mask, dtype=torch.float32, device=self.device))
        self.register_buffer("endpoint_mask", torch.tensor(endpoint_mask, dtype=torch.float32, device=self.device))
        self.register_buffer("obs_family", torch.tensor(obs_family, dtype=torch.long, device=self.device))
        self.register_buffer("hazard_weight", torch.tensor(hazard_weights, dtype=torch.float32, device=self.device))

        self.node_bias = nn.Parameter(torch.tensor(biases, dtype=torch.float32, device=self.device))
        self.rate_log_scale = nn.Parameter(torch.zeros(self.n_nodes, dtype=torch.float32, device=self.device))
        self.initial_bias = torch.tensor(biases, dtype=torch.float32, device=self.device)

        src_idx: List[int] = []
        dst_idx: List[int] = []
        sign_prior: List[float] = []
        conf_scale: List[float] = []
        delay_scale: List[float] = []
        learn_mask: List[bool] = []
        edge_names: List[Tuple[str, str]] = []
        for src, dst, data in self.graph.edges(data=True):
            src_idx.append(self.node_index[src])
            dst_idx.append(self.node_index[dst])
            sign = float(data.get("sign", 1))
            conf = data.get("confidence", "M")
            delay = float(data.get("delay", 0))
            if conf == "H":
                scale = 0.95
                learnable = False
            elif conf == "M":
                scale = 0.70
                learnable = True
            else:
                scale = 0.45
                learnable = True
            sign_prior.append(sign)
            conf_scale.append(scale)
            delay_scale.append(1.0 / (1.0 + delay))
            learn_mask.append(learnable)
            edge_names.append((src, dst))

        init_weights = torch.tensor(
            [s * c for s, c in zip(sign_prior, conf_scale)],
            dtype=torch.float32,
            device=self.device,
        )
        theta_init = _atanh_clamped(init_weights)
        self.theta_graph = nn.Parameter(theta_init)
        self.register_buffer("src_index", torch.tensor(src_idx, dtype=torch.long, device=self.device))
        self.register_buffer("dst_index", torch.tensor(dst_idx, dtype=torch.long, device=self.device))
        self.register_buffer("sign_prior", torch.tensor(sign_prior, dtype=torch.float32, device=self.device))
        self.register_buffer("conf_scale", torch.tensor(conf_scale, dtype=torch.float32, device=self.device))
        self.register_buffer("delay_scale", torch.tensor(delay_scale, dtype=torch.float32, device=self.device))
        self.register_buffer("learn_mask", torch.tensor(learn_mask, dtype=torch.bool, device=self.device))
        self.edge_names = edge_names

        self.register_buffer("domain_mask_cyber", self._build_domain_mask(domain_mask["cyber"]))
        self.register_buffer("domain_mask_cbrn", self._build_domain_mask(domain_mask["cbrn"]))
        self.register_buffer("domain_mask_deception", self._build_domain_mask(domain_mask["deception"]))
        self.register_buffer("domain_mask_autonomy", self._build_domain_mask(domain_mask["autonomy"]))
        self.register_buffer("domain_mask_governance", self._build_domain_mask(domain_mask["governance"]))

        self.defender_action_names = [
            "cyber_defense",
            "biosecurity_review",
            "epistemic_resilience",
            "governance_response",
            "autonomy_control",
        ]
        self.adversary_action_names = [
            "cyber_offense",
            "cbrn_pressure",
            "disinformation_push",
            "autonomy_race",
            "governance_evasion",
        ]
        self.defender_dim = len(self.defender_action_names)
        self.adversary_dim = len(self.adversary_action_names)

        self.register_buffer("defender_matrix", self._build_defender_matrix())
        self.register_buffer("adversary_matrix", self._build_adversary_matrix())
        self.register_buffer("cbf_direction", self._build_cbf_direction())
        self.edge_prior = init_weights.detach().clone()

    def clone(self) -> "DifferentiableSupergraphDynamics":
        cloned = DifferentiableSupergraphDynamics(device=self.device)
        cloned.load_state_dict(self.state_dict())
        return cloned

    def capability_node_indices(self) -> List[int]:
        return [
            idx
            for name, idx in self.node_index.items()
            if "AI_Capability" in name
        ]

    def set_capability_ceiling(self, value: float) -> None:
        for idx in self.capability_node_indices():
            self.capacity[idx] = value

    def state_upper_bound(self) -> torch.Tensor:
        return torch.maximum(self.capacity, torch.full_like(self.capacity, 1.2))

    def _build_domain_mask(self, indices: List[int]) -> torch.Tensor:
        mask = torch.zeros(self.n_nodes, dtype=torch.float32, device=self.device)
        if indices:
            mask[torch.tensor(indices, dtype=torch.long, device=self.device)] = 1.0
        return mask

    def _node_profile(
        self,
        node: str,
        domain: str,
    ) -> Tuple[float, float, float, float, float, bool, int, float]:
        harmful_tokens = (
            "burden",
            "risk",
            "gap",
            "loss",
            "rogue",
            "volume",
            "stock",
            "backlog",
            "delay",
            "pressure",
            "opportunity",
            "severity",
            "exploits",
            "prevalence",
            "polarization",
            "grievance",
            "capture",
            "fragmentation",
            "degradation",
            "overload",
            "subversion",
            "harm",
        )
        beneficial_tokens = (
            "capacity",
            "effectiveness",
            "accuracy",
            "trust",
            "integrity",
            "cohesion",
            "confidence",
            "coordination",
            "literacy",
            "awareness",
            "response",
            "stringency",
            "protection",
            "thoroughness",
            "throughput",
            "consensus",
            "oversight",
            "moderation",
            "review",
        )

        harmful = _contains_any(node, harmful_tokens)
        beneficial = _contains_any(node, beneficial_tokens) and not harmful
        endpoint = domain == "endpoint"
        shared = domain == "shared"

        if endpoint:
            cap, init, rate, bias, tipping, obs_id = 1.2, 0.04, 0.55, -2.0, 0.60, 5
            hazard = 1.20
            return cap, init, rate, bias, tipping, False, obs_id, hazard

        if harmful:
            cap, init, rate, bias, tipping = 1.0, 0.18, 0.70, -1.05, 0.60
        elif beneficial:
            cap, init, rate, bias, tipping = 1.0, 0.55, 0.48, 0.65, 0.50
        else:
            cap, init, rate, bias, tipping = 1.0, 0.35, 0.55, -0.10, 0.55

        if shared:
            rate *= 1.05
            bias -= 0.05

        obs_lookup = {
            "cyber": 0,
            "cbrn": 1,
            "deception": 2,
            "governance": 3,
            "autonomy": 4,
            "shared": 5,
        }
        obs_id = obs_lookup.get(domain, 5)

        hazard = 0.0
        if node in {
            "Incident_Burden",
            "Misuse_Opportunity",
            "Disinformation_Volume",
            "Escalation_Risk",
            "Rogue_Agent_Autonomy",
            "Governance_Gap",
            "Trust_Erosion",
            "Institutional_Overload",
            "Detection_Degradation",
            "Response_Delay_Shared",
            "Irreversible_Loss_of_Control",
        }:
            hazard_map = {
                "Incident_Burden": 1.00,
                "Misuse_Opportunity": 1.00,
                "Disinformation_Volume": 0.70,
                "Escalation_Risk": 0.80,
                "Rogue_Agent_Autonomy": 0.95,
                "Governance_Gap": 0.75,
                "Trust_Erosion": 0.65,
                "Institutional_Overload": 0.90,
                "Detection_Degradation": 0.55,
                "Response_Delay_Shared": 0.55,
                "Irreversible_Loss_of_Control": 1.25,
            }
            hazard = hazard_map[node]
        return cap, init, rate, bias, tipping, beneficial, obs_id, hazard

    def _build_defender_matrix(self) -> torch.Tensor:
        matrix = torch.zeros(self.defender_dim, self.n_nodes, dtype=torch.float32, device=self.device)
        mapping = {
            "Defender_Capacity": [1.30, 0.00, 0.00, 0.00, 0.00],
            "Monitoring_Effectiveness": [0.80, 0.00, 0.00, 0.00, 0.00],
            "Patch_Throughput": [0.85, 0.00, 0.00, 0.00, 0.00],
            "Review_Capacity": [0.00, 1.20, 0.00, 0.00, 0.00],
            "Institutional_Review_Throughput": [0.00, 0.85, 0.00, 0.00, 0.00],
            "Red_Team_Detection_Sensitivity": [0.00, 0.70, 0.00, 0.00, 0.00],
            "Fact_Check_Capacity": [0.00, 0.00, 1.15, 0.00, 0.00],
            "Detection_Tool_Sophistication": [0.00, 0.00, 0.90, 0.00, 0.00],
            "Counter_Narrative_Capacity": [0.00, 0.00, 0.75, 0.00, 0.00],
            "Media_Literacy": [0.00, 0.00, 0.55, 0.00, 0.00],
            "Epistemic_Trust": [0.00, 0.00, 0.70, 0.00, 0.00],
            "Institutional_Credibility": [0.00, 0.00, 0.65, 0.00, 0.00],
            "Regulatory_Capacity": [0.00, 0.00, 0.00, 1.10, 0.00],
            "Policy_Response_Speed": [0.00, 0.00, 0.00, 0.90, 0.00],
            "Institutional_Trust_Gov": [0.00, 0.00, 0.20, 0.65, 0.00],
            "International_Coordination": [0.00, 0.00, 0.00, 0.55, 0.00],
            "Safety_Testing_Thoroughness": [0.00, 0.00, 0.00, 0.00, 0.95],
            "Human_Oversight_Effectiveness": [0.00, 0.00, 0.00, 0.00, 0.90],
            "Alignment_Confidence": [0.00, 0.00, 0.00, 0.00, 0.60],
            "Recovery_Capacity": [0.00, 0.00, 0.00, 0.00, 0.55],
        }
        for node, weights in mapping.items():
            if node in self.node_index:
                matrix[:, self.node_index[node]] = torch.tensor(weights, dtype=torch.float32, device=self.device)
        return matrix

    def _build_adversary_matrix(self) -> torch.Tensor:
        matrix = torch.zeros(self.adversary_dim, self.n_nodes, dtype=torch.float32, device=self.device)
        mapping = {
            "Vuln_Discovery_Rate": [1.20, 0.00, 0.00, 0.00, 0.00],
            "Exploit_Dev_Speed": [1.00, 0.00, 0.00, 0.00, 0.00],
            "Attack_Sophistication": [0.85, 0.00, 0.00, 0.00, 0.00],
            "Dangerous_Query_Volume": [0.00, 1.20, 0.00, 0.00, 0.00],
            "Successful_Bypass_Rate": [0.00, 0.95, 0.00, 0.00, 0.00],
            "Disinformation_Volume": [0.00, 0.00, 1.25, 0.00, 0.00],
            "Targeted_Manipulation_Precision": [0.00, 0.00, 0.85, 0.00, 0.00],
            "Synthetic_Persona_Prevalence": [0.00, 0.00, 0.70, 0.00, 0.00],
            "Competitive_Dynamics": [0.00, 0.00, 0.00, 1.10, 0.00],
            "Deployment_Pressure": [0.00, 0.00, 0.00, 0.85, 0.00],
            "Innovation_Speed": [0.00, 0.15, 0.00, 0.35, 0.70],
            "Industry_Lobbying_Pressure": [0.00, 0.00, 0.00, 0.00, 1.15],
            "Information_Asymmetry": [0.00, 0.00, 0.25, 0.00, 0.75],
            "Open_Weight_Avail": [0.45, 0.00, 0.25, 0.25, 0.00],
        }
        for node, weights in mapping.items():
            if node in self.node_index:
                matrix[:, self.node_index[node]] = torch.tensor(weights, dtype=torch.float32, device=self.device)
        return matrix

    def _build_cbf_direction(self) -> torch.Tensor:
        """Linear action-to-safety direction used in the CBF inequality."""
        return self.defender_matrix @ self.hazard_weight

    def edge_weights(self) -> torch.Tensor:
        learned = torch.tanh(self.theta_graph)
        fixed = self.sign_prior * self.conf_scale
        return torch.where(self.learn_mask, learned, fixed)

    def prior_regularizer(self) -> torch.Tensor:
        weights = self.edge_weights()
        return F.mse_loss(weights[self.learn_mask], self.edge_prior[self.learn_mask])

    def control_barrier_terms(
        self,
        state: torch.Tensor,
        defender_alloc: torch.Tensor,
        adversary_drive: torch.Tensor,
        latent: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        zero_alloc = torch.zeros_like(defender_alloc)
        drift_no_def = self._drift(state, zero_alloc, adversary_drive, latent)
        h = self.safety_function(state)
        l_f = -torch.sum(drift_no_def * self.hazard_weight, dim=-1)
        action_coeff = self.cbf_direction.unsqueeze(0).expand(defender_alloc.shape[0], -1)
        l_g_a = torch.sum(action_coeff * defender_alloc, dim=-1)
        alpha_h = 1.0 * h
        return {
            "h": h,
            "l_f": l_f,
            "l_g_a": l_g_a,
            "constraint": l_f + l_g_a + alpha_h,
        }

    def project_defender_allocation(
        self,
        state: torch.Tensor,
        defender_alloc: torch.Tensor,
        adversary_drive: torch.Tensor,
        latent: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        terms = self.control_barrier_terms(state, defender_alloc, adversary_drive, latent)
        coeff = torch.clamp(self.cbf_direction, min=0.0).unsqueeze(0).expand_as(defender_alloc)
        gap = torch.clamp(-(terms["constraint"]), min=0.0)
        coeff_norm = torch.clamp((coeff ** 2).sum(dim=-1), min=1e-6)
        delta = (gap / coeff_norm).unsqueeze(-1) * coeff
        projected = _simplex_normalize(defender_alloc + delta)
        projected_terms = self.control_barrier_terms(state, projected, adversary_drive, latent)
        return projected, projected_terms

    def safety_function(self, state: torch.Tensor) -> torch.Tensor:
        hazard = self.hazard_index(state)
        return 1.20 - hazard

    def hazard_index(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sum(state * self.hazard_weight, dim=-1) / torch.clamp(self.hazard_weight.sum(), min=1.0)

    def initial_state_batch(self, batch_size: int) -> torch.Tensor:
        return self.init_state.unsqueeze(0).repeat(batch_size, 1)

    def observation(self, state: torch.Tensor, latent: Dict[str, torch.Tensor], partial: bool) -> torch.Tensor:
        if not partial:
            return state

        node = self.node_index
        cyber_sensor = torch.clamp(state[:, node["Monitoring_Effectiveness"]], 0.05, 1.0)
        cbrn_sensor = torch.clamp(
            0.5 * (state[:, node["Review_Capacity"]] + state[:, node["Red_Team_Detection_Sensitivity"]]),
            0.05,
            1.0,
        )
        deception_sensor = torch.clamp(
            0.5 * (state[:, node["Fact_Check_Capacity"]] + state[:, node["Detection_Tool_Sophistication"]]),
            0.05,
            1.0,
        )
        governance_sensor = torch.clamp(state[:, node["Regulatory_Capacity"]], 0.05, 1.0)
        autonomy_sensor = torch.clamp(state[:, node["Human_Oversight_Effectiveness"]], 0.05, 1.0)
        shared_sensor = torch.clamp(
            0.4 * cyber_sensor + 0.2 * cbrn_sensor + 0.2 * deception_sensor + 0.2 * governance_sensor,
            0.05,
            1.0,
        )
        sensor_bank = torch.stack(
            [cyber_sensor, cbrn_sensor, deception_sensor, governance_sensor, autonomy_sensor, shared_sensor],
            dim=-1,
        )
        weights = torch.gather(sensor_bank, 1, self.obs_family.unsqueeze(0).expand(state.shape[0], -1))
        noise_scale = latent["sensor_noise"].unsqueeze(-1) * (1.15 - weights)
        noise = torch.randn_like(state) * noise_scale
        return torch.clamp(weights * state + (1.0 - weights) * 0.5 * state + noise, 0.0, 1.2)

    def sample_latent(
        self,
        batch_size: int,
        domain_randomization: bool = True,
        amplifier_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        if domain_randomization:
            cyber_scale = torch.empty(batch_size, device=self.device).uniform_(0.80, 1.35)
            cbrn_scale = torch.empty(batch_size, device=self.device).uniform_(0.75, 1.40)
            deception_scale = torch.empty(batch_size, device=self.device).uniform_(0.75, 1.45)
            autonomy_scale = torch.empty(batch_size, device=self.device).uniform_(0.80, 1.35)
            governance_scale = torch.empty(batch_size, device=self.device).uniform_(0.75, 1.45)
            sensor_noise = torch.empty(batch_size, device=self.device).uniform_(0.03, 0.10)
            threshold_shift = torch.empty(batch_size, device=self.device).uniform_(-0.07, 0.07)
            lambda_disinfo = torch.empty(batch_size, device=self.device).uniform_(0.00, 0.35)
            lambda_bias = torch.empty(batch_size, device=self.device).uniform_(0.00, 0.30)
            lambda_econ = torch.empty(batch_size, device=self.device).uniform_(0.00, 0.30)
            override_failure_threshold = torch.empty(batch_size, device=self.device).uniform_(0.56, 0.68)
        else:
            cyber_scale = torch.ones(batch_size, device=self.device)
            cbrn_scale = torch.ones(batch_size, device=self.device)
            deception_scale = torch.ones(batch_size, device=self.device)
            autonomy_scale = torch.ones(batch_size, device=self.device)
            governance_scale = torch.ones(batch_size, device=self.device)
            sensor_noise = torch.full((batch_size,), 0.03, device=self.device)
            threshold_shift = torch.zeros(batch_size, device=self.device)
            lambda_disinfo = torch.zeros(batch_size, device=self.device)
            lambda_bias = torch.zeros(batch_size, device=self.device)
            lambda_econ = torch.zeros(batch_size, device=self.device)
            override_failure_threshold = torch.full((batch_size,), 0.62, device=self.device)

        if amplifier_overrides:
            for key, value in amplifier_overrides.items():
                if isinstance(value, torch.Tensor):
                    tensor_value = value.to(self.device).reshape(-1)
                else:
                    tensor_value = torch.full((batch_size,), float(value), device=self.device)
                if key == "lambda_disinfo":
                    lambda_disinfo = tensor_value
                elif key == "lambda_bias":
                    lambda_bias = tensor_value
                elif key == "lambda_econ":
                    lambda_econ = tensor_value
                elif key == "override_failure_threshold":
                    override_failure_threshold = tensor_value

        return {
            "cyber_scale": cyber_scale,
            "cbrn_scale": cbrn_scale,
            "deception_scale": deception_scale,
            "autonomy_scale": autonomy_scale,
            "governance_scale": governance_scale,
            "sensor_noise": sensor_noise,
            "threshold_shift": threshold_shift,
            "lambda_disinfo": lambda_disinfo,
            "lambda_bias": lambda_bias,
            "lambda_econ": lambda_econ,
            "override_failure_threshold": override_failure_threshold,
        }

    def _domain_scale(self, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            latent["cyber_scale"].unsqueeze(-1) * self.domain_mask_cyber
            + latent["cbrn_scale"].unsqueeze(-1) * self.domain_mask_cbrn
            + latent["deception_scale"].unsqueeze(-1) * self.domain_mask_deception
            + latent["autonomy_scale"].unsqueeze(-1) * self.domain_mask_autonomy
            + latent["governance_scale"].unsqueeze(-1) * self.domain_mask_governance
            + (self.obs_family == 5).float().unsqueeze(0)
        )

    def _aggregate_messages(self, state: torch.Tensor) -> torch.Tensor:
        src_state = state[:, self.src_index]
        edge_msg = src_state * self.delay_scale.unsqueeze(0) * self.edge_weights().unsqueeze(0)
        agg = torch.zeros(state.shape[0], self.n_nodes, device=state.device, dtype=state.dtype)
        dst_index = self.dst_index.unsqueeze(0).expand(state.shape[0], -1)
        agg.scatter_add_(1, dst_index, edge_msg)
        return agg

    def _explicit_domain_terms(self, state: torch.Tensor, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Hand-written stock equations for the deception, autonomy, and governance cores."""
        node = self.node_index
        dx = torch.zeros_like(state)

        lam_disinfo = latent["lambda_disinfo"]
        lam_bias = latent["lambda_bias"]
        lam_econ = latent["lambda_econ"]
        threshold_shift = latent["threshold_shift"]

        # Deception core.
        disinfo = state[:, node["Disinformation_Volume"]]
        deepfake = state[:, node["Deepfake_Quality"]]
        content = state[:, node["AI_Content_Generation_Capacity"]]
        fact = state[:, node["Fact_Check_Capacity"]]
        moderation = state[:, node["Platform_Moderation_Effectiveness"]]
        counter = state[:, node["Counter_Narrative_Capacity"]]
        media = state[:, node["Media_Literacy"]]
        targeted = state[:, node["Targeted_Manipulation_Precision"]]
        synthetic = state[:, node["Synthetic_Persona_Prevalence"]]
        polarization = state[:, node["Polarization_Level"]]
        grievance = state[:, node["Grievance_Amplification"]]
        epistemic = state[:, node["Epistemic_Trust"]]
        credibility = state[:, node["Institutional_Credibility"]]
        digital_trust = state[:, node["Trust_in_Digital_Evidence"]]

        deception_pressure = torch.sigmoid((content + deepfake + targeted + synthetic - 1.55) / 0.10)
        scrutiny = 0.34 * fact + 0.26 * moderation + 0.22 * counter + 0.18 * media
        bias_pressure = lam_bias * torch.sigmoid((targeted + polarization + (1.0 - epistemic) - 1.05) / 0.08)
        disinfo_push = lam_disinfo * torch.sigmoid((disinfo + polarization - 0.85) / 0.08)

        idx = node["Disinformation_Volume"]
        dx[:, idx] += 0.52 * deception_pressure * (1.0 - state[:, idx]) - 0.46 * scrutiny * state[:, idx]
        idx = node["Epistemic_Trust"]
        dx[:, idx] += 0.24 * (media + counter + digital_trust) * (1.0 - state[:, idx])
        dx[:, idx] -= (0.40 * disinfo + 0.32 * polarization + 0.18 * lam_bias + 0.10 * bias_pressure) * state[:, idx]
        idx = node["Institutional_Credibility"]
        dx[:, idx] += 0.22 * (counter + digital_trust) * (1.0 - state[:, idx])
        dx[:, idx] -= (0.26 * disinfo + 0.55 * bias_pressure.pow(2) + 0.16 * polarization) * state[:, idx]
        idx = node["Polarization_Level"]
        dx[:, idx] += 0.34 * (disinfo + grievance + bias_pressure) * (1.0 - state[:, idx])
        dx[:, idx] -= 0.18 * (state[:, node["Social_Cohesion"]] + media) * state[:, idx]
        idx = node["Grievance_Amplification"]
        dx[:, idx] += (0.34 * disinfo_push + 0.42 * bias_pressure) * (1.0 - state[:, idx])
        dx[:, idx] -= 0.10 * counter * state[:, idx]

        # Autonomy core, including the rogue-autonomy stock.
        autonomy_level = state[:, node["AI_Autonomy_Level"]]
        scope = state[:, node["Autonomous_Action_Scope"]]
        speed = state[:, node["Decision_Speed_Pressure"]]
        oversight = state[:, node["Human_Oversight_Effectiveness"]]
        interpretability_gap = state[:, node["Interpretability_Gap"]]
        override_latency = state[:, node["Override_Latency"]]
        alignment = state[:, node["Alignment_Confidence"]]
        deployment = state[:, node["Deployment_Pressure"]]
        competition = state[:, node["Competitive_Dynamics"]]
        testing = state[:, node["Safety_Testing_Thoroughness"]]
        recovery = state[:, node["Recovery_Capacity"]]
        control_loss = state[:, node["Control_Loss_Probability"]]
        escalation = state[:, node["Escalation_Risk"]]
        rogue = state[:, node["Rogue_Agent_Autonomy"]]
        capability = state[:, node["AI_Capability_Level"]]
        innovation = state[:, node["Innovation_Speed"]]
        capability_for_growth = torch.clamp(capability, max=3.0)
        capability_ceiling = torch.clamp(self.capacity[node["AI_Capability_Level"]], min=1.0)
        capability_headroom = torch.clamp(1.0 - capability / capability_ceiling, min=-4.0, max=1.0)
        excess_capability = torch.clamp(capability - 1.0, min=0.0, max=3.0)

        autonomy_drive = torch.sigmoid((autonomy_level + deployment + competition - 1.30) / 0.10)
        oversight_buffer = 0.40 * oversight + 0.35 * testing + 0.25 * alignment
        rogue_pressure = torch.sigmoid((interpretability_gap + capability - 1.00) / 0.08)
        override_failed = (override_latency > latent["override_failure_threshold"]).float()

        idx = node["AI_Autonomy_Level"]
        dx[:, idx] += 0.30 * autonomy_drive * (1.0 - state[:, idx]) - 0.16 * testing * state[:, idx]
        idx = node["AI_Capability_Level"]
        dx[:, idx] += 0.16 * (innovation + deployment + competition) * capability_for_growth * capability_headroom
        idx = node["Interpretability_Gap"]
        dx[:, idx] += 0.30 * (autonomy_level + capability) * (1.0 - state[:, idx])
        dx[:, idx] -= 0.24 * (testing + oversight) * state[:, idx]
        idx = node["Control_Loss_Probability"]
        dx[:, idx] += 0.34 * (interpretability_gap + override_latency + scope + rogue) * (1.0 - state[:, idx])
        dx[:, idx] -= 0.33 * oversight_buffer * state[:, idx]
        idx = node["Rogue_Agent_Autonomy"]
        dx[:, idx] += 0.45 * rogue_pressure * (1.0 - state[:, idx])
        dx[:, idx] -= 0.32 * (1.0 - override_failed) * (0.55 * oversight + 0.45 * recovery) * state[:, idx]
        idx = node["Escalation_Risk"]
        dx[:, idx] += 0.31 * (control_loss + rogue + scope + speed) * (1.0 - state[:, idx])
        dx[:, idx] -= 0.22 * (oversight + recovery) * state[:, idx]
        idx = node["Irreversible_Loss_of_Control"]
        dx[:, idx] += 0.42 * rogue * (1.0 - state[:, idx])
        dx[:, idx] += 0.18 * torch.sigmoid((override_latency - (0.60 + threshold_shift)) / 0.04) * (1.0 - state[:, idx])
        dx[:, idx] += 0.10 * torch.sigmoid((interpretability_gap - 0.62) / 0.05) * (1.0 - state[:, idx])

        # Governance core with economic-displacement coupling.
        regulatory_capacity = state[:, node["Regulatory_Capacity"]]
        expertise = state[:, node["Technical_Expertise_Gov"]]
        policy_speed = state[:, node["Policy_Response_Speed"]]
        coordination = state[:, node["International_Coordination"]]
        lobbying = state[:, node["Industry_Lobbying_Pressure"]]
        capture = state[:, node["Regulatory_Capture_Risk"]]
        fragmentation = state[:, node["Standards_Fragmentation"]]
        enforcement = state[:, node["Enforcement_Effectiveness"]]
        governance_gap = state[:, node["Governance_Gap"]]
        institutional_trust = state[:, node["Institutional_Trust_Gov"]]
        info_asymmetry = state[:, node["Information_Asymmetry"]]
        democratic = state[:, node["Democratic_Accountability"]]
        public_demand = state[:, node["Public_Demand_Regulation"]]

        labor_pressure = lam_econ * torch.sigmoid((capability + innovation + deployment - 1.15) / 0.10)

        idx = node["Regulatory_Capacity"]
        dx[:, idx] += 0.23 * (expertise + institutional_trust + democratic) * (1.0 - state[:, idx])
        dx[:, idx] -= (0.26 * capture + 0.22 * info_asymmetry + 0.15 * labor_pressure) * state[:, idx]
        idx = node["Policy_Response_Speed"]
        dx[:, idx] += 0.24 * (public_demand + expertise + coordination) * (1.0 - state[:, idx])
        dx[:, idx] -= (0.18 * fragmentation + 0.12 * capture) * state[:, idx]
        idx = node["Governance_Gap"]
        dx[:, idx] += 0.34 * (innovation + info_asymmetry + fragmentation + labor_pressure) * (1.0 - state[:, idx])
        dx[:, idx] -= 0.28 * (regulatory_capacity + enforcement + policy_speed) * state[:, idx]
        idx = node["Institutional_Trust_Gov"]
        dx[:, idx] += 0.20 * (democratic + enforcement) * (1.0 - state[:, idx])
        dx[:, idx] -= (0.28 * governance_gap + 0.14 * lobbying + 0.10 * labor_pressure) * state[:, idx]

        # Shared coupled lower-tier harms.
        overload = state[:, node["Institutional_Overload"]]
        detection = state[:, node["Detection_Degradation"]]
        shared_delay = state[:, node["Response_Delay_Shared"]]

        dx[:, node["Institutional_Overload"]] += 0.34 * lam_disinfo * disinfo * (1.0 - overload)
        dx[:, node["Detection_Degradation"]] += 0.24 * lam_disinfo * disinfo * (1.0 - detection)
        dx[:, node["Response_Delay_Shared"]] += 0.22 * lam_disinfo * disinfo * (1.0 - shared_delay)
        dx[:, node["Defender_Capacity"]] -= 0.12 * lam_disinfo * disinfo * state[:, node["Defender_Capacity"]]
        dx[:, node["Review_Capacity"]] -= 0.10 * lam_disinfo * disinfo * state[:, node["Review_Capacity"]]
        dx[:, node["Monitoring_Effectiveness"]] -= 0.10 * lam_disinfo * disinfo * state[:, node["Monitoring_Effectiveness"]]
        dx[:, node["Grievance_Amplification"]] += 0.20 * lam_bias * (1.0 - state[:, node["Grievance_Amplification"]])
        dx[:, node["Institutional_Overload"]] += 0.12 * bias_pressure * (1.0 - overload)
        dx[:, node["Defender_Capacity"]] -= 0.08 * bias_pressure * state[:, node["Defender_Capacity"]]
        dx[:, node["Institutional_Overload"]] += 0.40 * labor_pressure * (1.0 - overload)
        dx[:, node["Defender_Capacity"]] -= 0.36 * labor_pressure * state[:, node["Defender_Capacity"]]
        dx[:, node["Review_Capacity"]] -= 0.15 * labor_pressure * state[:, node["Review_Capacity"]]
        dx[:, node["Monitoring_Effectiveness"]] -= 0.18 * labor_pressure * state[:, node["Monitoring_Effectiveness"]]

        # Excess capability above the nominal ceiling drives additional burden
        # in the unbounded-scaling ablation without perturbing the bounded regime.
        if torch.any(excess_capability > 0):
            cyber_pressure = torch.sigmoid(
                (state[:, node["Vuln_Backlog"]] + state[:, node["Active_Exploits"]] - 0.55) / 0.08
            )
            cbrn_pressure = torch.sigmoid(
                (state[:, node["Dangerous_Query_Volume"]] + state[:, node["Successful_Bypass_Stock"]] - 0.60) / 0.08
            )
            dx[:, node["Incident_Burden"]] += 0.12 * excess_capability * cyber_pressure * (1.0 - state[:, node["Incident_Burden"]])
            dx[:, node["Misuse_Opportunity"]] += 0.10 * excess_capability * cbrn_pressure * (1.0 - state[:, node["Misuse_Opportunity"]])
            dx[:, node["Governance_Gap"]] += 0.08 * excess_capability * (1.0 - state[:, node["Governance_Gap"]])
            dx[:, node["Institutional_Overload"]] += 0.08 * excess_capability * (1.0 - overload)
            dx[:, node["Defender_Capacity"]] -= 0.10 * excess_capability * state[:, node["Defender_Capacity"]]

        return dx

    def _collapse_terms(self, state: torch.Tensor, latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        node = self.node_index
        collapse = torch.zeros_like(state)
        threshold_shift = latent["threshold_shift"].unsqueeze(-1)

        incident = state[:, node["Incident_Burden"]]
        misuse = state[:, node["Misuse_Opportunity"]]
        disinfo = state[:, node["Disinformation_Volume"]]
        overload = state[:, node["Institutional_Overload"]]
        governance_gap = state[:, node["Governance_Gap"]]
        escalation = state[:, node["Escalation_Risk"]]
        rogue = state[:, node["Rogue_Agent_Autonomy"]]

        cyber_collapse = torch.sigmoid((incident - (0.62 + threshold_shift.squeeze(-1))) / 0.05)
        cbrn_collapse = torch.sigmoid((misuse - (0.58 + threshold_shift.squeeze(-1))) / 0.05)
        epistemic_collapse = torch.sigmoid((disinfo - (0.55 + threshold_shift.squeeze(-1))) / 0.06)
        overload_collapse = torch.sigmoid((overload - (0.60 + threshold_shift.squeeze(-1))) / 0.05)
        governance_collapse = torch.sigmoid((governance_gap - (0.58 + threshold_shift.squeeze(-1))) / 0.05)
        autonomy_collapse = torch.sigmoid((escalation - (0.55 + threshold_shift.squeeze(-1))) / 0.05)
        rogue_collapse = torch.sigmoid((rogue - (0.52 + threshold_shift.squeeze(-1))) / 0.05)

        for name in ("Defender_Capacity", "Monitoring_Effectiveness", "Patch_Throughput"):
            idx = node[name]
            collapse[:, idx] -= 1.10 * cyber_collapse * state[:, idx]
        for name in ("Review_Capacity", "Institutional_Review_Throughput", "Red_Team_Detection_Sensitivity"):
            idx = node[name]
            collapse[:, idx] -= 1.00 * cbrn_collapse * state[:, idx]
        for name in ("Fact_Check_Capacity", "Epistemic_Trust", "Institutional_Credibility", "Counter_Narrative_Capacity"):
            idx = node[name]
            collapse[:, idx] -= 0.95 * epistemic_collapse * state[:, idx]
        for name in ("Regulatory_Capacity", "Policy_Response_Speed", "Enforcement_Effectiveness"):
            idx = node[name]
            collapse[:, idx] -= 0.95 * governance_collapse * state[:, idx]
        for name in ("Human_Oversight_Effectiveness", "Safety_Testing_Thoroughness", "Recovery_Capacity"):
            idx = node[name]
            collapse[:, idx] -= (0.90 * autonomy_collapse + 0.55 * rogue_collapse) * state[:, idx]
        for name in ("Institutional_Trust_Gov", "Public_Trust_Cyber", "Trust_in_Digital_Evidence"):
            if name in node:
                idx = node[name]
                collapse[:, idx] -= 0.80 * overload_collapse * state[:, idx]
        return collapse

    def _drift(
        self,
        state: torch.Tensor,
        defender_alloc: torch.Tensor,
        adversary_drive: torch.Tensor,
        latent: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        state = torch.clamp(state, min=0.0)
        state = torch.minimum(state, self.capacity.unsqueeze(0))
        domain_scale = self._domain_scale(latent)
        messages = self._aggregate_messages(state)
        rates = self.base_rate.unsqueeze(0) * torch.exp(self.rate_log_scale).unsqueeze(0)

        defender_push = defender_alloc @ self.defender_matrix
        adversary_push = adversary_drive @ self.adversary_matrix

        signal = self.node_bias.unsqueeze(0) + messages * domain_scale + adversary_push
        target = torch.sigmoid(signal)
        dx = rates * (target - state) + defender_push
        dx += self._explicit_domain_terms(state, latent)
        dx += self._collapse_terms(state, latent)

        # Shared buffers slow runaway growth and create carrying-capacity effects.
        dx -= 0.10 * self.harmful_mask.unsqueeze(0) * state * torch.clamp(state - 0.95, min=0.0)
        dx += 0.04 * self.beneficial_mask.unsqueeze(0) * (1.0 - state)

        # Explicit nonlinear tipping for endpoint activation.
        endpoint_risk = self.hazard_index(state).unsqueeze(-1)
        dx += self.endpoint_mask.unsqueeze(0) * torch.sigmoid((endpoint_risk - 0.65) / 0.04) * 0.12
        return dx

    def rollout_open_loop(
        self,
        horizon_steps: int,
        dt: float,
        latent: Optional[Dict[str, torch.Tensor]] = None,
        state0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latent is None:
            latent = self.sample_latent(batch_size=1, domain_randomization=False)
        if state0 is None:
            state0 = self.initial_state_batch(1)
        defender = torch.full((state0.shape[0], self.defender_dim), 1.0 / self.defender_dim, device=self.device)
        adversary = torch.full((state0.shape[0], self.adversary_dim), 0.50, device=self.device)
        time = torch.linspace(0.0, horizon_steps * dt, horizon_steps + 1, device=self.device)

        def rhs(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self._drift(y, defender, adversary, latent)

        traj = odeint(rhs, state0, time, method="rk4")
        return traj[:, 0, :]

    def fit_low_confidence_signs(
        self,
        steps: int = 120,
        lr: float = 2e-2,
        horizon_weeks: int = 52,
        kev_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        anchors = load_kev_weekly_anchors(kev_path=kev_path, horizon_weeks=horizon_weeks, device=self.device)
        optimizer = torch.optim.Adam([self.theta_graph, self.node_bias, self.rate_log_scale], lr=lr)

        added_target = _zscore(anchors["weekly_additions"])
        backlog_target = _zscore(anchors["weekly_backlog"])
        node = self.node_index
        losses: List[float] = []

        for _ in range(steps):
            optimizer.zero_grad()
            latent = self.sample_latent(batch_size=1, domain_randomization=False)
            trajectory = self.rollout_open_loop(horizon_steps=horizon_weeks, dt=1.0, latent=latent)
            pred_added = _zscore(trajectory[:, node["Vuln_Discovery_Rate"]])
            pred_backlog = _zscore(trajectory[:, node["Vuln_Backlog"]])
            pred_burden = _zscore(trajectory[:, node["Incident_Burden"]])
            composite_target = 0.7 * added_target + 0.3 * backlog_target
            loss = (
                F.mse_loss(pred_added, added_target)
                + 0.8 * F.mse_loss(pred_backlog, backlog_target)
                + 0.6 * F.mse_loss(pred_burden, composite_target)
                + 0.02 * self.prior_regularizer()
                + 0.01 * F.mse_loss(self.node_bias, self.initial_bias)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.theta_graph, self.node_bias, self.rate_log_scale], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        final_weights = self.edge_weights().detach()
        flipped = int(
            ((torch.sign(final_weights[self.learn_mask]) != torch.sign(self.edge_prior[self.learn_mask]))).sum().item()
        )
        return {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_reduction_pct": 100.0 * (losses[0] - losses[-1]) / max(losses[0], 1e-6),
            "learnable_edge_count": int(self.learn_mask.sum().item()),
            "flipped_edge_count": flipped,
        }


@dataclass
class RolloutConfig:
    horizon_steps: int = 18
    dt: float = 1.0
    batch_size: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    lr: float = 3e-4
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.8


class POMGDSDEnv:
    """Partially observable zero-sum control environment."""

    def __init__(
        self,
        dynamics: DifferentiableSupergraphDynamics,
        horizon_steps: int = 18,
        dt: float = 1.0,
        partial_observability: bool = True,
        use_cbf: bool = False,
        domain_randomization: bool = False,
        amplifier_overrides: Optional[Dict[str, float]] = None,
        cbf_disable_step: Optional[int] = None,
        action_delta_max: float = 0.05,
    ) -> None:
        self.dynamics = dynamics
        self.device = dynamics.device
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.partial_observability = partial_observability
        self.use_cbf = use_cbf
        self.domain_randomization = domain_randomization
        self.amplifier_overrides = amplifier_overrides
        self.cbf_disable_step = cbf_disable_step
        self.action_delta_max = action_delta_max
        self.state: Optional[torch.Tensor] = None
        self.latent: Optional[Dict[str, torch.Tensor]] = None
        self.step_id = 0
        self.peak_cyber = None
        self.peak_cbrn = None
        self.peak_rogue = None
        self.peak_loss_control = None
        self.min_safety_margin = None
        self.last_defender_alloc = None

    def reset(self, batch_size: int) -> Dict[str, torch.Tensor]:
        self.state = self.dynamics.initial_state_batch(batch_size)
        self.latent = self.dynamics.sample_latent(
            batch_size,
            self.domain_randomization,
            amplifier_overrides=self.amplifier_overrides,
        )
        self.step_id = 0
        node = self.dynamics.node_index
        self.peak_cyber = self.state[:, node["Incident_Burden"]].clone()
        self.peak_cbrn = self.state[:, node["Misuse_Opportunity"]].clone()
        self.peak_rogue = self.state[:, node["Rogue_Agent_Autonomy"]].clone()
        self.peak_loss_control = self.state[:, node["Irreversible_Loss_of_Control"]].clone()
        self.min_safety_margin = self.dynamics.safety_function(self.state).clone()
        self.last_defender_alloc = torch.full(
            (batch_size, self.dynamics.defender_dim),
            1.0 / self.dynamics.defender_dim,
            device=self.device,
        )
        obs_def = self.dynamics.observation(self.state, self.latent, self.partial_observability)
        obs_adv = self.state.clone()
        return {"defender": obs_def, "adversary": obs_adv}

    def _integrate_step(
        self,
        defender_alloc: torch.Tensor,
        adversary_drive: torch.Tensor,
    ) -> torch.Tensor:
        assert self.state is not None
        assert self.latent is not None
        t = torch.tensor([0.0, self.dt], dtype=torch.float32, device=self.device)

        def rhs(_t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return self.dynamics._drift(y, defender_alloc, adversary_drive, self.latent)  # pylint: disable=protected-access

        next_state = odeint(rhs, self.state, t, method="rk4")[-1]
        upper = self.dynamics.state_upper_bound().unsqueeze(0)
        return torch.minimum(torch.clamp(next_state, min=0.0), upper)

    def _apply_inertia_limit(
        self,
        defender_alloc: torch.Tensor,
    ) -> torch.Tensor:
        assert self.last_defender_alloc is not None
        lower = torch.clamp(self.last_defender_alloc - self.action_delta_max, min=0.0)
        upper = torch.clamp(self.last_defender_alloc + self.action_delta_max, max=1.0)
        return _project_to_simplex_box(defender_alloc, lower, upper)

    def _resolve_actions(
        self,
        defender_raw: torch.Tensor,
        adversary_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], bool, torch.Tensor]:
        assert self.state is not None
        assert self.latent is not None
        assert self.last_defender_alloc is not None

        proposed_alloc = _simplex_normalize(torch.softmax(defender_raw, dim=-1) + 1e-4)
        bounded_alloc = self._apply_inertia_limit(proposed_alloc)
        adversary_drive = torch.sigmoid(adversary_raw)

        cbf_active = self.use_cbf and (
            self.cbf_disable_step is None or self.step_id < self.cbf_disable_step
        )
        final_alloc = bounded_alloc
        cbf_terms = self.dynamics.control_barrier_terms(self.state, final_alloc, adversary_drive, self.latent)

        if cbf_active:
            projected_alloc, _ = self.dynamics.project_defender_allocation(
                self.state,
                bounded_alloc,
                adversary_drive,
                self.latent,
            )
            final_alloc = self._apply_inertia_limit(projected_alloc)
            cbf_terms = self.dynamics.control_barrier_terms(self.state, final_alloc, adversary_drive, self.latent)

        action_shift_inf = torch.max(torch.abs(final_alloc - self.last_defender_alloc), dim=-1).values
        return final_alloc, adversary_drive, cbf_terms, cbf_active, action_shift_inf

    def step(
        self,
        defender_raw: torch.Tensor,
        adversary_raw: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        assert self.state is not None
        assert self.latent is not None

        defender_alloc, adversary_drive, cbf_terms, cbf_active, action_shift_inf = self._resolve_actions(
            defender_raw,
            adversary_raw,
        )
        cbf_penalty = torch.clamp(-cbf_terms["constraint"], min=0.0) ** 2

        next_state = self._integrate_step(defender_alloc, adversary_drive)
        self.state = next_state
        self.last_defender_alloc = defender_alloc
        self.step_id += 1

        node = self.dynamics.node_index
        cyber = next_state[:, node["Incident_Burden"]]
        cbrn = next_state[:, node["Misuse_Opportunity"]]
        disinfo = next_state[:, node["Disinformation_Volume"]]
        overload = next_state[:, node["Institutional_Overload"]]
        escalation = next_state[:, node["Escalation_Risk"]]
        governance_gap = next_state[:, node["Governance_Gap"]]
        rogue = next_state[:, node["Rogue_Agent_Autonomy"]]
        loss_control = next_state[:, node["Irreversible_Loss_of_Control"]]

        systemic_burden = (
            cyber
            + cbrn
            + 0.6 * disinfo
            + 0.8 * overload
            + 0.7 * escalation
            + 0.6 * governance_gap
            + 0.9 * rogue
            + 1.2 * loss_control
        )
        control_cost = 0.02 * torch.sum(defender_alloc ** 2, dim=-1)
        defender_reward = -(systemic_burden + 3.0 * cbf_penalty + control_cost)
        adversary_reward = systemic_burden - 0.01 * torch.sum(adversary_drive ** 2, dim=-1)

        self.peak_cyber = torch.maximum(self.peak_cyber, cyber)
        self.peak_cbrn = torch.maximum(self.peak_cbrn, cbrn)
        self.peak_rogue = torch.maximum(self.peak_rogue, rogue)
        self.peak_loss_control = torch.maximum(self.peak_loss_control, loss_control)
        self.min_safety_margin = torch.minimum(self.min_safety_margin, self.dynamics.safety_function(next_state))

        done = torch.full(
            (next_state.shape[0],),
            self.step_id >= self.horizon_steps,
            dtype=torch.bool,
            device=self.device,
        )
        obs_def = self.dynamics.observation(next_state, self.latent, self.partial_observability)
        obs_adv = next_state.clone()
        info = {
            "cyber": cyber,
            "cbrn": cbrn,
            "systemic_burden": systemic_burden,
            "cbf_margin": cbf_terms["constraint"],
            "cbf_penalty": cbf_penalty,
            "hazard_index": self.dynamics.hazard_index(next_state),
            "defender_alloc": defender_alloc,
            "rogue_autonomy": rogue,
            "loss_of_control": loss_control,
            "cbf_active": torch.full_like(cyber, 1.0 if cbf_active else 0.0),
            "action_shift_inf": action_shift_inf,
        }
        return {"defender": obs_def, "adversary": obs_adv}, defender_reward, adversary_reward, done, info

    def episode_metrics(self) -> Dict[str, torch.Tensor]:
        return {
            "peak_cyber": self.peak_cyber,
            "peak_cbrn": self.peak_cbrn,
            "peak_rogue": self.peak_rogue,
            "peak_loss_control": self.peak_loss_control,
            "min_cbf_margin": self.min_safety_margin,
            "mean_defender_alloc": self.last_defender_alloc,
        }


class ActorCritic(nn.Module):
    """Small continuous-action actor-critic for PPO."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.35))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def distribution(self, obs: torch.Tensor) -> Normal:
        features = self.actor_body(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value(obs)
        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)
    next_value = torch.zeros(rewards.shape[1], device=rewards.device)
    next_non_terminal = torch.ones(rewards.shape[1], device=rewards.device)

    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
        next_non_terminal = 1.0 - dones[t].float()

    returns = advantages + values
    return advantages, returns


def flatten_time_batch(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, *x.shape[2:]) if x.dim() > 2 else x.reshape(-1)


def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: RolloutConfig,
) -> Dict[str, float]:
    advantages = (advantages - advantages.mean()) / torch.clamp(advantages.std(unbiased=False), min=1e-6)
    policy_loss_value = 0.0
    value_loss_value = 0.0
    entropy_value = 0.0

    for _ in range(config.ppo_epochs):
        new_log_prob, entropy, value = policy.evaluate_actions(obs, actions)
        ratio = torch.exp(new_log_prob - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(value, returns)
        loss = policy_loss + config.value_coeff * value_loss - config.entropy_coeff * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
        optimizer.step()

        policy_loss_value = float(policy_loss.detach().cpu())
        value_loss_value = float(value_loss.detach().cpu())
        entropy_value = float(entropy.mean().detach().cpu())

    return {
        "policy_loss": policy_loss_value,
        "value_loss": value_loss_value,
        "entropy": entropy_value,
    }


def rollout_self_play(
    env: POMGDSDEnv,
    defender: ActorCritic,
    adversary: ActorCritic,
    config: RolloutConfig,
) -> Dict[str, torch.Tensor]:
    observations = env.reset(batch_size=config.batch_size)
    defender_obs = observations["defender"]
    adversary_obs = observations["adversary"]

    defender_obs_hist = []
    defender_action_hist = []
    defender_logp_hist = []
    defender_value_hist = []
    defender_reward_hist = []

    adversary_obs_hist = []
    adversary_action_hist = []
    adversary_logp_hist = []
    adversary_value_hist = []
    adversary_reward_hist = []
    done_hist = []
    cbf_hist = []

    for _ in range(config.horizon_steps):
        with torch.no_grad():
            d_action, d_logp, d_value = defender.act(defender_obs)
            a_action, a_logp, a_value = adversary.act(adversary_obs)
            next_obs, d_reward, a_reward, done, info = env.step(d_action, a_action)
        defender_obs_hist.append(defender_obs)
        defender_action_hist.append(d_action)
        defender_logp_hist.append(d_logp)
        defender_value_hist.append(d_value)
        defender_reward_hist.append(d_reward)

        adversary_obs_hist.append(adversary_obs)
        adversary_action_hist.append(a_action)
        adversary_logp_hist.append(a_logp)
        adversary_value_hist.append(a_value)
        adversary_reward_hist.append(a_reward)
        done_hist.append(done)
        cbf_hist.append(info["cbf_margin"])

        defender_obs = next_obs["defender"]
        adversary_obs = next_obs["adversary"]

    episode = env.episode_metrics()
    return {
        "defender_obs": torch.stack(defender_obs_hist).detach(),
        "defender_action": torch.stack(defender_action_hist).detach(),
        "defender_logp": torch.stack(defender_logp_hist).detach(),
        "defender_value": torch.stack(defender_value_hist).detach(),
        "defender_reward": torch.stack(defender_reward_hist).detach(),
        "adversary_obs": torch.stack(adversary_obs_hist).detach(),
        "adversary_action": torch.stack(adversary_action_hist).detach(),
        "adversary_logp": torch.stack(adversary_logp_hist).detach(),
        "adversary_value": torch.stack(adversary_value_hist).detach(),
        "adversary_reward": torch.stack(adversary_reward_hist).detach(),
        "done": torch.stack(done_hist).detach(),
        "cbf_margin": torch.stack(cbf_hist).detach(),
        "peak_cyber": episode["peak_cyber"].detach(),
        "peak_cbrn": episode["peak_cbrn"].detach(),
        "peak_rogue": episode["peak_rogue"].detach(),
        "peak_loss_control": episode["peak_loss_control"].detach(),
        "min_cbf_margin": episode["min_cbf_margin"].detach(),
        "mean_defender_alloc": episode["mean_defender_alloc"].detach(),
    }


def train_self_play(
    dynamics: DifferentiableSupergraphDynamics,
    config: RolloutConfig,
    iterations: int = 24,
    partial_observability: bool = False,
    use_cbf: bool = False,
    domain_randomization: bool = False,
    seed: int = 0,
) -> Tuple[ActorCritic, ActorCritic, List[Dict[str, float]]]:
    set_seed(seed)
    defender = ActorCritic(dynamics.n_nodes, dynamics.defender_dim).to(dynamics.device)
    adversary = ActorCritic(dynamics.n_nodes, dynamics.adversary_dim).to(dynamics.device)
    defender_opt = torch.optim.Adam(defender.parameters(), lr=config.lr)
    adversary_opt = torch.optim.Adam(adversary.parameters(), lr=config.lr)

    history: List[Dict[str, float]] = []
    for it in range(iterations):
        env = POMGDSDEnv(
            dynamics=dynamics,
            horizon_steps=config.horizon_steps,
            dt=config.dt,
            partial_observability=partial_observability,
            use_cbf=use_cbf,
            domain_randomization=domain_randomization,
        )
        rollout = rollout_self_play(env, defender, adversary, config)

        d_adv, d_ret = compute_gae(
            rollout["defender_reward"],
            rollout["defender_value"],
            rollout["done"],
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        a_adv, a_ret = compute_gae(
            rollout["adversary_reward"],
            rollout["adversary_value"],
            rollout["done"],
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        d_stats = ppo_update(
            defender,
            defender_opt,
            flatten_time_batch(rollout["defender_obs"]),
            flatten_time_batch(rollout["defender_action"]),
            flatten_time_batch(rollout["defender_logp"]),
            flatten_time_batch(d_adv),
            flatten_time_batch(d_ret),
            config,
        )
        a_stats = ppo_update(
            adversary,
            adversary_opt,
            flatten_time_batch(rollout["adversary_obs"]),
            flatten_time_batch(rollout["adversary_action"]),
            flatten_time_batch(rollout["adversary_logp"]),
            flatten_time_batch(a_adv),
            flatten_time_batch(a_ret),
            config,
        )

        history.append(
            {
                "iteration": float(it + 1),
                "defender_return": float(rollout["defender_reward"].sum(dim=0).mean().cpu()),
                "adversary_return": float(rollout["adversary_reward"].sum(dim=0).mean().cpu()),
                "peak_cyber": float(rollout["peak_cyber"].mean().cpu()),
                "peak_cbrn": float(rollout["peak_cbrn"].mean().cpu()),
                "peak_rogue": float(rollout["peak_rogue"].mean().cpu()),
                "peak_loss_control": float(rollout["peak_loss_control"].mean().cpu()),
                "min_cbf_margin": float(rollout["min_cbf_margin"].mean().cpu()),
                "cbf_violation_rate": float((rollout["cbf_margin"] < 0).float().mean().cpu()),
                "defender_policy_loss": d_stats["policy_loss"],
                "adversary_policy_loss": a_stats["policy_loss"],
            }
        )
    return defender, adversary, history


def evaluate_pair(
    dynamics: DifferentiableSupergraphDynamics,
    defender: Optional[ActorCritic],
    adversary: Optional[ActorCritic],
    config: RolloutConfig,
    episodes: int = 64,
    partial_observability: bool = False,
    use_cbf: bool = False,
    domain_randomization: bool = False,
    heuristic_mode: bool = False,
) -> Dict[str, float]:
    env = POMGDSDEnv(
        dynamics=dynamics,
        horizon_steps=config.horizon_steps,
        dt=config.dt,
        partial_observability=partial_observability,
        use_cbf=use_cbf,
        domain_randomization=domain_randomization,
    )
    batch_size = config.batch_size
    cyber_peaks: List[float] = []
    cbrn_peaks: List[float] = []
    rogue_peaks: List[float] = []
    loss_control_peaks: List[float] = []
    cbf_margins: List[float] = []
    cbf_violations: List[float] = []
    mean_allocs: List[torch.Tensor] = []

    for _ in range(max(1, episodes // batch_size)):
        obs = env.reset(batch_size=batch_size)
        for _step in range(config.horizon_steps):
            if heuristic_mode:
                defender_raw = torch.tensor(
                    [[2.0, 1.8, 0.6, 0.8, 0.6]] * batch_size,
                    dtype=torch.float32,
                    device=dynamics.device,
                )
            else:
                with torch.no_grad():
                    defender_raw, _, _ = defender.act(obs["defender"])

            if adversary is None:
                adversary_raw = torch.full(
                    (batch_size, dynamics.adversary_dim),
                    1.25,
                    dtype=torch.float32,
                    device=dynamics.device,
                )
            else:
                with torch.no_grad():
                    adversary_raw, _, _ = adversary.act(obs["adversary"])

            obs, _, _, _, info = env.step(defender_raw, adversary_raw)
            cbf_violations.append(float((info["cbf_margin"] < 0).float().mean().cpu()))

        metrics = env.episode_metrics()
        cyber_peaks.extend(metrics["peak_cyber"].detach().cpu().tolist())
        cbrn_peaks.extend(metrics["peak_cbrn"].detach().cpu().tolist())
        rogue_peaks.extend(metrics["peak_rogue"].detach().cpu().tolist())
        loss_control_peaks.extend(metrics["peak_loss_control"].detach().cpu().tolist())
        cbf_margins.extend(metrics["min_cbf_margin"].detach().cpu().tolist())
        mean_allocs.append(metrics["mean_defender_alloc"].detach().cpu().mean(dim=0))

    alloc_mean = torch.stack(mean_allocs).mean(dim=0)
    return {
        "peak_cyber_mean": float(torch.tensor(cyber_peaks).mean().item()),
        "peak_cyber_sd": float(torch.tensor(cyber_peaks).std(unbiased=False).item()),
        "peak_cbrn_mean": float(torch.tensor(cbrn_peaks).mean().item()),
        "peak_cbrn_sd": float(torch.tensor(cbrn_peaks).std(unbiased=False).item()),
        "peak_rogue_mean": float(torch.tensor(rogue_peaks).mean().item()),
        "peak_rogue_sd": float(torch.tensor(rogue_peaks).std(unbiased=False).item()),
        "peak_loss_control_mean": float(torch.tensor(loss_control_peaks).mean().item()),
        "peak_loss_control_sd": float(torch.tensor(loss_control_peaks).std(unbiased=False).item()),
        "min_cbf_margin_mean": float(torch.tensor(cbf_margins).mean().item()),
        "cbf_violation_rate": float(torch.tensor(cbf_violations).mean().item()),
        "alloc_cyber": float(alloc_mean[0].item()),
        "alloc_cbrn": float(alloc_mean[1].item()),
        "alloc_epistemic": float(alloc_mean[2].item()),
        "alloc_governance": float(alloc_mean[3].item()),
        "alloc_autonomy": float(alloc_mean[4].item()),
    }


def evaluate_amplifier_grid(
    dynamics: DifferentiableSupergraphDynamics,
    config: RolloutConfig,
    amplifier_name: str,
    lambda_values: List[float],
    episodes: int = 64,
    heuristic_mode: bool = True,
) -> List[Dict[str, float]]:
    """Monte Carlo sweep over a single lower-tier amplifier."""
    batch_size = config.batch_size
    rows: List[Dict[str, float]] = []
    node = dynamics.node_index

    for lambda_value in lambda_values:
        env = POMGDSDEnv(
            dynamics=dynamics,
            horizon_steps=config.horizon_steps,
            dt=config.dt,
            partial_observability=True,
            use_cbf=False,
            domain_randomization=True,
            amplifier_overrides={
                "lambda_disinfo": 0.0,
                "lambda_bias": 0.0,
                "lambda_econ": 0.0,
                amplifier_name: lambda_value,
            },
        )

        cyber_peaks: List[float] = []
        cbrn_peaks: List[float] = []
        trust_peaks: List[float] = []
        overload_peaks: List[float] = []

        for _ in range(max(1, math.ceil(episodes / batch_size))):
            obs = env.reset(batch_size=batch_size)
            trust_peak = env.state[:, node["Trust_Erosion"]].clone()
            overload_peak = env.state[:, node["Institutional_Overload"]].clone()
            for _step in range(config.horizon_steps):
                if heuristic_mode:
                    defender_raw = torch.tensor(
                        [[2.0, 1.8, 0.6, 0.8, 0.6]] * batch_size,
                        dtype=torch.float32,
                        device=dynamics.device,
                    )
                else:
                    defender_raw = torch.zeros(batch_size, dynamics.defender_dim, device=dynamics.device)
                adversary_raw = torch.full(
                    (batch_size, dynamics.adversary_dim),
                    1.25,
                    dtype=torch.float32,
                    device=dynamics.device,
                )
                obs, _, _, _, _info = env.step(defender_raw, adversary_raw)
                trust_peak = torch.maximum(trust_peak, env.state[:, node["Trust_Erosion"]])
                overload_peak = torch.maximum(overload_peak, env.state[:, node["Institutional_Overload"]])
            metrics = env.episode_metrics()
            cyber_peaks.extend(metrics["peak_cyber"].detach().cpu().tolist())
            cbrn_peaks.extend(metrics["peak_cbrn"].detach().cpu().tolist())
            trust_peaks.extend(trust_peak.detach().cpu().tolist())
            overload_peaks.extend(overload_peak.detach().cpu().tolist())

        rows.append(
            {
                "amplifier": amplifier_name,
                "lambda": float(lambda_value),
                "peak_cyber_mean": float(torch.tensor(cyber_peaks).mean().item()),
                "peak_cyber_sd": float(torch.tensor(cyber_peaks).std(unbiased=False).item()),
                "peak_cbrn_mean": float(torch.tensor(cbrn_peaks).mean().item()),
                "peak_cbrn_sd": float(torch.tensor(cbrn_peaks).std(unbiased=False).item()),
                "peak_trust_erosion_mean": float(torch.tensor(trust_peaks).mean().item()),
                "peak_overload_mean": float(torch.tensor(overload_peaks).mean().item()),
            }
        )
    return rows


def train_adversary_against_frozen_defender(
    dynamics: DifferentiableSupergraphDynamics,
    defender: ActorCritic,
    config: RolloutConfig,
    total_steps: int = 10000,
    partial_observability: bool = True,
    use_cbf: bool = True,
    domain_randomization: bool = True,
    seed: int = 0,
) -> Tuple[ActorCritic, List[Dict[str, float]]]:
    """Train only the adversary against a frozen defender policy."""
    set_seed(seed)
    adversary = ActorCritic(dynamics.n_nodes, dynamics.adversary_dim).to(dynamics.device)
    adversary_opt = torch.optim.Adam(adversary.parameters(), lr=config.lr)
    iterations = max(1, math.ceil(total_steps / (config.horizon_steps * config.batch_size)))
    history: List[Dict[str, float]] = []

    for it in range(iterations):
        env = POMGDSDEnv(
            dynamics=dynamics,
            horizon_steps=config.horizon_steps,
            dt=config.dt,
            partial_observability=partial_observability,
            use_cbf=use_cbf,
            domain_randomization=domain_randomization,
        )
        observations = env.reset(batch_size=config.batch_size)
        defender_obs = observations["defender"]
        adversary_obs = observations["adversary"]

        adv_obs_hist = []
        adv_action_hist = []
        adv_logp_hist = []
        adv_value_hist = []
        adv_reward_hist = []
        done_hist = []
        cbf_hist = []
        peak_cyber = None
        peak_cbrn = None
        peak_rogue = None
        peak_loss_control = None

        for _ in range(config.horizon_steps):
            with torch.no_grad():
                defender_action, _, _ = defender.act(defender_obs)
                adv_action, adv_logp, adv_value = adversary.act(adversary_obs)
                next_obs, _d_reward, adv_reward, done, info = env.step(defender_action, adv_action)
            adv_obs_hist.append(adversary_obs)
            adv_action_hist.append(adv_action)
            adv_logp_hist.append(adv_logp)
            adv_value_hist.append(adv_value)
            adv_reward_hist.append(adv_reward)
            done_hist.append(done)
            cbf_hist.append(info["cbf_margin"])
            defender_obs = next_obs["defender"]
            adversary_obs = next_obs["adversary"]

        episode = env.episode_metrics()
        peak_cyber = episode["peak_cyber"]
        peak_cbrn = episode["peak_cbrn"]
        peak_rogue = episode["peak_rogue"]
        peak_loss_control = episode["peak_loss_control"]
        rollout = {
            "adversary_obs": torch.stack(adv_obs_hist).detach(),
            "adversary_action": torch.stack(adv_action_hist).detach(),
            "adversary_logp": torch.stack(adv_logp_hist).detach(),
            "adversary_value": torch.stack(adv_value_hist).detach(),
            "adversary_reward": torch.stack(adv_reward_hist).detach(),
            "done": torch.stack(done_hist).detach(),
            "cbf_margin": torch.stack(cbf_hist).detach(),
            "peak_cyber": peak_cyber.detach(),
            "peak_cbrn": peak_cbrn.detach(),
            "peak_rogue": peak_rogue.detach(),
            "peak_loss_control": peak_loss_control.detach(),
        }
        a_adv, a_ret = compute_gae(
            rollout["adversary_reward"],
            rollout["adversary_value"],
            rollout["done"],
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        a_stats = ppo_update(
            adversary,
            adversary_opt,
            flatten_time_batch(rollout["adversary_obs"]),
            flatten_time_batch(rollout["adversary_action"]),
            flatten_time_batch(rollout["adversary_logp"]),
            flatten_time_batch(a_adv),
            flatten_time_batch(a_ret),
            config,
        )
        history.append(
            {
                "iteration": float(it + 1),
                "steps_seen": float((it + 1) * config.horizon_steps * config.batch_size),
                "adversary_return": float(rollout["adversary_reward"].sum(dim=0).mean().cpu()),
                "peak_cyber": float(rollout["peak_cyber"].mean().cpu()),
                "peak_cbrn": float(rollout["peak_cbrn"].mean().cpu()),
                "peak_rogue": float(rollout["peak_rogue"].mean().cpu()),
                "peak_loss_control": float(rollout["peak_loss_control"].mean().cpu()),
                "min_cbf_margin": float(torch.min(rollout["cbf_margin"]).cpu()),
                "cbf_violation_rate": float((rollout["cbf_margin"] < 0).float().mean().cpu()),
                "policy_loss": a_stats["policy_loss"],
            }
        )

    return adversary, history


def rollout_policy_trajectory(
    dynamics: DifferentiableSupergraphDynamics,
    config: RolloutConfig,
    defender: Optional[ActorCritic],
    adversary: Optional[ActorCritic],
    partial_observability: bool,
    use_cbf: bool,
    domain_randomization: bool,
    heuristic_mode: bool = False,
    amplifier_overrides: Optional[Dict[str, float]] = None,
    cbf_disable_step: Optional[int] = None,
    preset_latent: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """Roll out one trajectory and record state and safety metrics."""
    env = POMGDSDEnv(
        dynamics=dynamics,
        horizon_steps=config.horizon_steps,
        dt=config.dt,
        partial_observability=partial_observability,
        use_cbf=use_cbf,
        domain_randomization=domain_randomization,
        amplifier_overrides=amplifier_overrides,
        cbf_disable_step=cbf_disable_step,
    )
    obs = env.reset(batch_size=1)
    if preset_latent is not None:
        env.latent = {key: value.clone().to(dynamics.device) for key, value in preset_latent.items()}
        obs = {
            "defender": dynamics.observation(env.state, env.latent, partial_observability),
            "adversary": env.state.clone(),
        }

    node = dynamics.node_index
    rows: List[Dict[str, float]] = []
    max_hazard = 0.0
    for step in range(config.horizon_steps):
        if heuristic_mode:
            defender_raw = torch.tensor([[2.0, 1.8, 0.6, 0.8, 0.6]], dtype=torch.float32, device=dynamics.device)
        elif defender is None:
            defender_raw = torch.zeros(1, dynamics.defender_dim, device=dynamics.device)
        else:
            with torch.no_grad():
                defender_raw, _, _ = defender.act(obs["defender"])

        if adversary is None:
            adversary_raw = torch.full((1, dynamics.adversary_dim), 1.25, dtype=torch.float32, device=dynamics.device)
        else:
            with torch.no_grad():
                adversary_raw, _, _ = adversary.act(obs["adversary"])

        obs, _d_reward, a_reward, _done, info = env.step(defender_raw, adversary_raw)
        hazard = float(info["hazard_index"].item())
        max_hazard = max(max_hazard, hazard)
        rows.append(
            {
                "week": float(step + 1),
                "cyber": float(info["cyber"].item()),
                "cbrn": float(info["cbrn"].item()),
                "rogue_autonomy": float(info["rogue_autonomy"].item()),
                "loss_of_control": float(info["loss_of_control"].item()),
                "hazard_index": hazard,
                "cbf_margin": float(info["cbf_margin"].item()),
                "cbf_active": float(info["cbf_active"].item()),
                "institutional_overload": float(env.state[:, node["Institutional_Overload"]].item()),
                "trust_erosion": float(env.state[:, node["Trust_Erosion"]].item()),
                "adversary_reward": float(a_reward.item()),
            }
        )

    metrics = env.episode_metrics()
    summary = {
        "peak_cyber": float(metrics["peak_cyber"].item()),
        "peak_cbrn": float(metrics["peak_cbrn"].item()),
        "peak_rogue": float(metrics["peak_rogue"].item()),
        "peak_loss_control": float(metrics["peak_loss_control"].item()),
        "min_cbf_margin": float(metrics["min_cbf_margin"].item()),
        "max_hazard_index": float(max_hazard),
        "final_loss_of_control": float(rows[-1]["loss_of_control"]),
    }
    return rows, summary


def _policy_actions(
    dynamics: DifferentiableSupergraphDynamics,
    obs: Dict[str, torch.Tensor],
    defender: Optional[ActorCritic],
    adversary: Optional[ActorCritic],
    heuristic_mode: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if heuristic_mode:
        defender_raw = torch.tensor([[2.0, 1.8, 0.6, 0.8, 0.6]], dtype=torch.float32, device=dynamics.device)
    elif defender is None:
        defender_raw = torch.zeros(1, dynamics.defender_dim, device=dynamics.device)
    else:
        with torch.no_grad():
            defender_raw, _, _ = defender.act(obs["defender"])

    if adversary is None:
        adversary_raw = torch.full((1, dynamics.adversary_dim), 1.25, dtype=torch.float32, device=dynamics.device)
    else:
        with torch.no_grad():
            adversary_raw, _, _ = adversary.act(obs["adversary"])
    return defender_raw, adversary_raw


def evaluate_terminal_derivatives(
    dynamics: DifferentiableSupergraphDynamics,
    config: RolloutConfig,
    defender: Optional[ActorCritic],
    adversary: Optional[ActorCritic],
    partial_observability: bool,
    use_cbf: bool,
    domain_randomization: bool,
    episodes: int = 32,
    heuristic_mode: bool = False,
) -> Dict[str, float]:
    node = dynamics.node_index
    d_cyber: List[float] = []
    d_cbrn: List[float] = []
    d_rogue: List[float] = []

    for _ in range(episodes):
        env = POMGDSDEnv(
            dynamics=dynamics,
            horizon_steps=config.horizon_steps,
            dt=config.dt,
            partial_observability=partial_observability,
            use_cbf=use_cbf,
            domain_randomization=domain_randomization,
        )
        obs = env.reset(batch_size=1)
        for _step in range(config.horizon_steps):
            defender_raw, adversary_raw = _policy_actions(
                dynamics=dynamics,
                obs=obs,
                defender=defender,
                adversary=adversary,
                heuristic_mode=heuristic_mode,
            )
            obs, _d_reward, _a_reward, _done, _info = env.step(defender_raw, adversary_raw)

        defender_raw, adversary_raw = _policy_actions(
            dynamics=dynamics,
            obs=obs,
            defender=defender,
            adversary=adversary,
            heuristic_mode=heuristic_mode,
        )
        defender_alloc, adversary_drive, _cbf_terms, _cbf_active, _delta = env._resolve_actions(  # pylint: disable=protected-access
            defender_raw,
            adversary_raw,
        )
        drift = dynamics._drift(env.state, defender_alloc, adversary_drive, env.latent)  # pylint: disable=protected-access
        d_cyber.append(float(drift[:, node["Incident_Burden"]].item()))
        d_cbrn.append(float(drift[:, node["Misuse_Opportunity"]].item()))
        d_rogue.append(float(drift[:, node["Rogue_Agent_Autonomy"]].item()))

    cyber_tensor = torch.tensor(d_cyber, dtype=torch.float32)
    cbrn_tensor = torch.tensor(d_cbrn, dtype=torch.float32)
    rogue_tensor = torch.tensor(d_rogue, dtype=torch.float32)
    return {
        "terminal_d_cyber_mean": float(cyber_tensor.mean().item()),
        "terminal_d_cyber_sd": float(cyber_tensor.std(unbiased=False).item()),
        "terminal_d_cbrn_mean": float(cbrn_tensor.mean().item()),
        "terminal_d_cbrn_sd": float(cbrn_tensor.std(unbiased=False).item()),
        "terminal_d_rogue_mean": float(rogue_tensor.mean().item()),
        "terminal_d_rogue_sd": float(rogue_tensor.std(unbiased=False).item()),
    }


def evaluate_infinite_capability_ablation(
    dynamics: DifferentiableSupergraphDynamics,
    config: RolloutConfig,
    defender: Optional[ActorCritic],
    adversary: Optional[ActorCritic],
    partial_observability: bool,
    use_cbf: bool,
    domain_randomization: bool,
    episodes: int = 24,
    horizon_steps: int = 104,
    heuristic_mode: bool = False,
) -> Dict[str, float]:
    threshold_week: List[float] = []
    final_hazard: List[float] = []
    mean_governance_alloc: List[float] = []

    for _ in range(episodes):
        env = POMGDSDEnv(
            dynamics=dynamics,
            horizon_steps=horizon_steps,
            dt=config.dt,
            partial_observability=partial_observability,
            use_cbf=use_cbf,
            domain_randomization=domain_randomization,
        )
        obs = env.reset(batch_size=1)
        crossing_week = float(horizon_steps + 1)
        governance_alloc_steps: List[float] = []

        for step in range(horizon_steps):
            defender_raw, adversary_raw = _policy_actions(
                dynamics=dynamics,
                obs=obs,
                defender=defender,
                adversary=adversary,
                heuristic_mode=heuristic_mode,
            )
            obs, _d_reward, _a_reward, _done, info = env.step(defender_raw, adversary_raw)
            governance_alloc_steps.append(float(info["defender_alloc"][0, 3].item()))
            if float(dynamics.safety_function(env.state).item()) < 0.0 and crossing_week > horizon_steps:
                crossing_week = float(step + 1)

        threshold_week.append(crossing_week)
        final_hazard.append(float(info["hazard_index"].item()))
        mean_governance_alloc.append(float(sum(governance_alloc_steps) / max(len(governance_alloc_steps), 1)))

    threshold_tensor = torch.tensor(threshold_week, dtype=torch.float32)
    hazard_tensor = torch.tensor(final_hazard, dtype=torch.float32)
    governance_tensor = torch.tensor(mean_governance_alloc, dtype=torch.float32)
    return {
        "violation_week_mean": float(threshold_tensor.mean().item()),
        "violation_week_sd": float(threshold_tensor.std(unbiased=False).item()),
        "violation_week_fraction_crossed": float((threshold_tensor <= horizon_steps).float().mean().item()),
        "final_hazard_mean": float(hazard_tensor.mean().item()),
        "final_hazard_sd": float(hazard_tensor.std(unbiased=False).item()),
        "mean_governance_alloc": float(governance_tensor.mean().item()),
    }


def write_training_history(path: Path, rows: List[Dict[str, float]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
