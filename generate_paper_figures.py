import csv
import importlib.util
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import networkx as nx
import numpy as np
from scipy import stats


BASE = Path("/Users/nathanheath/Downloads")
OUTDIR = BASE / "paper_figures"
OUTDIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "figure.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "stock": "#5B8FF9",
    "flow": "#5AD8A6",
    "aux": "#C9CDD4",
    "edge": "#4E5969",
    "pos": "#117733",
    "neg": "#A61E4D",
    "endpoint": "#D94841",
    "shared": "#F6C85F",
}
CLUSTER_COLORS = {
    "cyber": "#D6E4FF",
    "cbrn": "#D8F3DC",
    "deception": "#FFF3BF",
    "autonomy": "#FFE8CC",
    "governance": "#F1F3F5",
}

MANUAL_LAYOUTS = {
    "cyber": {
        "AI_Capability_Level": (0.12, 0.38),
        "Open_Weight_Avail": (0.12, 0.62),
        "Attacker_Skill_Amp": (0.27, 0.62),
        "Detection_Evasion": (0.27, 0.22),
        "Attack_Sophistication": (0.27, 0.82),
        "Vuln_Discovery_Rate": (0.42, 0.80),
        "Exploit_Dev_Speed": (0.42, 0.54),
        "Vuln_Backlog": (0.58, 0.70),
        "Active_Exploits": (0.58, 0.52),
        "Incident_Rate": (0.74, 0.52),
        "Incident_Burden": (0.74, 0.34),
        "Investment_in_Defense": (0.52, 0.16),
        "Public_Trust_Cyber": (0.74, 0.16),
        "Patch_Throughput": (0.74, 0.86),
        "Defender_Capacity": (0.90, 0.72),
        "Response_Delay": (0.90, 0.50),
        "Monitoring_Effectiveness": (0.96, 0.58),
        "Safeguard_Update_Freq": (0.96, 0.28),
    },
    "cbrn": {
        "AI_Capability_Level": (0.08, 0.43),
        "Grievance_Level": (0.08, 0.72),
        "Synthesis_Lit_Accessibility": (0.08, 0.16),
        "Dangerous_Query_Volume": (0.29, 0.71),
        "Successful_Bypass_Rate": (0.33, 0.49),
        "Refusal_Accuracy": (0.28, 0.90),
        "Review_Capacity": (0.56, 0.82),
        "Institutional_Review_Throughput": (0.56, 0.62),
        "Successful_Bypass_Stock": (0.58, 0.43),
        "Red_Team_Detection_Sensitivity": (0.80, 0.90),
        "Access_Gate_Stringency": (0.82, 0.72),
        "Actionable_CBRN_Knowledge": (0.83, 0.43),
        "Public_Awareness_CBRN_Risk": (0.78, 0.16),
        "Governance_Response_Level": (1.07, 0.72),
        "Oversight_Delay": (1.07, 0.90),
        "Misuse_Opportunity": (1.07, 0.43),
    },
    "deception": {
        "Media_Literacy": (0.08, 0.88),
        "AI_Content_Generation_Capacity": (0.08, 0.54),
        "Detection_Tool_Sophistication": (0.24, 0.90),
        "Deepfake_Quality": (0.24, 0.74),
        "Synthetic_Persona_Prevalence": (0.24, 0.54),
        "Targeted_Manipulation_Precision": (0.24, 0.34),
        "Cross_Platform_Coordination": (0.24, 0.14),
        "Trust_in_Digital_Evidence": (0.40, 0.78),
        "Disinformation_Volume": (0.42, 0.54),
        "Fact_Check_Capacity": (0.42, 0.26),
        "Platform_Moderation_Effectiveness": (0.62, 0.54),
        "Epistemic_Trust": (0.64, 0.78),
        "Polarization_Level": (0.62, 0.26),
        "Counter_Narrative_Capacity": (0.80, 0.88),
        "Regulatory_Response_Deception": (0.82, 0.70),
        "Social_Cohesion": (0.82, 0.54),
        "Grievance_Amplification": (0.82, 0.26),
        "Institutional_Credibility": (0.98, 0.74),
        "Election_Integrity": (0.98, 0.50),
        "Recruitment_Effectiveness": (0.98, 0.22),
        "Radicalization_Pipeline_Speed": (0.98, 0.10),
    },
    "autonomy": {
        "Competitive_Dynamics": (0.08, 0.80),
        "Safety_Testing_Thoroughness": (0.08, 0.32),
        "Deployment_Pressure": (0.24, 0.80),
        "AI_Autonomy_Level": (0.24, 0.56),
        "Decision_Speed_Pressure": (0.24, 0.32),
        "Human_Oversight_Effectiveness": (0.50, 0.80),
        "Autonomous_Action_Scope": (0.50, 0.56),
        "Interpretability_Gap": (0.50, 0.32),
        "Alignment_Confidence": (0.50, 0.10),
        "Override_Latency": (0.76, 0.32),
        "Control_Loss_Probability": (0.76, 0.56),
        "Escalation_Risk": (0.92, 0.56),
        "Incident_Severity_Ceiling": (0.92, 0.80),
        "Recovery_Capacity": (0.92, 0.28),
    },
    "governance": {
        "Democratic_Accountability": (0.08, 0.88),
        "Public_Demand_Regulation": (0.08, 0.66),
        "Whistleblower_Protection": (0.08, 0.26),
        "Institutional_Trust_Gov": (0.26, 0.82),
        "Expert_Consensus_Level": (0.26, 0.56),
        "Technical_Expertise_Gov": (0.42, 0.56),
        "Regulatory_Capacity": (0.58, 0.56),
        "Policy_Response_Speed": (0.74, 0.76),
        "Safety_Standard_Stringency": (0.92, 0.76),
        "Enforcement_Effectiveness": (0.92, 0.56),
        "Governance_Gap": (0.92, 0.36),
        "Compliance_Cost": (0.92, 0.16),
        "Innovation_Speed": (0.74, 0.16),
        "Industry_Lobbying_Pressure": (0.58, 0.16),
        "Regulatory_Capture_Risk": (0.42, 0.16),
        "Information_Asymmetry": (0.58, 0.34),
        "International_Coordination": (0.42, 0.34),
        "Cross_Jurisdictional_Gaps": (0.74, 0.36),
        "Standards_Fragmentation": (0.74, 0.50),
    },
}

MANUAL_LOOP_MARKERS = {
    "cyber": {
        "R1": (0.20, 0.66),
        "R2": (0.50, 0.80),
        "R3": (0.87, 0.24),
        "R4": (0.91, 0.64),
        "B1": (0.60, 0.44),
        "B2": (0.88, 0.80),
        "B3": (0.96, 0.42),
    },
    "cbrn": {
        "R1": (0.12, 0.82),
        "R2": (0.79, 0.84),
        "R3": (0.92, 0.18),
        "B1": (0.39, 0.59),
        "B2": (0.67, 0.59),
        "B3": (0.98, 0.62),
    },
    "deception": {
        "R1": (0.18, 0.78),
        "R2": (0.34, 0.58),
        "R3": (0.54, 0.22),
        "R4": (0.78, 0.20),
        "R5": (0.82, 0.82),
        "B1": (0.56, 0.76),
        "B2": (0.68, 0.52),
    },
    "autonomy": {
        "R1": (0.16, 0.70),
        "R2": (0.40, 0.16),
        "B1": (0.60, 0.76),
        "B2": (0.82, 0.40),
    },
    "governance": {
        "R1": (0.18, 0.74),
        "R2": (0.50, 0.66),
        "R3": (0.82, 0.68),
        "R4": (0.72, 0.22),
        "B1": (0.24, 0.18),
        "B2": (0.54, 0.28),
        "B3": (0.84, 0.48),
        "B4": (0.88, 0.18),
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


graphs = load_module("graphs", BASE / "graphs.py")
all_graphs = graphs.get_all_graphs()


def clean_label(name: str) -> str:
    label = name.replace("_", " ")
    lines = textwrap.wrap(label, width=11, break_long_words=False)
    return "\n".join(lines)


def node_type(node: str) -> str:
    flow_tokens = (
        "Rate",
        "Speed",
        "Throughput",
        "Volume",
        "Pressure",
        "Frequency",
    )
    stock_tokens = (
        "Stock",
        "Backlog",
        "Burden",
        "Capacity",
        "Trust",
        "Level",
        "Gap",
        "Opportunity",
        "Integrity",
        "Cohesion",
        "Ceiling",
        "Risk",
        "Scope",
        "Probability",
    )
    if any(token in node for token in flow_tokens):
        return "flow"
    if any(token in node for token in stock_tokens):
        return "stock"
    return "aux"


def confidence_linestyle(conf: str) -> str:
    return {"H": "-", "M": "--", "L": ":"}.get(conf, "-")


def build_layout(G: nx.DiGraph, seed: int) -> dict:
    pos = nx.spring_layout(G, seed=seed, k=1.8 / math.sqrt(max(G.number_of_nodes(), 1)), iterations=400)
    # normalize to [0, 1]
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    out = {}
    for n, (x, y) in pos.items():
        nx_ = 0.1 + 0.8 * ((x - min_x) / (max_x - min_x + 1e-9))
        ny_ = 0.1 + 0.8 * ((y - min_y) / (max_y - min_y + 1e-9))
        out[n] = (nx_, ny_)
    return out


def get_domain_layout(name: str, G: nx.DiGraph, seed: int) -> dict:
    manual = MANUAL_LAYOUTS.get(name.lower())
    if manual and set(G.nodes()).issubset(set(manual)):
        return {node: manual[node] for node in G.nodes()}
    return build_layout(G, seed)


def node_dims(node: str) -> tuple[float, float]:
    label = clean_label(node).splitlines()
    width = 0.068 + 0.0055 * max(len(line) for line in label)
    width = min(width, 0.125)
    height = 0.048 + 0.017 * (len(label) - 1)
    if node_type(node) == "flow":
        height = max(height, 0.062)
    return width, height


def layout_limits(pos: dict, loop_points: list | None = None) -> tuple[float, float, float, float]:
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    if loop_points:
        xs.extend(marker[0] for marker, _, _ in loop_points)
        ys.extend(marker[1] for marker, _, _ in loop_points)
    x_lo = min(xs) - 0.10
    x_hi = max(xs) + 0.12
    y_lo = min(ys) - 0.10
    y_hi = max(ys) + 0.16
    return x_lo, x_hi, y_lo, y_hi


def edge_rad(u: str, v: str, pos: dict) -> float:
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    dx = x2 - x1
    dy = y2 - y1
    if dx < -0.02:
        return 0.24 if dy >= 0 else -0.24
    if abs(dx) < 0.06:
        return 0.20 if dy >= 0 else -0.20
    if abs(dy) > 0.22:
        return 0.10 if dy > 0 else -0.10
    return 0.0


def cycle_annotations(G: nx.DiGraph, want_r: int, want_b: int):
    try:
        cycles = list(nx.simple_cycles(G, length_bound=6))
    except TypeError:
        cycles = [c for c in nx.simple_cycles(G) if len(c) <= 6]
    cycles = sorted(cycles, key=len)
    r_cycles, b_cycles = [], []
    for cyc in cycles:
        neg = 0
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            if G[u][v].get("sign", 1) < 0:
                neg += 1
        if neg % 2 == 0 and len(r_cycles) < want_r:
            r_cycles.append(cyc)
        elif neg % 2 == 1 and len(b_cycles) < want_b:
            b_cycles.append(cyc)
        if len(r_cycles) >= want_r and len(b_cycles) >= want_b:
            break
    return r_cycles, b_cycles


def draw_loop_marker(ax, pos, label, color):
    x, y = pos
    circ = Circle((x, y), 0.03, facecolor="white", edgecolor=color, linewidth=1.2)
    ax.add_patch(circ)
    ax.text(x, y, label, ha="center", va="center", fontsize=7, color=color, fontweight="bold")


def draw_cld(name: str, G: nx.DiGraph, seed: int, want_r: int, want_b: int):
    figsize = (13.8, 10.0) if name.lower() in {"deception", "governance"} else (13.2, 9.6)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    pos = get_domain_layout(name, G, seed)

    loop_points = []
    manual_loops = MANUAL_LOOP_MARKERS.get(name.lower())
    if manual_loops:
        for label, marker in manual_loops.items():
            color = "#0B6E4F" if label.startswith("R") else "#7B2CBF"
            loop_points.append((marker, label, color))
    else:
        r_cycles, b_cycles = cycle_annotations(G, want_r, want_b)
        for prefix, cycles, color in [("R", r_cycles, "#0B6E4F"), ("B", b_cycles, "#7B2CBF")]:
            for idx, cyc in enumerate(cycles, start=1):
                xs = [pos[n][0] for n in cyc]
                ys = [pos[n][1] for n in cyc]
                marker = (sum(xs) / len(xs), sum(ys) / len(ys))
                loop_points.append((marker, f"{prefix}{idx}", color))

    x_lo, x_hi, y_lo, y_hi = layout_limits(pos, loop_points)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    for u, v, data in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        rad = edge_rad(u, v, pos)
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.1,
            linestyle=confidence_linestyle(data.get("confidence", "H")),
            color=COLORS["edge"],
            alpha=0.92,
            shrinkA=16,
            shrinkB=16,
            zorder=1,
        )
        ax.add_patch(arrow)
        mx = 0.52 * x1 + 0.48 * x2
        my = 0.52 * y1 + 0.48 * y2
        dx = x2 - x1
        dy = y2 - y1
        norm = math.hypot(dx, dy) + 1e-9
        offset_mag = 0.032 if norm < 0.30 else 0.024
        if rad != 0:
            offset_mag *= 1.25
            offset = offset_mag if rad > 0 else -offset_mag
        else:
            offset = offset_mag if dy >= 0 else -offset_mag
        mx += -dy / norm * offset
        my += dx / norm * offset
        sign_text = "+" if data.get("sign", 1) > 0 else "-"
        sign_color = COLORS["pos"] if sign_text == "+" else COLORS["neg"]
        ax.text(
            mx,
            my,
            sign_text,
            ha="center",
            va="center",
            fontsize=8.2,
            color=sign_color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.92),
            zorder=2,
        )
        if data.get("delay", 0):
            ax.text(
                mx + 0.028,
                my - 0.028,
                "||",
                ha="center",
                va="center",
                fontsize=7,
                color="#2B2D42",
                bbox=dict(boxstyle="round,pad=0.05", facecolor="white", edgecolor="none", alpha=0.75),
                zorder=2,
            )

    for node, (x, y) in pos.items():
        kind = node_type(node)
        width, height = node_dims(node)
        if kind == "flow":
            patch = Ellipse((x, y), width, height, facecolor=COLORS[kind], edgecolor="#334155", linewidth=1.0)
        else:
            patch = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                facecolor=COLORS[kind],
                edgecolor="#334155",
                linewidth=1.0,
            )
        ax.add_patch(patch)
        patch.set_zorder(3)
        ax.text(x, y, clean_label(node), ha="center", va="center", fontsize=6.0, zorder=4)
    for marker, label, color in loop_points:
        draw_loop_marker(ax, marker, label, color)

    legend_x = x_lo + 0.01
    ax.text(legend_x, y_hi - 0.02, f"{name} CLD", ha="left", va="top", fontsize=12, fontweight="bold")
    ax.text(legend_x, y_hi - 0.055, "Stocks = blue, flows = green, auxiliaries = gray", fontsize=7.2, ha="left")
    ax.text(legend_x, y_hi - 0.085, "Confidence: solid = high, dashed = medium, dotted = low; delays marked as ||", fontsize=7.2, ha="left")
    fig.savefig(OUTDIR / f"fig_cld_{name.lower()}.pdf")
    plt.close(fig)


def draw_supergraph():
    G = all_graphs["supergraph"]
    shared_nodes = all_graphs["shared_nodes"]
    endpoints = all_graphs["endpoint_nodes"]
    domain_nodes = {
        "cyber": list(all_graphs["cyber"].nodes()),
        "cbrn": list(all_graphs["cbrn"].nodes()),
        "deception": list(all_graphs["deception"].nodes()),
        "autonomy": list(all_graphs["autonomy"].nodes()),
        "governance": list(all_graphs["governance"].nodes()),
    }
    cluster_centers = {
        "cyber": (0.14, 0.78),
        "cbrn": (0.14, 0.28),
        "deception": (0.42, 0.88),
        "autonomy": (0.42, 0.18),
        "governance": (0.72, 0.52),
    }
    cluster_sizes = {
        "cyber": (0.24, 0.28),
        "cbrn": (0.24, 0.28),
        "deception": (0.24, 0.24),
        "autonomy": (0.24, 0.24),
        "governance": (0.24, 0.38),
    }

    pos = {}
    for domain, nodes in domain_nodes.items():
        sub = G.subgraph(nodes)
        seed = {"cyber": 11, "cbrn": 13, "deception": 17, "autonomy": 19, "governance": 23}[domain]
        sub_pos = build_layout(sub, seed)
        cx, cy = cluster_centers[domain]
        w, h = cluster_sizes[domain]
        for node, (x, y) in sub_pos.items():
            pos[node] = (cx - w / 2 + x * w, cy - h / 2 + y * h)
    hub_positions = {
        "Trust_Erosion": (0.53, 0.62),
        "Institutional_Overload": (0.54, 0.52),
        "Detection_Degradation": (0.53, 0.42),
        "Grievance_Amplification": (0.62, 0.62),
        "Response_Delay_Shared": (0.63, 0.42),
    }
    pos.update(hub_positions)
    endpoint_positions = {
        "Mass_Casualty_Opportunity": (0.90, 0.72),
        "Critical_Infrastructure_Destabilization": (0.90, 0.58),
        "Democratic_Process_Subversion": (0.90, 0.44),
        "Catastrophic_Public_Harm": (0.90, 0.30),
    }
    pos.update(endpoint_positions)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for domain, (cx, cy) in cluster_centers.items():
        w, h = cluster_sizes[domain]
        patch = FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=CLUSTER_COLORS[domain],
            edgecolor="#94A3B8",
            linewidth=1.2,
            alpha=0.65,
        )
        ax.add_patch(patch)
        ax.text(cx - w / 2 + 0.01, cy + h / 2 - 0.015, domain.capitalize(), ha="left", va="top", fontsize=11, fontweight="bold")

    for u, v, data in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        cross = data.get("cross_domain", False)
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=7 if cross else 5,
            linewidth=1.0 if cross else 0.6,
            linestyle="--" if cross else "-",
            color="#64748B" if cross else "#CBD5E1",
            alpha=0.95 if cross else 0.35,
            shrinkA=10,
            shrinkB=10,
            connectionstyle="arc3,rad=0.05" if cross else "arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    for node in G.nodes():
        x, y = pos[node]
        if node in endpoints:
            fc = COLORS["endpoint"]
            patch = FancyBboxPatch((x - 0.06, y - 0.024), 0.12, 0.048, boxstyle="round,pad=0.01", facecolor=fc, edgecolor="#7F1D1D", linewidth=1.0)
            fs = 6.8
        elif node in shared_nodes:
            fc = COLORS["shared"]
            patch = FancyBboxPatch((x - 0.055, y - 0.026), 0.11, 0.052, boxstyle="round,pad=0.01", facecolor=fc, edgecolor="#8A6D1D", linewidth=1.0)
            fs = 7.2
        else:
            fc = COLORS[node_type(node)]
            patch = FancyBboxPatch((x - 0.036, y - 0.017), 0.072, 0.034, boxstyle="round,pad=0.006", facecolor=fc, edgecolor="#64748B", linewidth=0.6)
            fs = 4.8
        ax.add_patch(patch)
        ax.text(x, y, clean_label(node), ha="center", va="center", fontsize=fs)

    ax.text(0.02, 0.98, "Cross-Domain Convergence Supergraph", ha="left", va="top", fontsize=13, fontweight="bold")
    ax.text(0.02, 0.955, "Dashed edges denote cross-domain links through shared sociotechnical intermediaries.", ha="left", va="top", fontsize=8)
    fig.savefig(OUTDIR / "fig_supergraph.pdf")
    plt.close(fig)


def draw_stockflow_cyber():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    stocks = {
        "Vulnerability_Backlog\n$S_1$": (2.2, 5.2),
        "Active_Exploits\n$S_2$": (6.2, 5.2),
        "Incident_Burden\n$S_3$": (10.2, 5.2),
        "Defender_Capacity\n$S_4$": (10.2, 2.0),
        "Public_Trust_Cyber\n$S_5$": (6.2, 2.0),
        "AI_Capability_Stock\n$S_6$": (2.2, 2.0),
    }
    auxiliaries = {
        "$f_{discover}$": (1.1, 6.9),
        "$f_{patch}$": (4.1, 6.9),
        "$f_{weaponize}$": (5.1, 6.9),
        "$f_{mitigate}$": (8.1, 6.9),
        "$f_{incident}$": (9.1, 6.9),
        "$f_{resolve}$": (12.1, 6.9),
        "$f_{invest}$": (12.4, 3.6),
        "$f_{exhaust}$": (12.8, 1.0),
        "$f_{restore}$": (8.1, 0.8),
        "$f_{erode}$": (7.0, 3.4),
    }

    for label, (x, y) in stocks.items():
        rect = Rectangle((x - 1.1, y - 0.55), 2.2, 1.1, facecolor=COLORS["stock"], edgecolor="#334155", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8)
    for label, (x, y) in auxiliaries.items():
        circ = Circle((x, y), 0.42, facecolor=COLORS["aux"], edgecolor="#334155", linewidth=1.0)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", fontsize=7)

    def flow(x1, y1, x2, y2, label):
        ax.plot([x1, x2], [y1, y2], color="#334155", linewidth=1.6)
        vx = (x1 + x2) / 2
        vy = (y1 + y2) / 2
        valve = Polygon([[vx - 0.15, vy + 0.12], [vx + 0.15, vy], [vx - 0.15, vy - 0.12]], closed=True, facecolor="white", edgecolor="#334155")
        ax.add_patch(valve)
        ax.add_patch(FancyArrowPatch((x2 - 0.2, y2), (x2 + 0.3, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color="#334155"))
        ax.text(vx, vy + 0.35, label, ha="center", va="bottom", fontsize=7)

    flow(3.3, 5.2, 5.1, 5.2, "$f_{weaponize}$")
    flow(7.3, 5.2, 9.1, 5.2, "$f_{incident}$")
    flow(11.3, 5.2, 13.3, 5.2, "$f_{resolve}$")
    flow(3.3, 2.0, 5.1, 2.0, "$\\alpha_{cap}$")
    flow(7.3, 2.0, 9.1, 2.0, "$f_{restore}$")
    flow(11.3, 2.0, 13.3, 2.0, "$f_{invest}$")
    ax.add_patch(FancyArrowPatch((0.5, 5.2), (1.1, 5.2), arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color="#334155"))
    ax.text(0.35, 5.55, "$f_{discover}$", fontsize=7)
    ax.add_patch(FancyArrowPatch((4.1, 6.4), (2.2, 5.8), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.add_patch(FancyArrowPatch((8.1, 6.4), (6.2, 5.8), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.add_patch(FancyArrowPatch((12.1, 6.4), (10.2, 5.8), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.add_patch(FancyArrowPatch((7.0, 3.4), (6.2, 2.6), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.add_patch(FancyArrowPatch((12.4, 3.2), (10.9, 2.5), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.add_patch(FancyArrowPatch((12.8, 1.2), (10.9, 1.7), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.text(0.1, 7.7, "Cyber Stock-and-Flow Model", ha="left", va="top", fontsize=13, fontweight="bold")
    ax.text(0.1, 7.35, "Rectangles are stocks; circles are auxiliary flow functions; pipes and valves indicate net flows.", fontsize=8)
    fig.savefig(OUTDIR / "fig_stockflow_cyber.pdf")
    plt.close(fig)


def draw_stockflow_cbrn():
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    stocks = {
        "Dangerous_Query_Volume\n$S_1^C$": (2.2, 5.0),
        "Successful_Bypass_Stock\n$S_2^C$": (6.0, 5.0),
        "Misuse_Opportunity\n$S_3^C$": (9.8, 5.0),
        "Review_Capacity\n$S_4^C$": (9.8, 2.0),
        "Governance_Response\n$S_5^C$": (6.0, 2.0),
        "AI_Capability_Level\n$S_6$": (2.2, 2.0),
    }
    auxiliaries = {
        "$g_{generate}$": (1.0, 6.8),
        "$g_{filter}$": (3.8, 6.8),
        "$g_{bypass}$": (4.9, 6.8),
        "$g_{detect}$": (7.7, 6.8),
        "$g_{exploit}$": (8.8, 6.8),
        "$g_{degrade}$": (11.6, 6.8),
        "$g_{expand}$": (11.4, 3.4),
        "$g_{overload}$": (11.9, 1.0),
        "$g_{escalate}$": (7.0, 0.8),
        "$g_{decay}$": (4.2, 0.8),
    }

    for label, (x, y) in stocks.items():
        rect = Rectangle((x - 1.15, y - 0.55), 2.3, 1.1, facecolor=COLORS["stock"], edgecolor="#334155", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8)
    for label, (x, y) in auxiliaries.items():
        circ = Circle((x, y), 0.42, facecolor=COLORS["aux"], edgecolor="#334155", linewidth=1.0)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", fontsize=7)

    def flow(x1, y1, x2, y2, label):
        ax.plot([x1, x2], [y1, y2], color="#334155", linewidth=1.6)
        vx = (x1 + x2) / 2
        vy = (y1 + y2) / 2
        valve = Polygon([[vx - 0.15, vy + 0.12], [vx + 0.15, vy], [vx - 0.15, vy - 0.12]], closed=True, facecolor="white", edgecolor="#334155")
        ax.add_patch(valve)
        ax.add_patch(FancyArrowPatch((x2 - 0.2, y2), (x2 + 0.3, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color="#334155"))
        ax.text(vx, vy + 0.35, label, ha="center", va="bottom", fontsize=7)

    flow(3.35, 5.0, 4.9, 5.0, "$g_{bypass}$")
    flow(7.15, 5.0, 8.8, 5.0, "$g_{exploit}$")
    flow(10.95, 5.0, 12.6, 5.0, "$g_{degrade}$")
    flow(3.35, 2.0, 4.8, 2.0, "$g_{decay}$")
    flow(7.15, 2.0, 8.7, 2.0, "$g_{escalate}$")
    flow(10.95, 2.0, 12.5, 2.0, "$g_{expand}$")
    ax.add_patch(FancyArrowPatch((0.5, 5.0), (1.0, 5.0), arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color="#334155"))
    ax.text(0.2, 5.35, "$g_{generate}$", fontsize=7)
    ax.add_patch(FancyArrowPatch((3.8, 6.4), (2.2, 5.6), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.add_patch(FancyArrowPatch((7.7, 6.4), (6.0, 5.6), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.add_patch(FancyArrowPatch((11.6, 6.4), (9.8, 5.6), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.add_patch(FancyArrowPatch((11.4, 3.1), (10.4, 2.5), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.add_patch(FancyArrowPatch((11.9, 1.2), (10.4, 1.8), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.add_patch(FancyArrowPatch((7.0, 1.1), (6.0, 1.6), arrowstyle="-|>", mutation_scale=10, linestyle="--", color="#64748B"))
    ax.text(0.1, 7.7, "CBRN Stock-and-Flow Model", ha="left", va="top", fontsize=13, fontweight="bold")
    ax.text(0.1, 7.35, "Structure follows the Appendix equations with shared AI capability stock and governance-response loop.", fontsize=8)
    fig.savefig(OUTDIR / "fig_stockflow_cbrn.pdf")
    plt.close(fig)


def read_csv(name: str):
    with open(BASE / name, newline="") as f:
        return list(csv.DictReader(f))


def draw_exp2_kappa():
    rows = read_csv("csv/exp2_coding_reliability.csv")
    domains = [r["Domain"] for r in rows if r["Domain"] != "Aggregate"]
    node = [float(r["Node_Kappa"]) for r in rows if r["Domain"] != "Aggregate"]
    edge = [float(r["Edge_Kappa"]) for r in rows if r["Domain"] != "Aggregate"]
    sign = [float(r["Sign_Kappa"]) for r in rows if r["Domain"] != "Aggregate"]

    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.25, node, 0.25, label="Node", color="#5B8FF9")
    ax.bar(x, edge, 0.25, label="Edge", color="#5AD8A6")
    ax.bar(x + 0.25, sign, 0.25, label="Sign", color="#F6BD16")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=20, ha="right")
    ax.set_ylabel("Cohen's kappa")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Experiment 2: Inter-Annotator Agreement by Domain")
    fig.savefig(OUTDIR / "exp2_kappa_bars.pdf")
    plt.close(fig)


def draw_exp6_pathways():
    rows = [r for r in read_csv("exp6_convergence.csv") if r["Endpoint"] != "Total"]
    endpoints = [r["Endpoint"] for r in rows]
    full = [int(r["Full_Graph"]) for r in rows]
    silo = [int(r["Silo"]) for r in rows]
    direct = [int(r["Direct_Only"]) for r in rows]
    x = np.arange(len(endpoints))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.25, full, 0.25, label="Full graph", color="#5B8FF9")
    ax.bar(x, silo, 0.25, label="Silo baseline", color="#94A3B8")
    ax.bar(x + 0.25, direct, 0.25, label="Direct-only", color="#5AD8A6")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace(" ", "\n") for e in endpoints], fontsize=7)
    ax.set_ylabel("Pathway count")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Experiment 6: Pathway Reachability by Endpoint")
    fig.savefig(OUTDIR / "exp6_pathway_counts.pdf")
    plt.close(fig)


def draw_exp8_robustness():
    rows = read_csv("exp8_robustness.csv")
    specs = [r["Specification"] for r in rows]
    cyber = [float(r["Cyber_Disinfo_Delta%"]) for r in rows]
    cbrn = [float(r["CBRN_Disinfo_Delta%"]) for r in rows]
    x = np.arange(len(specs))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, cyber, 0.35, label="Cyber", color="#5B8FF9")
    ax.bar(x + 0.18, cbrn, 0.35, label="CBRN", color="#F08C00")
    ax.set_xticks(x)
    ax.set_xticklabels(specs)
    ax.set_ylabel("Disinformation delta (%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Experiment 8: Robustness of the Disinformation Effect")
    fig.savefig(OUTDIR / "exp8_robustness_bars.pdf")
    plt.close(fig)


def main():
    draw_cld("Cyber", all_graphs["cyber"], seed=7, want_r=4, want_b=3)
    draw_cld("CBRN", all_graphs["cbrn"], seed=9, want_r=3, want_b=3)
    draw_cld("Deception", all_graphs["deception"], seed=11, want_r=5, want_b=2)
    draw_cld("Autonomy", all_graphs["autonomy"], seed=13, want_r=2, want_b=2)
    draw_cld("Governance", all_graphs["governance"], seed=15, want_r=4, want_b=4)
    draw_supergraph()
    draw_stockflow_cyber()
    draw_stockflow_cbrn()
    draw_exp2_kappa()
    draw_exp6_pathways()
    draw_exp8_robustness()


if __name__ == "__main__":
    main()
