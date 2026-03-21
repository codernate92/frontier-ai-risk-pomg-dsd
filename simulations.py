"""
simulations.py - Stock-and-flow ODE models for cyber and CBRN domains.
Implements exact equations from the paper Sections 3.4 and Appendix B.
"""
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

DISINFO_COUPLINGS_BASELINE = {
    'kappa_delay': 0.5,
    'kappa_degrade': 0.4,
    'kappa_amplify': 0.6,
    'kappa_overload': 0.3,
}
DISINFO_COUPLINGS_RANGES = {
    'kappa_delay': (0.2, 0.8),
    'kappa_degrade': (0.2, 0.8),
    'kappa_amplify': (0.2, 0.8),
    'kappa_overload': (0.1, 0.6),
}


def _resolve_couplings(couplings=None):
    resolved = dict(DISINFO_COUPLINGS_BASELINE)
    if couplings:
        resolved.update(couplings)
    return resolved

# ===========================================================================
# CYBER DOMAIN MODEL
# ===========================================================================

CYBER_PARAMS_BASELINE = dict(
    alpha_vuln=5.0, alpha_weapon=0.3, beta_scale=0.8,
    tau_patch=4.0, tau_mitigate=2.0, tau_resolve=3.0,
    theta_patch=10.0, theta_mitigate=5.0, theta_resolve=8.0,
    theta_exhaust=15.0, alpha_invest=0.1, alpha_restore=0.05,
    beta_erode=0.08, alpha_cap=0.15, K_cap=1.0, lambda_monitor=0.6,
)
CYBER_PARAMS_RANGES = {
    'alpha_vuln': (2.5, 7.5), 'alpha_weapon': (0.15, 0.45),
    'beta_scale': (0.4, 1.2), 'tau_patch': (2.0, 6.0),
    'tau_mitigate': (1.0, 3.0), 'tau_resolve': (1.5, 4.5),
    'theta_patch': (5.0, 15.0), 'theta_mitigate': (2.5, 7.5),
    'theta_resolve': (4.0, 12.0), 'theta_exhaust': (7.5, 22.5),
    'alpha_invest': (0.05, 0.15), 'alpha_restore': (0.025, 0.075),
    'beta_erode': (0.04, 0.12), 'alpha_cap': (0.075, 0.225),
    'K_cap': (0.5, 1.5), 'lambda_monitor': (0.3, 0.9),
}
CYBER_IC = [5.0, 2.0, 1.0, 10.0, 0.7, 0.3]  # S1..S6


def cyber_ode(t, y, p, lambda_disinfo=0.0, couplings=None):
    """Cyber domain ODE system. y = [S1..S6]."""
    S1, S2, S3, S4, S5, S6 = y
    S1 = max(S1, 0.001); S2 = max(S2, 0.001); S3 = max(S3, 0.001)
    S4 = max(S4, 0.001); S5 = np.clip(S5, 0.001, 0.999); S6 = np.clip(S6, 0.001, p['K_cap'])

    # Disinformation coupling
    c = _resolve_couplings(couplings)
    tau_resolve_eff = p['tau_resolve'] * (1 + lambda_disinfo * c['kappa_delay'])
    lambda_monitor_eff = p['lambda_monitor'] * (1 - lambda_disinfo * c['kappa_degrade'])
    S4_eff = S4 * (1 - lambda_disinfo * c['kappa_overload'])

    # Flow functions
    f_discover = p['alpha_vuln'] * S6**0.7
    f_patch = S1 * S4_eff / (p['tau_patch'] * (S1 + p['theta_patch']))
    f_weaponize = p['alpha_weapon'] * S1**0.5 * S6**0.8
    f_mitigate = S2 * S4_eff / (p['tau_mitigate'] * (S2 + p['theta_mitigate']))
    f_incident = p['beta_scale'] * S2
    f_resolve = S3 * S4_eff / (tau_resolve_eff * (S3 + p['theta_resolve']))
    f_invest = p['alpha_invest'] * S3 * S5
    f_exhaust = S3 / (S3 + p['theta_exhaust']) * S4
    f_restore = p['alpha_restore'] * S4_eff * lambda_monitor_eff
    f_erode = p['beta_erode'] * S3

    dS1 = f_discover - f_patch
    dS2 = f_weaponize - f_mitigate
    dS3 = f_incident - f_resolve
    dS4 = f_invest - f_exhaust
    dS5 = f_restore - f_erode
    dS6 = p['alpha_cap'] * (1 - S6 / p['K_cap'])
    return [dS1, dS2, dS3, dS4, dS5, dS6]


def run_cyber_sim(params=None, ic=None, T=52, lambda_disinfo=0.0, couplings=None):
    """Run cyber simulation, return solution and metrics."""
    p = dict(CYBER_PARAMS_BASELINE)
    if params:
        p.update(params)
    y0 = ic if ic is not None else CYBER_IC[:]
    sol = solve_ivp(cyber_ode, [0, T], y0, args=(p, lambda_disinfo, couplings),
                    method='RK45', max_step=0.1, rtol=1e-6, atol=1e-9,
                    dense_output=True)
    t_eval = np.linspace(0, T, 520)
    y_eval = sol.sol(t_eval)
    S3 = y_eval[2]  # Incident Burden
    peak = np.max(S3)
    t_peak = t_eval[np.argmax(S3)]
    cum_harm = np.trapezoid(S3, t_eval)
    # Time to control: first time S3 returns below initial after peak
    baseline_level = y0[2] * 1.1
    post_peak = t_eval > t_peak
    below = S3[post_peak] < baseline_level
    t_ctrl = t_eval[post_peak][np.argmax(below)] if np.any(below) else T
    return dict(t=t_eval, y=y_eval, peak=peak, t_peak=t_peak,
                t_ctrl=t_ctrl, cum_harm=cum_harm, params=p)


# ===========================================================================
# CBRN DOMAIN MODEL
# ===========================================================================

CBRN_PARAMS_BASELINE = dict(
    alpha_q=8.0, phi=0.85, alpha_e=0.4, tau_d=2.0, theta_d=3.0,
    theta_deg=5.0, alpha_x=0.12, theta_ovl=20.0, alpha_esc=0.08,
    theta_esc=10.0, tau_g=26.0, psi=0.2,
    alpha_cap=0.15, K_cap=1.0,  # shared AI capability params
)
CBRN_PARAMS_RANGES = {
    'alpha_q': (4.0, 12.0), 'phi': (0.70, 0.95), 'alpha_e': (0.2, 0.6),
    'tau_d': (1.0, 3.0), 'theta_d': (1.5, 4.5), 'theta_deg': (2.5, 7.5),
    'alpha_x': (0.06, 0.18), 'theta_ovl': (10.0, 30.0), 'alpha_esc': (0.04, 0.12),
    'theta_esc': (5.0, 15.0), 'tau_g': (13.0, 39.0), 'psi': (0.1, 0.3),
    'alpha_cap': (0.075, 0.225), 'K_cap': (0.5, 1.5),
}
CBRN_IC = [10.0, 1.0, 0.5, 5.0, 0.3, 0.3]  # S1C..S5C + S6 (shared cap)


def cbrn_ode(t, y, p, lambda_disinfo=0.0, couplings=None):
    """CBRN domain ODE system. y = [S1C..S5C, S6]."""
    S1, S2, S3, S4, S5, S6 = y
    S1 = max(S1, 0.001); S2 = max(S2, 0.001); S3 = max(S3, 0.001)
    S4 = max(S4, 0.001); S5 = np.clip(S5, 0.001, 0.999); S6 = np.clip(S6, 0.001, p['K_cap'])

    # Disinformation coupling on CBRN
    c = _resolve_couplings(couplings)
    psi_eff = p['psi'] * (1 + lambda_disinfo * c['kappa_amplify'])
    S4_eff = S4 * (1 - lambda_disinfo * c['kappa_overload'])
    tau_g_eff = p['tau_g'] * (1 + lambda_disinfo * c['kappa_delay'])

    g_generate = p['alpha_q'] * S6**0.6 * (1 + psi_eff)
    g_filter = p['phi'] * S1
    g_bypass = (1 - p['phi']) * S1 * S6**0.5
    g_detect = S2 * S4_eff / (p['tau_d'] * (S2 + p['theta_d']))
    g_exploit = p['alpha_e'] * S2
    g_degrade = S5 * S3 / (S3 + p['theta_deg'])
    g_expand = p['alpha_x'] * S5
    g_overload = S1 / (S1 + p['theta_ovl']) * S4
    g_escalate = p['alpha_esc'] * S3 / (S3 + p['theta_esc'])
    g_decay = S5 / tau_g_eff

    dS1 = g_generate - g_filter
    dS2 = g_bypass - g_detect
    dS3 = g_exploit - g_degrade
    dS4 = g_expand - g_overload
    dS5 = g_escalate - g_decay
    dS6 = p['alpha_cap'] * (1 - S6 / p['K_cap'])
    return [dS1, dS2, dS3, dS4, dS5, dS6]


def run_cbrn_sim(params=None, ic=None, T=52, lambda_disinfo=0.0, couplings=None):
    """Run CBRN simulation, return solution and metrics."""
    p = dict(CBRN_PARAMS_BASELINE)
    if params:
        p.update(params)
    y0 = ic if ic is not None else CBRN_IC[:]
    sol = solve_ivp(cbrn_ode, [0, T], y0, args=(p, lambda_disinfo, couplings),
                    method='RK45', max_step=0.1, rtol=1e-6, atol=1e-9,
                    dense_output=True)
    t_eval = np.linspace(0, T, 520)
    y_eval = sol.sol(t_eval)
    S3 = y_eval[2]  # Misuse Opportunity
    peak = np.max(S3)
    t_peak = t_eval[np.argmax(S3)]
    cum_risk = np.trapezoid(S3, t_eval)
    baseline_level = y0[2] * 1.1
    post_peak = t_eval > t_peak
    below = S3[post_peak] < baseline_level
    t_ctrl = t_eval[post_peak][np.argmax(below)] if np.any(below) else T
    return dict(t=t_eval, y=y_eval, peak=peak, t_peak=t_peak,
                t_ctrl=t_ctrl, cum_risk=cum_risk, params=p)


# ===========================================================================
# MONTE CARLO + SENSITIVITY
# ===========================================================================

def monte_carlo_cyber(N=1000, lambda_disinfo=0.0):
    """Run N Monte Carlo simulations with uniform parameter sampling."""
    results = []
    failures = 0
    last_error: Exception | None = None
    rng = np.random.default_rng(42)
    for _ in range(N):
        params = {}
        for k, (lo, hi) in CYBER_PARAMS_RANGES.items():
            params[k] = rng.uniform(lo, hi)
        try:
            r = run_cyber_sim(params=params, lambda_disinfo=lambda_disinfo)
            results.append(dict(peak=r['peak'], t_peak=r['t_peak'],
                               t_ctrl=r['t_ctrl'], cum_harm=r['cum_harm'],
                               **params))
        except Exception as exc:
            failures += 1
            last_error = exc
            continue
    if failures:
        warnings.warn(
            f"Cyber Monte Carlo dropped {failures}/{N} failed runs; inspect parameter ranges if this is frequent.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not results:
        raise RuntimeError("Cyber Monte Carlo produced no successful runs.") from last_error
    return results


def monte_carlo_cbrn(N=1000, lambda_disinfo=0.0):
    """Run N Monte Carlo simulations for CBRN."""
    results = []
    failures = 0
    last_error: Exception | None = None
    rng = np.random.default_rng(42)
    for _ in range(N):
        params = {}
        for k, (lo, hi) in CBRN_PARAMS_RANGES.items():
            params[k] = rng.uniform(lo, hi)
        try:
            r = run_cbrn_sim(params=params, lambda_disinfo=lambda_disinfo)
            results.append(dict(peak=r['peak'], t_peak=r['t_peak'],
                               t_ctrl=r['t_ctrl'], cum_risk=r['cum_risk'],
                               **params))
        except Exception as exc:
            failures += 1
            last_error = exc
            continue
    if failures:
        warnings.warn(
            f"CBRN Monte Carlo dropped {failures}/{N} failed runs; inspect parameter ranges if this is frequent.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not results:
        raise RuntimeError("CBRN Monte Carlo produced no successful runs.") from last_error
    return results


def compute_prcc(mc_results, output_key, param_keys):
    """Compute Partial Rank Correlation Coefficients."""
    import pandas as pd
    df = pd.DataFrame(mc_results)
    ranks = df[param_keys + [output_key]].rank()
    prcc = {}
    for pk in param_keys:
        corr, pval = spearmanr(ranks[pk], ranks[output_key])
        prcc[pk] = (corr, pval)
    return prcc
