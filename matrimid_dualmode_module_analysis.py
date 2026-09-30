#!/usr/bin/env python3
# =============================================================================
# Can pure-gas permeability data design a mixed-gas module?
# A correctability map for dual-mode CO2/N2 transport in Matrimid 5218 hollow fibres
#
# Consolidated, reproducible analysis script (all results, tables and figures).
# Authors: M.V. Ajoku, E.D. Ezeokolie
#
# Requirements: Python >= 3.9, numpy, scipy, matplotlib
# Usage:        python matrimid_dualmode_module_analysis.py
#               (select sections in RUN below; full run takes roughly 1-2 h on one CPU)
# Output:       ./results/*.json (numerical results), ./figures/*.png|.pdf (600 dpi)
#
# Section map (manuscript cross-reference)
#   S1  Verification: Eq.(2) vs Eq.(3); comparison with David et al. (DWT 2011)   -> Sec. 2.2, 4.1
#   S2  F*K calibration (Bos two-point MC, Esposito bootstrap, 25->35 C, combined) -> Sec. 3.2, 4.2, Table 2
#   S3  Numerical checks: grid convergence, thickness invariance                  -> Sec. 2.6, Table S1
#   S4  Recovery/purity bias at p_ref = 1 bar across the F*K interval             -> Sec. 4.3, Table 3
#   S5  Area design error (flow-equivalence method)                               -> Sec. 4.3, Table 3
#   S6  Stage-cut sensitivity of the bias                                         -> Sec. 4.3, Fig. S1a
#   S7  Correcting pressure p* and thresholds at baseline                         -> Sec. 4.4, Table 4
#   S8  Regime map (feed pCO2 x stage cut) and practical threshold vs stage cut   -> Sec. 4.4
#   S9  Axial profiles (below-floor mechanism)                                    -> Sec. 4.4, Fig. S2
#   S10 CO2 / N2 decomposition, incl. F(N2) sensitivity                           -> Sec. 4.5, Table 5
#   S11 Temperature sub-model, Ea,D calibration, F(T) and Ea,D sensitivities      -> Sec. 3.3, 4.6, 4.7
#   S12 Plasticization check                                                      -> Sec. 4.8
#   S13 Figures 1-7, S1, S2
#
# Numerical notes (important for reproducibility)
#   * Relaxation factor omega = 0.4 is FIXED. Do not reduce omega between retries:
#     with a fixed iterate-change tolerance a smaller omega signals convergence early.
#   * Acceptance is on the CO2 mass-balance error (< 0.005 %). If not met, the iteration
#     budget (x3) and iterate-change tolerance (x0.1) are tightened TOGETHER.
#   * Independent random seeds are used for each Monte Carlo / bootstrap.
#   * Structural test: Model A evaluated at the Henry floor (p_ref -> 1e6 bar).
# =============================================================================
import os, json, copy, time
import numpy as np
from scipy.optimize import brentq, minimize_scalar

RUN = {'S1': True, 'S2': True, 'S3': True, 'S4': True, 'S5': True, 'S6': True, 'S7': True,
       'S8': True, 'S9': True, 'S10': True, 'S11': True, 'S12': True, 'S13': True}
RES_DIR, FIG_DIR = 'results', 'figures'
os.makedirs(RES_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
RESULTS = {}
T0 = time.time()
def log(msg): print(f"[{time.time() - T0:7.0f} s] {msg}", flush=True)
def dump(name, obj):
    RESULTS[name] = obj
    with open(os.path.join(RES_DIR, f'{name}.json'), 'w') as f:
        json.dump(obj, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else float(x))

# =============================================================================
# 0. Core model library
# =============================================================================
BAR_TO_CMHG = 75.0064
ATM = 1.01325
MOLAR_VOL = 22414.0                     # cm3(STP)/mol

def barrer_to_SI(P_barrer, L_cm):
    """Permeability [Barrer] -> permeance [mol m-2 s-1 bar-1] for a layer of thickness L_cm [cm]."""
    return P_barrer * 1e-10 * BAR_TO_CMHG / L_cm / MOLAR_VOL * 1e4

def make_module(n_fibers, l_membrane_m, L=1.0, d_outer=3e-4):
    a = n_fibers * np.pi * d_outer      # membrane area per unit length [m2/m]
    return {'L': L, 'n_fibers': n_fibers, 'a': a, 'A_total': a * L,
            'L_cm': l_membrane_m * 100.0, 'l_membrane_m': l_membrane_m}

MOD = make_module(10_000, 50e-6)        # 9.42 m2, l = 50 um (thickness cancels; see S3)

def make_matrimid_T1(F_N2=0.0, P_N2=0.19, bN2=None, FK_CO2=None):
    """Matrimid 5218 at 35 C. Sorption: Moore & Koros (2007). CO2 DD, DH: David et al. (DWT 2011).
    If FK_CO2 is given, DD(CO2) is re-solved so P_pure(CO2, 4 bar) stays at David's value (5.464 Barrer)."""
    kD = {'CO2': 1.42, 'N2': 0.120 / ATM}
    CH = {'CO2': 25.5, 'N2': 3.94}
    b = {'CO2': 0.367, 'N2': 0.087 if bN2 is None else bN2}
    DD_c, DH_c = 2.141e-8, 2.79e-9                     # cm2/s
    K_c = CH['CO2'] * b['CO2'] / kD['CO2']
    if FK_CO2 is not None:
        p_keep = 4.0
        P_ref = (kD['CO2'] / BAR_TO_CMHG) * DD_c * 1e10 * (1 + (DH_c / DD_c) * K_c / (1 + b['CO2'] * p_keep))
        F_c = FK_CO2 / K_c
        DD_c = P_ref / ((kD['CO2'] / BAR_TO_CMHG) * 1e10 * (1 + F_c * K_c / (1 + b['CO2'] * p_keep)))
        DH_c = DD_c * F_c
    KN = CH['N2'] * b['N2'] / kD['N2']
    DD_n = P_N2 / ((kD['N2'] / BAR_TO_CMHG) * 1e10 * (1.0 + F_N2 * KN))
    return {'kD': kD, 'CH': CH, 'b': b,
            'DD': {'CO2': DD_c, 'N2': DD_n}, 'DH': {'CO2': DH_c, 'N2': DD_n * F_N2},
            'F': {'CO2': DH_c / DD_c, 'N2': F_N2}, 'P_pure': {'CO2': 5.5, 'N2': P_N2}}

def _sorb(m, gas, p_i, p_j, other):
    """Henry and competitive-Langmuir sorbed concentrations, Eq.(1)."""
    kD, CH = m['kD'][gas], m['CH'][gas]
    bi, bj = m['b'][gas], m['b'][other]
    return kD * p_i, CH * bi * p_i / (1.0 + bi * p_i + bj * p_j)

def P_eff_exact(m, gas, pf_i, pf_j, pp_i, pp_j, other):
    """Exact two-face effective permeability [Barrer], Eq.(2)."""
    dp = pf_i - pp_i
    if dp <= 0.0:
        return 0.0
    CDf, CHf = _sorb(m, gas, pf_i, pf_j, other)
    CDp, CHp = _sorb(m, gas, pp_i, pp_j, other)
    Nl = m['DD'][gas] * (CDf - CDp) + m['DH'][gas] * (CHf - CHp)
    return 1e10 * Nl / (dp * BAR_TO_CMHG)

def P_vac(m, gas, p, p_other=0.0):
    """Pure-gas (or single-face competitive) permeability with vacuum downstream."""
    other = 'N2' if gas == 'CO2' else 'CO2'
    return P_eff_exact(m, gas, p, p_other, 0.0, 0.0, other)

def get_J(m, model, yc, xc, p_feed, p_perm, SI):
    yc = float(np.clip(yc, 1e-9, 1 - 1e-9)); xc = float(np.clip(xc, 1e-9, 1 - 1e-9))
    pcf, pnf = yc * p_feed, (1 - yc) * p_feed
    pcp, pnp = xc * p_perm, (1 - xc) * p_perm
    if model == 'A':                                   # pure-gas design: constant permeability
        Pc, Pn = m['P_pure']['CO2'], m['P_pure']['N2']
    else:                                              # true mixed-gas: local exact two-face flux
        Pc = P_eff_exact(m, 'CO2', pcf, pnf, pcp, pnp, 'N2')
        Pn = P_eff_exact(m, 'N2', pnf, pcf, pnp, pcp, 'CO2')
    return SI * Pc * max(pcf - pcp, 0.0), SI * Pn * max(pnf - pnp, 0.0)

def run_module(m, model, cond, module, N=50, max_iter=4000, tol=1e-9, omega=0.4, return_profiles=False):
    """Counter-current hollow-fibre module, Eqs.(4)-(5); Gauss-Seidel, alternating sweeps."""
    F_in, y_in = cond['F_feed_in'], cond['y_CO2_feed']
    p_feed, p_perm = cond['p_feed'], cond['p_perm']
    L, a = module['L'], module['a']
    SI = barrer_to_SI(1.0, module['L_cm']); dz = L / N
    Fc_in, Fn_in = F_in * y_in, F_in * (1.0 - y_in)
    floor = F_in * 1e-9                                # relative floor
    Pc0, Pn0 = m['P_pure']['CO2'], m['P_pure']['N2']
    Jc0 = SI * Pc0 * (y_in * (p_feed - p_perm)); Jn0 = SI * Pn0 * ((1 - y_in) * (p_feed - p_perm))
    xc_est = Jc0 / max(Jc0 + Jn0, 1e-30)
    Fp_est = min((Jc0 + Jn0) * a * L * 0.5, F_in * 0.4)
    Fcf = np.linspace(Fc_in, max(Fc_in - Fp_est * xc_est, Fc_in * 0.5), N)
    Fnf = np.linspace(Fn_in, max(Fn_in - Fp_est * (1 - xc_est), Fn_in * 0.5), N)
    Fcp = np.linspace(Fp_est * xc_est, 0.0, N)
    Fnp = np.linspace(Fp_est * (1 - xc_est), 0.0, N)
    converged = False
    for it in range(max_iter):
        o = [Fcf.copy(), Fnf.copy(), Fcp.copy(), Fnp.copy()]
        fc, fn = Fc_in, Fn_in
        for k in range(N):                             # feed sweep, z = 0 -> L
            yc = Fcf[k] / max(Fcf[k] + Fnf[k], 1e-20); xc = Fcp[k] / max(Fcp[k] + Fnp[k], 1e-20)
            Jc, Jn = get_J(m, model, yc, xc, p_feed, p_perm, SI)
            Fcf[k] = omega * max(fc - dz * a * Jc, floor) + (1 - omega) * o[0][k]
            Fnf[k] = omega * max(fn - dz * a * Jn, floor) + (1 - omega) * o[1][k]
            fc, fn = Fcf[k], Fnf[k]
        pc, pn = 0.0, 0.0
        for k in range(N - 1, -1, -1):                 # permeate sweep, z = L -> 0
            yc = Fcf[k] / max(Fcf[k] + Fnf[k], 1e-20); xc = Fcp[k] / max(Fcp[k] + Fnp[k], 1e-20)
            Jc, Jn = get_J(m, model, yc, xc, p_feed, p_perm, SI)
            Fcp[k] = omega * max(pc + dz * a * Jc, 0.0) + (1 - omega) * o[2][k]
            Fnp[k] = omega * max(pn + dz * a * Jn, 0.0) + (1 - omega) * o[3][k]
            pc, pn = Fcp[k], Fnp[k]
        delta = sum(np.max(np.abs(A - B)) for A, B in zip([Fcf, Fnf, Fcp, Fnp], o))
        if delta < tol:
            converged = True
            break
    Fcp_out, Fnp_out = float(Fcp[0]), float(Fnp[0])
    Fp_out = Fcp_out + Fnp_out
    r = {'converged': converged, 'iterations': it + 1,
         'CO2_recovery': Fcp_out / Fc_in * 100, 'CO2_purity': Fcp_out / max(Fp_out, 1e-20) * 100,
         'stage_cut': Fp_out / F_in * 100,
         'mb_error_pct': abs(Fc_in - Fcp_out - float(Fcf[-1])) / Fc_in * 100}
    if return_profiles:
        r['profiles'] = (Fcf.copy(), Fnf.copy(), Fcp.copy(), Fnp.copy())
    return r

def run_module_robust(m, model, cond, module=MOD, N=50, mb_tol=0.005, tol_start=1e-9,
                      max_iter_start=4000, return_profiles=False):
    """Accept on mass balance; escalate tolerance AND iteration budget together (omega fixed)."""
    tol, mi = tol_start, max_iter_start
    for attempt in range(8):
        r = run_module(m, model, cond, module, N=N, max_iter=mi, tol=tol, omega=0.4,
                       return_profiles=return_profiles)
        if r['mb_error_pct'] < mb_tol:
            r['converged'] = True
            return r
        tol *= 0.1; mi *= 3
    r['converged'] = False
    print(f"   WARNING: mass-balance criterion not met (MB = {r['mb_error_pct']:.4f} %)")
    return r

# ---------------- design-basis helpers ----------------
P_REF_CO2, P_REF_N2 = 5.5, 0.19          # Eq.(6)
PHI_BASE = 5.0                           # baseline flow parameter phi
P_FLOOR = 1e6                            # p_ref representing the Henry's-law floor

def F_in_rule(y, p, module=MOD, factor=PHI_BASE, clip=True):
    """Feed flow, Eq.(6). clip=True reproduces the [1e-5, 1e-2] mol/s bound used for the reference module."""
    SI = barrer_to_SI(1.0, module['L_cm'])
    F = SI * (P_REF_CO2 * y + P_REF_N2 * (1 - y)) * (p - 1.0) * module['a'] * module['L'] * factor
    return max(min(F, 1e-2), 1e-5) if clip else F

def design_basis(m, p_ref):
    """Model A: pure-gas CO2 permeability at p_ref (Eq.3), N2 at its pure-gas value."""
    d = copy.deepcopy(m); d['P_pure']['CO2'] = P_vac(m, 'CO2', p_ref); return d

def cond_of(y, pf, F_in):
    return {'F_feed_in': F_in, 'y_CO2_feed': y, 'p_feed': float(pf), 'p_perm': 1.0}

def rec_pair(m, y, pf, p_ref, module=MOD, factor=PHI_BASE, F_in=None, N=50, clip=True):
    if F_in is None:
        F_in = F_in_rule(y, pf, module, factor, clip)
    c = cond_of(y, pf, F_in)
    rB = run_module_robust(m, 'B', c, module, N=N)
    rA = run_module_robust(design_basis(m, p_ref), 'A', c, module, N=N)
    return {'RecA': rA['CO2_recovery'], 'RecB': rB['CO2_recovery'], 'PurA': rA['CO2_purity'],
            'PurB': rB['CO2_purity'], 'dRec': rA['CO2_recovery'] - rB['CO2_recovery'],
            'dPur': rA['CO2_purity'] - rB['CO2_purity'], 'stage_cut_B': rB['stage_cut'],
            'MB_max': max(rA['mb_error_pct'], rB['mb_error_pct'])}

def dRec(m, y, pf, p_ref, **kw):
    return rec_pair(m, y, pf, p_ref, **kw)['dRec']

def threshold_y(m, pf, p_ref, lo, hi, **kw):
    """Feed CO2 fraction where dRec(p_ref) = 0 (p_ref = P_FLOOR -> structural threshold)."""
    f = lambda y: dRec(m, y, pf, p_ref, **kw)
    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):
        return float('nan')
    return brentq(f, lo, hi, xtol=2e-4)

def pstar(m, y, pf=10.0, factor=PHI_BASE):
    """Correcting measurement pressure p*: root of dRec(p_ref) in log(p_ref); nan if none."""
    c = cond_of(y, pf, F_in_rule(y, pf, factor=factor))
    RB = run_module_robust(m, 'B', c)['CO2_recovery']
    f = lambda lp: run_module_robust(design_basis(m, np.exp(lp)), 'A', c)['CO2_recovery'] - RB
    lo, hi = np.log(0.25), np.log(P_FLOOR)
    if f(hi) > 0 or f(lo) < 0:
        return float('nan')
    return float(np.exp(brentq(f, lo, hi, xtol=1e-3)))

FK_P, FK_L, FK_H = 0.191, 0.123, 0.272   # primary and 90 % interval (confirmed by S2)
FK_DAVID = 0.859                          # single-laboratory fit, comparison only
CONDS6 = [(0.05, 10), (0.05, 20), (0.15, 10), (0.15, 20), (0.30, 10), (0.30, 20)]

# =============================================================================
# S1. Verification
# =============================================================================
if RUN['S1']:
    log('S1 verification')
    m = make_matrimid_T1(FK_CO2=FK_P)
    K = m['CH']['CO2'] * m['b']['CO2'] / m['kD']['CO2']
    base = (m['kD']['CO2'] / BAR_TO_CMHG) * m['DD']['CO2'] * 1e10
    red = {p: abs(P_vac(m, 'CO2', p) - base * (1 + m['F']['CO2'] * K / (1 + m['b']['CO2'] * p))) / P_vac(m, 'CO2', p)
           for p in [1, 3, 6, 10, 20]}
    # David et al. (DWT 2011): pure CO2 at 30 C and H2/CO2 mixtures. Model at 35 C parameters.
    # NOTE: F*K = 0.859 and b_H2 = 0.05 were fitted by David et al. to these same data;
    # agreement at 0.859 verifies implementation, not the parameter set.
    dwt_pure = [(2, 6.1), (4, 5.5), (6, 5.2)]
    dwt_mix = [(0.41, 3.53, 5.4), (1.21, 2.75, 5.5), (2.01, 1.97, 5.8), (2.81, 1.18, 6.1), (3.60, 0.40, 6.9),
               (0.61, 5.23, 5.2), (1.82, 4.08, 5.3), (3.02, 2.92, 5.5), (4.22, 1.76, 5.6), (5.41, 0.59, 6.2)]
    bH2 = 0.05
    dav = {}
    for FK in [FK_DAVID, FK_P, FK_L, FK_H]:
        mm = make_matrimid_T1(FK_CO2=FK)
        Kc = mm['CH']['CO2'] * mm['b']['CO2'] / mm['kD']['CO2']
        bs = (mm['kD']['CO2'] / BAR_TO_CMHG) * mm['DD']['CO2'] * 1e10
        pe = [100 * (P_vac(mm, 'CO2', p) - v) / v for p, v in dwt_pure]
        me = [100 * (bs * (1 + mm['F']['CO2'] * Kc / (1 + mm['b']['CO2'] * pC + bH2 * pH)) - v) / v for pH, pC, v in dwt_mix]
        dav[str(FK)] = {'pure_err_pct_2_4_6bar': pe, 'mix_rms_pct': float(np.sqrt(np.mean(np.square(me)))),
                        'mix_max_abs_pct': float(np.max(np.abs(me)))}
    # Implied F*K from raw pure-gas ratios P(6)/P(2) (companion study, David et al. JMS 378 Table 1, 30 C)
    b0 = 0.367
    ratio = lambda x: (1 + x / (1 + b0 * 6)) / (1 + x / (1 + b0 * 2))
    FK_jms_raw = brentq(lambda x: ratio(x) - 4.0 / 4.6, 1e-4, 20.0)
    dump('S1_verification', {'eq2_vs_eq3_relative_diff': red, 'david_DWT_comparison': dav,
                             'FK_implied_JMS378_raw_P6_over_P2': FK_jms_raw})
    log(f"  Eq2 vs Eq3 max rel diff {max(red.values()):.1e}; David @0.859 mix RMS {dav[str(FK_DAVID)]['mix_rms_pct']:.1f} %, "
        f"@0.191 {dav[str(FK_P)]['mix_rms_pct']:.1f} %; JMS378 raw F*K {FK_jms_raw:.2f}")

# =============================================================================
# S2. F*K calibration
# =============================================================================
b0, bN2v = 0.367, 0.087
def Mshape(pc, po, FK): return 1.0 + FK / (1.0 + b0 * pc + bN2v * po)
pt_g = np.array([1, 2, 3, 4, 5, 6, 5.5, 4.5, 3.5, 2.5, 1.5, 1.0], float)            # Esposito et al., Table A1
P_g = np.array([11.3, 11.0, 10.8, 10.7, 10.4, 10.8, 10.9, 10.5, 10.7, 10.8, 11.0, 11.1])  # 15/85 CO2/N2, 25 C

# temperature scaling (also used in S11)
Rg = 8.314e-3; T_ref = 308.15
eps_k = {'CO2': 195.2, 'N2': 71.4}; m_LJ = 6.67
dHs = {'CO2': -14.9, 'N2': -3.4}
def kD_scale(gas, T): return np.exp(m_LJ * eps_k[gas] * (1.0 / T - 1.0 / T_ref))
def CH_scale(gas, T): return np.exp(-dHs[gas] / Rg * (1.0 / T - 1.0 / T_ref))
K_RATIO_25_35 = kD_scale('CO2', 298.15) / CH_scale('CO2', 298.15)   # K(35 C)/K(25 C)

def FK_from_bos(P0, P12):
    R = P12 / P0
    return (R - 1.0) / (1.0 / (1.0 + 12 * b0) - R)

def fit_esp(idx):
    def sse(FK):
        f = Mshape(0.15 * pt_g[idx], 0.85 * pt_g[idx], FK); a = P_g[idx] @ f / (f @ f)
        return np.sum(((a * f - P_g[idx]) / a) ** 2)
    return minimize_scalar(sse, bounds=(0.001, 3.0), method='bounded').x

def resid_esp_frac(FK):
    f = Mshape(0.15 * pt_g, 0.85 * pt_g, FK); a = P_g @ f / (f @ f)
    return (a * f - P_g) / a

if RUN['S2'] or RUN['S13']:
    log('S2 F*K calibration')
    rng_b = np.random.default_rng(101)
    bos = {}
    for sig in [0.03, 0.05]:
        s = FK_from_bos(5.7 * (1 + sig * rng_b.standard_normal(5000)), 4.8 * (1 + sig * rng_b.standard_normal(5000)))
        bos[sig] = s[(s > 0) & (s < 3)]
    rng_e = np.random.default_rng(202)
    esp25 = np.array([fit_esp(rng_e.integers(0, 12, 12)) for _ in range(5000)])
    esp35 = esp25 * K_RATIO_25_35
    comb = {}
    for sig in [0.03, 0.05]:
        n = min(len(bos[sig]), len(esp35))
        wb, we = 1 / np.var(bos[sig]), 1 / np.var(esp35)
        comb[sig] = (wb * bos[sig][:n] + we * esp35[:n]) / (wb + we)
    q = lambda s: [float(np.percentile(s, 5)), float(np.median(s)), float(np.percentile(s, 95))]
    r859 = resid_esp_frac(FK_DAVID)
    cal = {'bos_two_point_FK': FK_from_bos(5.7, 4.8),
           'bos_sigma3': q(bos[0.03]), 'bos_sigma5': q(bos[0.05]),
           'esposito_point_25C': float(fit_esp(np.arange(12))), 'esposito_25C': q(esp25),
           'K_ratio_35_over_25': float(K_RATIO_25_35), 'esposito_35C': q(esp35),
           'combined_sigma3_primary': q(comb[0.03]), 'combined_sigma5_conservative': q(comb[0.05]),
           'heldout_FK0859_frac_rmse_vs_esposito': float(np.sqrt(np.mean(r859 ** 2))),
           'heldout_flat_baseline_cv': float(P_g.std() / P_g.mean())}
    dump('S2_calibration', cal)
    CAL_SAMPLES = {'bos3': bos[0.03], 'esp35': esp35, 'comb3': comb[0.03]}
    log(f"  combined (primary) {cal['combined_sigma3_primary']}; held-out RMSE {cal['heldout_FK0859_frac_rmse_vs_esposito']:.3f} "
        f"vs baseline {cal['heldout_flat_baseline_cv']:.3f}")

# =============================================================================
# S3. Numerical checks
# =============================================================================
if RUN['S3'] or RUN['S13']:
    log('S3 grid convergence and thickness invariance')
    m = make_matrimid_T1(FK_CO2=FK_P)
    grid = {}
    for N in [25, 50, 100, 200]:
        r = rec_pair(m, 0.15, 10, 1.0, N=N)
        grid[N] = {'RecB': r['RecB'], 'dRec': r['dRec'], 'MB': r['MB_max']}
    d = [grid[N]['dRec'] for N in [25, 50, 100, 200]]
    grid['dRec_extrapolated_first_order'] = d[-1] + (d[-1] - d[-2])
    thick = {}
    for l in [50e-6, 5e-6, 0.5e-6]:
        mod = make_module(10_000, l)
        r = rec_pair(m, 0.15, 10, 1.0, module=mod, F_in=F_in_rule(0.15, 10, module=mod, clip=False))
        thick[f'{l * 1e6:g}um'] = {'RecB': r['RecB'], 'dRec': r['dRec'], 'MB': r['MB_max']}
    dump('S3_numerics', {'grid': grid, 'thickness': thick})

# =============================================================================
# S4. Recovery / purity bias at p_ref = 1 bar
# =============================================================================
if RUN['S4'] or RUN['S13']:
    log('S4 bias table')
    bias = {}
    for FK in [FK_L, FK_P, FK_H, FK_DAVID]:
        m = make_matrimid_T1(FK_CO2=FK)
        bias[str(FK)] = {f'{y}_{pf}': rec_pair(m, y, pf, 1.0) for y, pf in CONDS6}
    dump('S4_bias', bias)

# =============================================================================
# S5. Area design error (flow equivalence: A_B/A_A = F_in,A / F_in,B at equal recovery)
# =============================================================================
def phi_for_target(m, model, y, pf, target, lo=0.8, hi=20.0):
    def rec(phi):
        return run_module_robust(m, model, cond_of(y, pf, F_in_rule(y, pf, factor=phi, clip=False)))
    f = lambda phi: rec(phi)['CO2_recovery'] - target
    if np.sign(f(lo)) == np.sign(f(hi)):
        return float('nan'), float('nan')
    phi = brentq(f, lo, hi, xtol=1e-3)
    return phi, rec(phi)['CO2_purity']

if RUN['S5'] or RUN['S13']:
    log('S5 area error')
    m = make_matrimid_T1(FK_CO2=FK_P); mA = design_basis(m, 1.0)
    area = {}
    for y, pf in [(0.05, 10), (0.15, 10), (0.15, 20), (0.30, 10), (0.30, 20)]:
        for tgt in [50.0, 75.0]:
            fA, PuA = phi_for_target(mA, 'A', y, pf, tgt)
            fB, PuB = phi_for_target(m, 'B', y, pf, tgt)
            ratio = fA / fB
            area[f'{y}_{pf}_{int(tgt)}'] = {'A_B_over_A_A': ratio, 'area_error_pct': (1 - 1 / ratio) * 100,
                                            'PurA': PuA, 'PurB': PuB}
    dump('S5_area', area)

# =============================================================================
# S6. Stage-cut sensitivity of the bias
# =============================================================================
if RUN['S6'] or RUN['S13']:
    log('S6 stage-cut sensitivity')
    m = make_matrimid_T1(FK_CO2=FK_P)
    sc = {}
    for y, pf in [(0.05, 10), (0.15, 10), (0.30, 10), (0.15, 20)]:
        sc[f'{y}_{pf}'] = {str(phi): rec_pair(m, y, pf, 1.0, factor=phi) for phi in [2.0, 5.0, 10.0]}
    dump('S6_stagecut', sc)

# =============================================================================
# S7. Correcting pressure and thresholds at baseline (10 bar)
# =============================================================================
PC_LIST = [0.86, 0.88, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.5, 1.64, 2.0, 2.5, 3.0, 3.5]
if RUN['S7'] or RUN['S13']:
    log('S7 p* and thresholds')
    ps, thr = {}, {}
    for FK in [FK_P, FK_L, FK_H]:
        m = make_matrimid_T1(FK_CO2=FK)
        ps[str(FK)] = [pstar(m, pc / 10) for pc in PC_LIST]
        thr[str(FK)] = {'structural_pCO2_bar': 10 * threshold_y(m, 10, P_FLOOR, 0.04, 0.14),
                        'practical_pCO2_bar': 10 * threshold_y(m, 10, 20.0, 0.10, 0.30)}
        log(f"  F*K={FK}: {thr[str(FK)]}")
    dump('S7_pstar', {'pCO2_bar': PC_LIST, 'pstar_bar': ps, 'thresholds': thr})

# =============================================================================
# S8. Regime map and practical threshold vs stage cut
# =============================================================================
PHIS = [2, 3, 4, 5, 6, 7.5, 10, 15]
PGRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2]
if RUN['S8'] or RUN['S13']:
    log('S8 regime map')
    m = make_matrimid_T1(FK_CO2=FK_P)
    reg = []
    for phi in PHIS:
        for p in PGRID:
            r = rec_pair(m, p / 10, 10, P_FLOOR, factor=phi)
            reg.append({'phi': phi, 'pCO2': p, 'dRec_floor': r['dRec'], 'stage_cut_B': r['stage_cut_B']})
    prac = []
    for phi in PHIS:
        yp = threshold_y(m, 10, 20.0, 0.08, 0.40, factor=phi)
        scp = rec_pair(m, yp, 10, 1.0, factor=phi)['stage_cut_B'] if np.isfinite(yp) else float('nan')
        prac.append({'phi': phi, 'practical_pCO2_bar': 10 * yp, 'stage_cut_B': scp})
    dump('S8_regime', {'map': reg, 'practical': prac})

# =============================================================================
# S9. Axial profiles (mechanism)
# =============================================================================
def axial_profile(m, y, phi, pf=10.0):
    r = run_module_robust(m, 'B', cond_of(y, pf, F_in_rule(y, pf, factor=phi)), return_profiles=True)
    Fcf, Fnf, Fcp, Fnp = r['profiles']
    yc = Fcf / (Fcf + Fnf); xc = Fcp / np.maximum(Fcp + Fnp, 1e-20)
    P = np.array([P_eff_exact(m, 'CO2', yc[k] * pf, (1 - yc[k]) * pf, xc[k], 1 - xc[k], 'N2') for k in range(len(yc))])
    dCH = np.array([_sorb(m, 'CO2', yc[k] * pf, (1 - yc[k]) * pf, 'N2')[1] - _sorb(m, 'CO2', xc[k], 1 - xc[k], 'N2')[1]
                    for k in range(len(yc))])
    return np.linspace(0, 1, len(yc)), P, dCH, r['mb_error_pct']

if RUN['S9'] or RUN['S13']:
    log('S9 axial profiles')
    m = make_matrimid_T1(FK_CO2=FK_P); Pfl = P_vac(m, 'CO2', P_FLOOR)
    prof = {}
    for phi, y in [(5, 0.03), (5, 0.06), (5, 0.15), (10, 0.03), (10, 0.06), (2, 0.06), (3, 0.03)]:
        z, P, dCH, mb = axial_profile(m, y, phi)
        prof[f'phi{phi}_pCO2_{y * 10:.1f}'] = {'min_P': float(P.min()), 'z_at_min': float(z[np.argmin(P)]),
                                               'frac_below_floor': float(np.mean(P < Pfl)), 'min_dCH': float(dCH.min()),
                                               'MB': mb}
    dump('S9_profiles', {'P_floor': Pfl, 'cases': prof})

# =============================================================================
# S10. Decomposition and F(N2) sensitivity
# =============================================================================
if RUN['S10'] or RUN['S13']:
    log('S10 decomposition')
    m_full = make_matrimid_T1(FK_CO2=FK_P, bN2=0.087)
    m_noN2 = make_matrimid_T1(FK_CO2=FK_P, bN2=1e-9)          # N2 competition switched off
    dec = {}
    for y, pf in CONDS6:
        c = cond_of(y, pf, F_in_rule(y, pf))
        RA = run_module_robust(design_basis(m_full, 1.0), 'A', c)['CO2_recovery']
        RBno = run_module_robust(m_noN2, 'B', c)['CO2_recovery']
        row = {'RecA': RA, 'RecB_noN2': RBno}
        for FN2 in [0.0, 0.1, 0.2]:
            RB = run_module_robust(make_matrimid_T1(FK_CO2=FK_P, bN2=0.087, F_N2=FN2), 'B', c)['CO2_recovery']
            row[f'N2_part_FN2_{FN2}'] = RBno - RB
            if FN2 == 0.0:
                row['total'] = RA - RB; row['CO2_part'] = RA - RBno
        row['check_total_minus_sum'] = row['total'] - row['CO2_part'] - row['N2_part_FN2_0.0']
        dec[f'{y}_{pf}'] = row
    dump('S10_decomposition', dec)

# =============================================================================
# S11. Temperature
# =============================================================================
EaP_target = {'CO2': 8.1, 'N2': 20.2}     # David et al. JMS 378, 4 bar, 30-100 C

def make_matrimid_T(T, FK_CO2, Ea_D_CO2, Ea_D_N2, bN2=0.087, F_N2=0.0, F_shift=0.0):
    """Parameters at T [K]. F(CO2) may drift linearly: F(T) = F(Tref)(1 + F_shift (T-Tref)/(373.15-Tref))."""
    m0 = make_matrimid_T1(FK_CO2=FK_CO2, bN2=bN2, F_N2=F_N2)
    kD = {g: m0['kD'][g] * kD_scale(g, T) for g in ['CO2', 'N2']}
    CH = {g: m0['CH'][g] * CH_scale(g, T) for g in ['CO2', 'N2']}
    b = dict(m0['b'])                                             # b constant with T (Scholes et al.)
    DD = {'CO2': m0['DD']['CO2'] * np.exp(-Ea_D_CO2 / Rg * (1.0 / T - 1.0 / T_ref)),
          'N2': m0['DD']['N2'] * np.exp(-Ea_D_N2 / Rg * (1.0 / T - 1.0 / T_ref))}
    F = {'CO2': m0['F']['CO2'] * (1.0 + F_shift * (T - T_ref) / (373.15 - T_ref)), 'N2': F_N2}
    DH = {g: DD[g] * F[g] for g in ['CO2', 'N2']}
    m = {'kD': kD, 'CH': CH, 'b': b, 'DD': DD, 'DH': DH, 'F': F, 'P_pure': {'CO2': 1.0, 'N2': 1.0}}
    m['P_pure'] = {'CO2': P_vac(m, 'CO2', 1.0), 'N2': P_vac(m, 'N2', 1.0)}
    return m

def EaP_model(gas, FK, EaC, EaN, p_ref=4.0, T_lo=303.15, T_hi=373.15):
    P_lo = P_vac(make_matrimid_T(T_lo, FK, EaC, EaN), gas, p_ref)
    P_hi = P_vac(make_matrimid_T(T_hi, FK, EaC, EaN), gas, p_ref)
    return -Rg * np.log(P_hi / P_lo) / (1.0 / T_hi - 1.0 / T_lo)

def calibrate_Ea_D(gas, FK):
    def resid(Ea):
        return EaP_model(gas, FK, Ea if gas == 'CO2' else 25.0, Ea if gas == 'N2' else 25.0) - EaP_target[gas]
    return brentq(resid, -50.0, 80.0, xtol=1e-6)

def dRec_T(Tc, EaC, EaN, F_shift=0.0, y=0.15, pf=10.0):
    m = make_matrimid_T(Tc + 273.15, FK_P, EaC, EaN, F_shift=F_shift)
    c = cond_of(y, pf, F_in_rule(y, pf))
    RB = run_module_robust(m, 'B', c)['CO2_recovery']
    RA = run_module_robust(design_basis(m, 1.0), 'A', c)['CO2_recovery']
    return RA - RB, P_vac(m, 'CO2', 1.0) / P_vac(m, 'N2', 1.0)

if RUN['S11'] or RUN['S13']:
    log('S11 temperature')
    EaC, EaN = calibrate_Ea_D('CO2', FK_P), calibrate_Ea_D('N2', FK_P)
    temp = {'Ea_D_CO2': EaC, 'Ea_D_N2': EaN, 'Ea_D_CO2_at_FK0859': calibrate_Ea_D('CO2', FK_DAVID), 'series': {}}
    for Tc in [30, 35, 55, 75, 80, 100]:
        d, a = dRec_T(Tc, EaC, EaN); temp['series'][Tc] = {'dRec': d, 'alpha_ideal_1bar': a}
    fsh = {}
    for s in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        v = {Tc: dRec_T(Tc, EaC, EaN, F_shift=s)[0] for Tc in [30, 75, 100]}
        fsh[s] = {'drop_30_75_pct': 100 * (v[30] - v[75]) / v[30], 'drop_30_100_pct': 100 * (v[30] - v[100]) / v[30]}
    eas = {}
    for f in [0.8, 1.0, 1.2]:
        v30, v100 = dRec_T(30, f * EaC, f * EaN)[0], dRec_T(100, f * EaC, f * EaN)[0]
        eas[f] = {'dRec30': v30, 'dRec100': v100, 'drop_30_100_pct': 100 * (v30 - v100) / v30}
    temp['F_shift_sensitivity'] = fsh; temp['EaD_sensitivity'] = eas
    dump('S11_temperature', temp)

# =============================================================================
# S12. Plasticization check (worst case 35 % CO2, 20 bar)
# =============================================================================
if RUN['S12']:
    m = make_matrimid_T1(FK_CO2=FK_P)
    pc, pn = 0.35 * 20, 0.65 * 20
    CD, CHc = _sorb(m, 'CO2', pc, pn, 'N2')
    dump('S12_plasticization', {'C_total_cm3_per_cm3': CD + CHc, 'fraction_of_47': (CD + CHc) / 47.0})

# =============================================================================
# S13. Figures
# =============================================================================
if RUN['S13']:
    log('S13 figures')
    import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.tri as mtri
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9,
                         'legend.fontsize': 7, 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True,
                         'ytick.right': True, 'savefig.dpi': 600, 'mathtext.fontset': 'stix'})
    C_B, C_R, C_G, C_K = '#1f4e9c', '#c0392b', '#7f8c8d', '#222222'
    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), bbox_inches='tight', pad_inches=0.06)
        plt.close(fig)
    lab6 = [f"{int(y * 100)}%\n{pf} bar" for y, pf in CONDS6]

    # Fig 1 calibration
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2)); p = np.linspace(0, 14, 200)
    rt = lambda FK: Mshape(p, 0, FK) / Mshape(0, 0, FK)
    ax[0].fill_between(p, rt(FK_L), rt(FK_H), color=C_B, alpha=0.18, lw=0, label='90% CI')
    ax[0].plot(p, rt(FK_P), color=C_B, label=f'Calibrated, F·K = {FK_P}')
    ax[0].plot(p, rt(FK_DAVID), '--', color=C_R, label=f'Single-lab, F·K = {FK_DAVID}')
    R12 = 4.8 / 5.7
    ax[0].errorbar([0, 12], [1, R12], yerr=[0, R12 * np.sqrt(2) * 0.03], fmt='o', color=C_K, ms=4, capsize=2.5, label='Bos et al. (1999)')
    ax[0].set_xlabel('CO$_2$ pressure [bar]'); ax[0].set_ylabel('$P(p)/P(0)$ [–]'); ax[0].set_title('(a)', loc='left')
    pp = np.linspace(0.8, 6.3, 200)
    def ec(FK35):
        FK25 = FK35 / K_RATIO_25_35; f = Mshape(0.15 * pt_g, 0.85 * pt_g, FK25); a = P_g @ f / (f @ f)
        return a * Mshape(0.15 * pp, 0.85 * pp, FK25)
    ax[1].fill_between(pp, ec(FK_L), ec(FK_H), color=C_B, alpha=0.18, lw=0, label='90% CI')
    ax[1].plot(pp, ec(FK_P), color=C_B, label=f'Calibrated, F·K = {FK_P}')
    ax[1].plot(pp, ec(FK_DAVID), '--', color=C_R, label=f'Single-lab, F·K = {FK_DAVID}')
    ax[1].plot(pt_g, P_g, 'o', color=C_K, ms=4, label='Esposito et al. (2019)')
    ax[1].set_xlabel('Total feed pressure [bar] (15/85 CO$_2$/N$_2$, 25 °C)'); ax[1].set_ylabel('CO$_2$ permeability [Barrer]')
    ax[1].set_title('(b)', loc='left')
    for a_ in ax: a_.legend(frameon=False, fontsize=6.2, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(); save(fig, 'Fig1_FK_calibration')

    # Fig 2 distributions
    fig, ax = plt.subplots(figsize=(3.5, 2.7)); bins = np.linspace(0, 0.6, 61)
    for s, c, lab in [(CAL_SAMPLES['bos3'], C_G, 'Bos-alone (MC, σ = 3%)'), (CAL_SAMPLES['esp35'], C_R, 'Esposito-alone (35 °C)'),
                      (CAL_SAMPLES['comb3'], C_B, 'Combined')]:
        ax.hist(s, bins=bins, density=True, histtype='stepfilled', alpha=0.28, color=c)
        ax.hist(s, bins=bins, density=True, histtype='step', color=c, lw=1.0, label=f'{lab}: median {np.median(s):.3f}')
    cb3 = CAL_SAMPLES['comb3']
    ax.axvspan(np.percentile(cb3, 5), np.percentile(cb3, 95), color=C_B, alpha=0.08, lw=0)
    ax.set_xlim(0, 0.6); ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.set_xlabel('F·K (CO$_2$) [–]'); ax.set_ylabel('Probability density'); ax.legend(frameon=False, loc='upper right', fontsize=5.5)
    fig.tight_layout(); save(fig, 'Fig2_FK_distributions')

    # Fig 3 bias and area (manuscript Fig. 3)
    B = RESULTS['S4_bias']; x = np.arange(6); w = 0.36
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={'width_ratios': [1.25, 1]})
    for j, (key, c, lab) in enumerate([('dRec', C_B, 'Recovery bias'), ('dPur', C_R, 'Purity bias')]):
        v = np.array([B[str(FK_P)][f'{y}_{pf}'][key] for y, pf in CONDS6])
        lo = np.array([B[str(FK_L)][f'{y}_{pf}'][key] for y, pf in CONDS6]); hi = np.array([B[str(FK_H)][f'{y}_{pf}'][key] for y, pf in CONDS6])
        ax[0].bar(x + (j - 0.5) * w, v, w, color=c, alpha=0.85, label=lab, capsize=2,
                  yerr=[v - np.minimum(lo, hi), np.maximum(lo, hi) - v], error_kw={'lw': 0.8})
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab6, fontsize=7); ax[0].set_ylabel('Bias, Model A − Model B [pp]')
    ax[0].legend(frameon=False); ax[0].set_title('(a)', loc='left')
    A_ = RESULTS['S5_area']; ks = [(0.05, 10), (0.15, 10), (0.15, 20), (0.30, 10), (0.30, 20)]; x2 = np.arange(len(ks))
    ax[1].bar(x2 - 0.18, [A_[f'{y}_{pf}_50']['area_error_pct'] for y, pf in ks], 0.36, color='#34495e', label='Target recovery 50%')
    ax[1].bar(x2 + 0.18, [A_[f'{y}_{pf}_75']['area_error_pct'] for y, pf in ks], 0.36, color='#95a5a6', label='Target recovery 75%')
    ax[1].set_xticks(x2); ax[1].set_xticklabels([f"{int(y * 100)}%\n{pf} bar" for y, pf in ks], fontsize=7)
    ax[1].set_ylabel('Module undersizing [% of true area]'); ax[1].legend(frameon=False, loc='upper left'); ax[1].set_title('(b)', loc='left')
    fig.tight_layout(); save(fig, 'Fig3_bias_and_area')

    # Fig 4 p* at baseline
    PS = RESULTS['S7_pstar']['pstar_bar']; pc = np.array(PC_LIST)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    thrP = RESULTS['S7_pstar']['thresholds'][str(FK_P)]
    ax.axvspan(0.6, thrP['structural_pCO2_bar'], color=C_R, alpha=0.10, lw=0)
    ax.axvspan(thrP['structural_pCO2_bar'], thrP['practical_pCO2_bar'], color='#f39c12', alpha=0.10, lw=0)
    ax.axvspan(thrP['practical_pCO2_bar'], 3.6, color='#27ae60', alpha=0.10, lw=0)
    lo_b = np.fmin(PS[str(FK_L)], PS[str(FK_H)]); hi_b = np.fmax(PS[str(FK_L)], PS[str(FK_H)]); ok = np.isfinite(lo_b) & np.isfinite(hi_b)
    ax.fill_between(pc[ok], lo_b[ok], hi_b[ok], color=C_B, alpha=0.2, lw=0, label='F·K 90% CI')
    ax.plot(pc, PS[str(FK_P)], 'o-', color=C_B, ms=3.5, label=f'F·K = {FK_P}')
    ax.axhline(20, ls=':', color=C_K, lw=0.9); ax.set_yscale('log'); ax.set_xlim(0.6, 3.6); ax.set_ylim(8, 2e3)
    ax.set_xlabel('Feed CO$_2$ partial pressure [bar]'); ax.set_ylabel('Correcting measurement pressure $p^{*}$ [bar]')
    ax.legend(frameon=False, loc='center right', fontsize=7); fig.tight_layout(); save(fig, 'Fig4_pstar_baseline')

    # Fig 5 regime map
    R = RESULTS['S8_regime']; X = np.array([d['pCO2'] for d in R['map']]); Y = np.array([d['stage_cut_B'] for d in R['map']])
    Z = np.array([d['dRec_floor'] for d in R['map']]); tri = mtri.Triangulation(X, Y)
    fig, ax = plt.subplots(figsize=(4.6, 3.3)); lim = np.max(np.abs(Z))
    cf = ax.tricontourf(tri, Z, levels=np.linspace(-lim, lim, 21), cmap='RdBu_r')
    ax.tricontour(tri, Z, levels=[0.0], colors=C_K, linewidths=1.6); ax.plot(X, Y, '.', color='k', ms=2, alpha=0.4)
    pr = sorted(R['practical'], key=lambda d: d['phi'])
    ax.plot([d['practical_pCO2_bar'] for d in pr], [d['stage_cut_B'] for d in pr], 's-', color='#27ae60', ms=3.5, lw=1.4,
            label='Practical threshold ($p^{*}$ = 20 bar)')
    base = sorted([(d['pCO2'], d['stage_cut_B']) for d in R['map'] if d['phi'] == 5])
    ax.plot([b_[0] for b_ in base], [b_[1] for b_ in base], '--', color=C_G, lw=1, label='Baseline operating line (φ = 5)')
    ax.plot([], [], color=C_K, lw=1.6, label='Structural boundary ($\\Delta Rec_{\\infty}$ = 0)')
    cbar = fig.colorbar(cf, ax=ax, pad=0.02); cbar.set_label('$\\Delta Rec$ at Henry floor [pp]')
    ax.set_xlim(0, 2.0); ax.set_ylim(3, 45); ax.set_xlabel('Feed CO$_2$ partial pressure [bar]'); ax.set_ylabel('Stage cut (Model B) [%]')
    ax.legend(frameon=False, loc='upper right', fontsize=6.3); fig.tight_layout(); save(fig, 'Fig5_regime_map')

    # Fig 6 decomposition
    Dd = RESULTS['S10_decomposition']
    co2 = np.array([Dd[f'{y}_{pf}']['CO2_part'] for y, pf in CONDS6])
    n2 = {f: np.array([Dd[f'{y}_{pf}'][f'N2_part_FN2_{f}'] for y, pf in CONDS6]) for f in [0.0, 0.1, 0.2]}
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.bar(x, co2, 0.55, color=C_R, alpha=0.85, label='CO$_2$ self-suppression')
    ax.bar(x, n2[0.0], 0.55, bottom=np.maximum(co2, 0), color=C_B, alpha=0.85, label='N$_2$ competition, F(N$_2$) = 0')
    ax.plot(x, co2 + n2[0.0], 'D', color=C_K, ms=4, label='Total, F(N$_2$) = 0')
    ax.plot(x, co2 + n2[0.1], '^', mfc='none', color=C_K, ms=4.5, label='Total, F(N$_2$) = 0.1')
    ax.plot(x, co2 + n2[0.2], 'v', mfc='none', color=C_K, ms=4.5, label='Total, F(N$_2$) = 0.2')
    ax.axhline(0, color=C_K, lw=0.6); ax.set_xticks(x); ax.set_xticklabels(lab6, fontsize=7)
    ax.set_ylabel('Recovery bias contribution [pp]'); ax.legend(frameon=False, fontsize=6.3, loc='upper right')
    fig.tight_layout(); save(fig, 'Fig6_decomposition')

    # Fig 7 temperature
    Tm = RESULTS['S11_temperature']; Ts = np.array(sorted(Tm['series'])); dR = np.array([Tm['series'][t]['dRec'] for t in Ts])
    al = np.array([Tm['series'][t]['alpha_ideal_1bar'] for t in Ts]); v = Ts <= 75
    fsh, eas = Tm['F_shift_sensitivity'], Tm['EaD_sensitivity']; d30 = Tm['series'][30]['dRec']
    env = {75: (d30 * (1 - max(f['drop_30_75_pct'] for f in fsh.values()) / 100), d30 * (1 - min(f['drop_30_75_pct'] for f in fsh.values()) / 100)),
           100: (min(d30 * (1 - max(f['drop_30_100_pct'] for f in fsh.values()) / 100), min(e['dRec100'] for e in eas.values())),
                 max(d30 * (1 - min(f['drop_30_100_pct'] for f in fsh.values()) / 100), max(e['dRec100'] for e in eas.values())))}
    fig, ax = plt.subplots(figsize=(3.6, 2.8)); ax2 = ax.twinx(); ax.axvspan(75, 102, color=C_G, alpha=0.12, lw=0)
    ax.plot(Ts[v], dR[v], 'o-', color=C_B, ms=4); ax.plot(Ts[Ts >= 75], dR[Ts >= 75], 'o--', mfc='white', color=C_B, ms=4)
    for t, (lo, hi) in env.items():
        i = list(Ts).index(t); ax.errorbar(t, dR[i], yerr=[[dR[i] - lo], [hi - dR[i]]], color=C_B, capsize=2.5, lw=0.9)
    ax2.plot(Ts[v], al[v], 's-', color=C_R, ms=3.8); ax2.plot(Ts[Ts >= 75], al[Ts >= 75], 's--', mfc='white', color=C_R, ms=3.8)
    ax.set_xlabel('Temperature [°C]'); ax.set_ylabel('Recovery bias [pp]', color=C_B); ax2.set_ylabel('Ideal CO$_2$/N$_2$ selectivity [–]', color=C_R)
    fig.tight_layout(); save(fig, 'Fig7_temperature')

    # Fig S1 stage cut and grid
    S6, S3 = RESULTS['S6_stagecut'], RESULTS['S3_numerics']['grid']
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for (k, d), mk in zip(S6.items(), ['o', 's', '^', 'D']):
        y_, pf_ = k.split('_'); ph = sorted(d, key=float)
        ax[0].plot([d[p_]['stage_cut_B'] for p_ in ph], [d[p_]['dRec'] for p_ in ph], mk + '-', ms=4,
                   label=f"{int(float(y_) * 100)}%, {pf_} bar")
    ax[0].set_xlabel('Stage cut (Model B) [%]'); ax[0].set_ylabel('Recovery bias at $p_{ref}$ = 1 bar [pp]'); ax[0].legend(frameon=False)
    Ns = np.array([25, 50, 100, 200]); dN = np.array([S3[n]['dRec'] for n in Ns])
    ax[1].plot(1 / Ns, dN, 'o-', color=C_B, ms=4, label='Computed')
    ax[1].plot(0, S3['dRec_extrapolated_first_order'], '*', color=C_R, ms=8, label='First-order extrapolation')
    ax[1].set_xlabel('1/N [–]'); ax[1].set_ylabel('Recovery bias [pp]'); ax[1].legend(frameon=False)
    fig.tight_layout(); save(fig, 'FigS1_stagecut_and_grid')

    # Fig S2 mechanism
    m = make_matrimid_T1(FK_CO2=FK_P); Pfl = P_vac(m, 'CO2', P_FLOOR)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for phi, y, lab, c, ls in [(5, 0.03, 'pCO$_2$ 0.3 bar, φ = 5 (structural)', C_R, '-'),
                               (5, 0.15, 'pCO$_2$ 1.5 bar, φ = 5 (correctable)', C_B, '-'),
                               (2, 0.06, 'pCO$_2$ 0.6 bar, φ = 2 (correctable)', '#27ae60', '--')]:
        z, P, dCH, _ = axial_profile(m, y, phi)
        ax[0].plot(z, P, ls, color=c, label=lab); ax[1].plot(z, dCH, ls, color=c, label=lab)
    ax[0].axhline(Pfl, color=C_K, lw=0.9, ls=':', label=f'Henry floor ({Pfl:.2f} Barrer)')
    ax[1].axhline(0, color=C_K, lw=0.7, ls=':', label='Zero Langmuir driving force')
    ax[0].set_xlabel('Axial position $z/L$ [–]'); ax[0].set_ylabel('Local CO$_2$ permeability, Model B [Barrer]')
    ax[1].set_xlabel('Axial position $z/L$ [–]'); ax[1].set_ylabel('$C_{H,feed} - C_{H,perm}$ [cm$^3$(STP) cm$^{-3}$]')
    for a_ in ax: a_.legend(frameon=False, fontsize=6.0)
    fig.tight_layout(); save(fig, 'FigS2_axial_mechanism')

log('done')
