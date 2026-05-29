"""
Synthetic dataset generator that mimics CST Particle Studio cold-state output
of a magnetron injection gun (MIG) for a 170 GHz / 1 MW-class gyrotron.

Physics behind the surrogate response surface (used only to fabricate a
realistic-looking dataset for paper figures; the real workflow expects the
user's CST CSV):

    - Magnetic compression  b = B_cav / B_cat ~ (R_c/R_g)^2   (Busch theorem)
    - Pitch factor          alpha = v_perp / v_para
                            ~ sqrt( b * f_geom(theta_c, geometry) )
                            scaled with V_a^{-1/2} and (I_c/I_L)^{1/2}
    - Velocity spread       delta_alpha ~ A * (w_belt / R_c)
                            + B * |dr_cat| + C * |dr_gun| + thermal jitter
    - Guiding-center radius R_g = R_c / sqrt(b) + small offset perturbations
    - Beam thickness        Delta_r = w_belt / sqrt(b) + emission non-uniformity

11 inputs (cold-state design variables), 4 outputs (beam-quality figures-of-merit).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc

RNG_SEED = 20260529

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# 1. Input parameter ranges (physically realistic for a 170 GHz, 1 MW MIG)
# ---------------------------------------------------------------------------
INPUT_RANGES = {
    # name             :  (low, high, unit, latex)
    "w_belt":           (3.0,  9.0,   "mm",  r"$w_{\rm belt}$"),
    "R_cat":            (45.0, 60.0,  "mm",  r"$R_{\rm cat}$"),
    "theta_cat":        (20.0, 35.0,  "deg", r"$\theta_{\rm cat}$"),
    "V_a":              (60.0, 90.0,  "kV",  r"$V_a$"),
    "I_c":              (40.0, 90.0,  "A",   r"$I_c$"),
    "R_anode_cav":      (18.0, 26.0,  "mm",  r"$R_{\rm a,cav}$"),
    "R_anode_arc":      (10.0, 18.0,  "mm",  r"$R_{\rm a,arc}$"),
    "dr_cat_anode":     (-0.5, 0.5,   "mm",  r"$\Delta r_{c-a}$"),
    "dz_cat_anode":     (-0.8, 0.8,   "mm",  r"$\Delta z_{c-a}$"),
    "dr_gun_field":     (-1.0, 1.0,   "mm",  r"$\Delta r_{g-B}$"),
    "dz_gun_field":     (-2.0, 2.0,   "mm",  r"$\Delta z_{g-B}$"),
}

OUTPUT_NAMES = {
    "alpha":         (r"$\alpha = v_\perp/v_\parallel$",            "-"),
    "delta_alpha":   (r"$\delta\alpha/\alpha$",                      "%"),
    "R_g":           (r"$R_g$",                                       "mm"),
    "Delta_r":       (r"$\Delta r$",                                  "mm"),
}

INPUT_NAMES = list(INPUT_RANGES.keys())
N_IN, N_OUT = len(INPUT_NAMES), len(OUTPUT_NAMES)


# ---------------------------------------------------------------------------
# 2. LHS sampling
# ---------------------------------------------------------------------------
def latin_hypercube(n_samples: int, seed: int = RNG_SEED) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=N_IN, seed=seed)
    u = sampler.random(n_samples)
    lows = np.array([INPUT_RANGES[k][0] for k in INPUT_NAMES])
    highs = np.array([INPUT_RANGES[k][1] for k in INPUT_NAMES])
    return qmc.scale(u, lows, highs)


# ---------------------------------------------------------------------------
# 3. Physics-flavored response surface
# ---------------------------------------------------------------------------
def physics_response(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return Y[N,4] = (alpha, delta_alpha[%], R_g[mm], Delta_r[mm])."""
    (w_belt, R_cat, theta_cat, V_a, I_c,
     R_a_cav, R_a_arc,
     dr_ca, dz_ca, dr_gB, dz_gB) = [X[:, i] for i in range(N_IN)]

    # --- magnetic compression -------------------------------------------------
    # cavity field B_cav fixed by frequency; cathode field B_cat = B_cav / b.
    # Effective compression depends on R_cat and anode geometry:
    R_g_nom = (R_cat * 0.18) * (1.0 + 0.012 * (R_a_cav - 22.0))
    b = (R_cat / np.maximum(R_g_nom, 1e-3)) ** 2
    b *= 1.0 + 0.004 * (theta_cat - 27.5)

    # --- pitch factor alpha ---------------------------------------------------
    # Pierce-style synchronous: alpha grows with sqrt(b), drops with V_a, mild
    # increase with I_c (space-charge defocusing of v_parallel).
    geom_corr = 1.0 + 0.0008 * (R_a_arc - 14.0) ** 2
    alpha = (
        0.255 * np.sqrt(b)
        * (75.0 / V_a) ** 0.5
        * (1.0 + 0.0015 * (I_c - 65.0))
        * geom_corr
        - 0.04 * np.abs(dr_ca)
        - 0.02 * np.abs(dz_ca)
    )

    # --- velocity spread (in % of alpha) -------------------------------------
    delta_alpha = (
        1.5
        + 12.0 * (w_belt / R_cat)            # belt-width contribution
        + 3.0 * np.abs(dr_ca)
        + 1.5 * np.abs(dz_ca)
        + 4.0 * np.abs(dr_gB)
        + 1.0 * np.abs(dz_gB)
        + 0.015 * (I_c - 40.0)
    )

    # --- guiding-center radius R_g -------------------------------------------
    R_g = R_cat / np.sqrt(b) + 0.3 * dr_gB + 0.05 * dr_ca

    # --- beam thickness at cavity entrance -----------------------------------
    Delta_r = (
        w_belt / np.sqrt(b)
        + 0.06 * np.abs(dr_ca)
        + 0.18 * np.abs(dr_gB)
        + 0.02 * w_belt * (delta_alpha / 5.0)
    )

    # --- numerical-fidelity noise (CST mesh + tracking jitter) ---------------
    alpha       += rng.normal(0, 0.005, size=alpha.size)
    delta_alpha += rng.normal(0, 0.08,  size=alpha.size)
    R_g         += rng.normal(0, 0.03,  size=alpha.size)
    Delta_r     += rng.normal(0, 0.02,  size=alpha.size)

    return np.stack([alpha, delta_alpha, R_g, Delta_r], axis=1)


# ---------------------------------------------------------------------------
# 4. Build dataset
# ---------------------------------------------------------------------------
def build_dataset(n: int = 1500, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = latin_hypercube(n, seed=seed)
    Y = physics_response(X, rng)

    cols_in  = INPUT_NAMES
    cols_out = list(OUTPUT_NAMES.keys())
    df = pd.DataFrame(
        np.concatenate([X, Y], axis=1),
        columns=cols_in + cols_out,
    )

    # remove obvious unphysical rows
    mask = (
        (df["alpha"] > 0.6) & (df["alpha"] < 2.5)
        & (df["delta_alpha"] > 0) & (df["delta_alpha"] < 20)
        & (df["R_g"] > 5) & (df["R_g"] < 14)
        & (df["Delta_r"] > 0.2) & (df["Delta_r"] < 4.0)
    )
    return df.loc[mask].reset_index(drop=True)


if __name__ == "__main__":
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    out = build_dataset(n=1500)
    out.to_csv(out_dir / "cst_cold_state_dataset.csv", index=False)
    print(f"Generated {len(out)} rows -> data/cst_cold_state_dataset.csv")
    print(out.describe().T[["mean", "std", "min", "max"]].round(3))
