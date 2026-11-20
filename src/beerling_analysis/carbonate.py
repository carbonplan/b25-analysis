"""B25's carbonate system ported alongside the thermodynamic counterpart.

Why we use a port instead of a library call (i.e., pyco2sys)
-----------------------------------------
We wanted the carbonate comparison between B25 and our calculations to be as 
apples-to-apples as possible. Therefore, we copied the carbonate constants from
`Ca_sat_not_buffered.m`.

HOW WELL THE PORT AGREES (measured in notebook 01, section 1)
  thermodynamic branch, vs B25's own tabulated grid : 4.7e-8 max rel err
  buffered branch, vs the model's saved run output         : 9.8e-5 median
For context, the model does not perform a bisection at runtime, and instead 
interpolates from a pre-built table. So we don't expect exact agreement. 

The worst agreement is in the buffered branch at extreme calcite saturation. The 
buffered branch disagrees by >1 percent at 1.1 percent of points and, in all cases, 
Ca_sat is about 700x below the dissolved Ca concentration. This leads calcite to 
precipitate regardless of the value used. 

Distinguishing between buffered and not_buffered cases
-----------------------------------------
`Ca_sat_buffered.m` and `Ca_sat_not_buffered.m` are mostly the same file. Both define

    Alk(H) = Kh*rho*CO2*(a1(H) + 2*a2(H))/a0(H) + Kw/H - H     # true carbonate alk

and solve `Alk(H) = 2*Ca(H) + delAlk` for H by bisection, with `Ca(H) = Ksp/CO3(H)`.
The buffered file then overwrites that with the Nelson and Su sigmoid before solving.
Consequently, the two cases differ only in the alk-pH relation that the bisection is 
solved against.

Note also that `delAlk` here EXCLUDES Ca -- `solution2CaSat.m` comments Ca out of
it because it is handled separately via calcite saturation then added back into alk later.
"""
from __future__ import annotations

import numpy as np

# --- constants, verbatim from Ca_sat_not_buffered.m ------------------------
KSP = 3.3e-9
KW = 10 ** -14.16
KH = 10 ** -1.41
K1 = 10 ** -6.38
K2 = 10 ** -10.38
RHO = 1e-6          # ppm -> atm

# --- Nelson & Su buffer constants, from alk2pH_buffer.m --------------------
PH_MIN = 1.0
PH_MAX_CAP = 13.0
B_SHAPE = -26.6e-3  # mol/L; the 'b' of the sigmoid
A_MID = 0.0
MAX_ALK = 0.1       # the alk at which pHmax is evaluated


def _a0(H):
    return 1.0 / (1.0 + K1 / H + K1 * K2 / H**2)


def _a1(H):
    return 1.0 / (1.0 + H / K1 + K2 / H)


def _a2(H):
    return 1.0 / (1.0 + H / K2 + H**2 / (K1 * K2))


def carbonate_alk(pH, co2_ppm):
    """True carbonate alkalinity (mol/L) at a given pH and soil pCO2 (ppm).

    This is `pH2Alk_no_buffer.m`.
    """
    H = 10.0 ** (-np.asarray(pH, dtype=float))
    c = KH * RHO * np.asarray(co2_ppm, dtype=float)
    return c * (_a1(H) + 2 * _a2(H)) / _a0(H) + KW / H - H


def _bisect_pH(f, lo=0.1, hi=12.0, tol=1e-10, itmax=100):
    """Bisection on pH, matching the loop in Ca_sat_*.m (same bounds, same tol)."""
    lo = np.broadcast_to(np.asarray(lo, float), np.shape(f(np.asarray(1.0)))).copy()
    hi = np.broadcast_to(np.asarray(hi, float), lo.shape).copy()
    for _ in range(itmax):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        hi = np.where(v > 0, mid, hi)
        lo = np.where(v > 0, lo, mid)
        if np.all(np.abs(v) < tol):
            break
    return 0.5 * (lo + hi)


def alk_to_pH(alk, co2_ppm):
    """Carbonate-equilibrium pH for a given total alkalinity and soil pCO2.

    The thermodynamic `alk2pH`. Inverts `carbonate_alk` by bisection.
    """
    alk = np.asarray(alk, dtype=float)
    co2 = np.broadcast_to(np.asarray(co2_ppm, dtype=float), alk.shape)

    def resid(pH):
        return carbonate_alk(pH, co2) - alk

    return _bisect_pH(lambda pH: resid(pH))


def alk_pH_to_co2(alk, pH):
    """Soil pCO2 (ppm) that puts a given alkalinity and pH on the carbonate curve.

    Note, this is not sign-constrained. The CO2 term is strictly positive. But in
    cases where alk is below the carbonate floor at a given pH the answer would have
    to be negative (implying a strong acid input). This happens at 0.3 percent of 
    node-years and, in each case, the model's total alk also appears negative.
    """
    H = 10.0 ** (-np.asarray(pH, dtype=float))
    alk = np.asarray(alk, dtype=float)
    return (alk - KW / H + H) / (KH * RHO * (_a1(H) + 2 * _a2(H)) / _a0(H))


def ph_max(co2_ppm):
    """The sigmoid's upper asymptote: theoretical pH at alk = 0.1 mol/L.

    `alk2pH_buffer.m` sets `pHmax = alk2pH_theoretical(CO2, 0.1)` -- the buffer's
    ceiling is pinned to carbonate equilibrium at a hard-coded alkalinity, which
    is the only place soil CO2 enters the buffered pH at all.
    """
    co2 = np.asarray(co2_ppm, dtype=float)
    return alk_to_pH(np.full(co2.shape, MAX_ALK), co2)


def buffered_pH(alk, co2_ppm):
    """The Nelson & Su sigmoid: `alk2pH_buffer.m`."""
    alk = np.asarray(alk, dtype=float)
    a = ph_max(co2_ppm) - PH_MIN
    return PH_MIN + a / (1.0 + np.exp((alk - A_MID) / B_SHAPE))


def buffered_alk(pH, co2_ppm):
    """Inverse of the sigmoid: `pH2Alk_buffer.m`. Clipped to +-MAX_ALK as upstream."""
    pH = np.asarray(pH, dtype=float)
    a = ph_max(co2_ppm) - PH_MIN
    with np.errstate(divide="ignore", invalid="ignore"):
        alk = A_MID + B_SHAPE * np.log((a - pH + PH_MIN) / (pH - PH_MIN))
    alk = np.where(pH >= PH_MIN + a, MAX_ALK, alk)
    alk = np.where(pH <= PH_MIN, -MAX_ALK, alk)
    return np.clip(alk, -MAX_ALK, MAX_ALK)


def ca_sat(co2_ppm, del_alk, buffered):
    """Dissolved Ca at calcite saturation (mol/L).

    Port of `Ca_sat_buffered.m` / `Ca_sat_not_buffered.m`.

    Parameters
    ----------
    co2_ppm : soil pCO2, ppm
    del_alk : the NON-Ca alkalinity, mol/L (alk0 - alk_acid + 2Mg + K + Na + N_alk).
              Ca is deliberately excluded so it can be handled via calcite saturation state
              as `solution2CaSat.m` handles it.
    buffered : True  -> solve against the Nelson & Su sigmoid (what the model does)
               False -> solve against true carbonate alkalinity (what E1 uses)

    Returns Ca at Omega = 1. Values are roughly 100-1000x apart between the two branches
    over the model's operating range.
    """
    co2 = np.asarray(co2_ppm, dtype=float)
    dA = np.asarray(del_alk, dtype=float)
    co2, dA = np.broadcast_arrays(co2, dA)

    if buffered:
        a = ph_max(co2) - PH_MIN

        def alk_of(pH):
            # Note on matlab semantics -- The sigmoid inverse is only real for
            # pH in (1, 1+a); outside it the log argument goes negative. MATLAB
            # returns a COMPLEX log there and `if eq>0` compares the real part,
            # so the bisection keeps moving. numpy returns NaN.
            #
            # For real x, real(log(x)) == log(|x|), so taking the modulus
            # reproduces MATLAB's real part exactly without complex arithmetic.
            with np.errstate(divide="ignore", invalid="ignore"):
                return A_MID + B_SHAPE * np.log(np.abs((a - pH + PH_MIN) / (pH - PH_MIN)))
    else:
        def alk_of(pH):
            return carbonate_alk(pH, co2)

    def ca_of(pH):
        H = 10.0 ** (-pH)
        return KSP / (KH * RHO * co2 / _a0(H) * _a2(H))

    def resid(pH):
        with np.errstate(over="ignore", invalid="ignore"):
            return alk_of(pH) - (2 * ca_of(pH) + dA)

    pH = _bisect_pH(resid)
    out = ca_of(pH)
    # Degenerate column: the model's first stored timestep has soil CO2 = 0 (no
    # respiration profile yet), which sends CO3 -> 0 and Ca = Ksp/CO3 -> inf.
    # 
    return np.where(co2 > 0, out, 0.0)


def omega(ca, pH, co2_ppm):
    """Calcite saturation state at an explicit Ca, pH and soil pCO2."""
    H = 10.0 ** (-np.asarray(pH, dtype=float))
    co3 = KH * RHO * np.asarray(co2_ppm, dtype=float) / _a0(H) * _a2(H)
    return np.asarray(ca, dtype=float) * co3 / KSP
