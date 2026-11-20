"""Reading the model's carbonate state out of a run, and the E1/E2 post-processes.

The carbonate system has two degrees of freedom, but the model fixes pCO2,
alkalinity and pH independently. If we want to close the carbonate system on
saved model output, we need to pick two and drop the third:

    E1       keep pCO2 + carbonate alkalinity, solve for pH   -> pH rises, calcite precipitates
    E2       keep pCO2 + pH, solve for alkalinity   -> the alk export is capped
    co2_req  keep pH + alkalinity, solve for pCO2   -> diagnostic only

`co2_req` is not an experiment because soil CO2 is a CLM5 driver, not a state
variable. 

E1 holds E0's dissolution fixed, so E1/E0 is the export effect alone --- the
dissolution and transport responses to thermodynamic pH would need an in-model
simulation rather than an offline analysis.

The `_cb` fields close the untreated (initial or background) porewater first.
This ensures that only the added carbonate alkalinity from weathering contributes
to deviations from carbonate system closure.
"""
from __future__ import annotations

import numpy as np

from . import carbonate as carb

# np.trapz was removed in NumPy 2.0, replaced by np.trapezoid 
_trapz = getattr(np, "trapezoid", None) or np.trapz

#: Row order of `diag.alk`, set by calc_pH.m:18.
ALK_ROWS = ["alk", "alk0", "-alk_acid", "N_alk", "2Ca", "2Mg", "K", "Na", "CO2"]

M_CO2 = 44.0

#: How much less Ca E1 may precipitate than E0, as a fraction of the Ca budget
#: (see `e1`). Max seen 9.4e-6; the redistribution bug it guards against was O(0.1).
PRECIP_RTOL = 1e-4


def neta(co2_ppm, salinity=35.0, tmp=17.0):
    """Ocean re-equilibration efficiency. Port of `neta.m`.

    mol CO2 per mol monovalent cation; divalent cations count 2x this.
    """
    co2 = np.asarray(co2_ppm, dtype=float)
    s_term = salinity * 10 ** -3.009 + 10 ** -1.519
    g13 = s_term * np.log(co2) - salinity * 10 ** -2.1
    h13 = co2 * tmp * (salinity * 10 ** -7.501 - 10 ** -5.598)
    i13 = tmp * 10 ** -2.337
    return g13 - h13 - i13 + 10 ** -0.102


def eta_series(run):
    """Ocean efficiency at every output time, as `mass_balance.m:229` builds it.

    neta is evaluated at the atmospheric CO2 of the moment and applied to the
    cumulative export; a run-mean eta is off by up to 1.4% on long runs.
    Salinity and temperature are `neta`'s defaults (`mod5.m:121`).
    """
    co2 = np.atleast_1d(run.co2_air).astype(float)
    x = (np.arange(1, len(co2) + 1) - 0.5) / 12.0     # mass_balance.m:229
    return neta(np.interp(run.t, x, co2))


class Run:
    """Convenience view over one saved run, with the carbonate state unpacked."""

    def __init__(self, mat):
        self.mat = mat
        d = mat["totals"].diag
        self.d = d
        self.z = np.atleast_1d(d.z).astype(float)          # 31 nodes, m
        self.t = np.atleast_1d(d.t).astype(float)          # years since application
        self.pH = np.atleast_2d(d.pH).astype(float)        # 31 x nt
        self.pH_avg = np.atleast_1d(d.pH_avg).astype(float)
        self.Ca_sol = np.atleast_2d(d.Ca_sol).astype(float)
        self.Ca_sat = np.atleast_2d(d.Ca_sat).astype(float)
        self.Ca_precip = np.atleast_2d(d.Ca_precip).astype(float)
        self.q = np.atleast_1d(d.q).astype(float)          # m/yr, post-bucket
        self.alk_budget = np.asarray(d.alk, dtype=float)   # 9 x 31 x nt
        self.phi = float(d.phi)
        self.buffer = str(d.buffer)
        self.co2_air = np.atleast_1d(mat["inps"].co2).astype(float)

    # -- the two alkalinities -----------------------------------
    @property
    def alk_total(self):
        """Total alkalinity as the model computes it (calc_pH.m:14), mol/L."""
        return self.alk_budget[ALK_ROWS.index("alk")]

    @property
    def del_alk(self):
        """The NON-Ca alkalinity that CaSat is parameterised on, mol/L.

        `solution2CaSat.m` builds this as alk0 - alk_acid + 2Mg + K + Na + N_alk,
        i.e. total alk minus the 2Ca term. 
        """
        return self.alk_total - self.alk_budget[ALK_ROWS.index("2Ca")]

    @property
    def cation_alk(self):
        """Feedstock contribution to alkalinity, eq/L -- 2Ca+2Mg+K+Na.

        The charge `mass_balance.m:232` credits with CDR once it has left.
        """
        b = self.alk_budget
        return sum(b[ALK_ROWS.index(k)] for k in ("2Ca", "2Mg", "K", "Na"))

    @property
    def alk_ctrl(self):
        """Alkalinity with the feedstock removed, mol/L -- alk0 - alk_acid + N_alk.

        alk is treated as a linear budget, so this is the same soil without ERW. E2 uses it
        to credit only the added alkalinity.
        """
        return self.alk_total - self.cation_alk

    @property
    def soil_co2(self):
        """Soil pCO2 the chemistry actually saw, ppm -- time-varying, 31 x nt.

        Not `diag.soil_CO2` (the static spin-up profile used only to set alk0);
        fd15.m:210 rebuilds this every timestep.
        """
        return self.alk_budget[ALK_ROWS.index("CO2")]

    # -- non-Ca base cations, mol/L (the Mg budget row is 2*Mg) -----------------
    @property
    def Mg_sol(self):
        return self.alk_budget[ALK_ROWS.index("2Mg")] / 2.0

    @property
    def K_sol(self):
        return self.alk_budget[ALK_ROWS.index("K")]

    @property
    def Na_sol(self):
        return self.alk_budget[ALK_ROWS.index("Na")]

    # -- integrals ----------------------------------------------------------
    def column_mol_per_m2(self, conc):
        """Depth-integrate a mol/L field to mol/m2, as run_simple4/mass_balance do.

        The 1000 is upstream's: mol/L integrated over metres is mol per mm per m2.
        """
        return _trapz(conc, self.z, axis=0) * 1000.0 * self.phi

    def export_mol_per_m2(self, conc_base):
        """Cumulative export of a base-node concentration, mol/m2.

        Trapezoid version of fd15.m:495's accumulation, on the regular output
        grid rather than the adaptive one. 
        """
        return np.concatenate([[0.0], np.cumsum(
            0.5 * (conc_base[1:] + conc_base[:-1])
            * 0.5 * (self.q[1:] + self.q[:-1])
            * np.diff(self.t) * 1000.0)])


def recompute_ca_sat(run, buffered):
    """Re-derive the calcite ceiling from the run's own state.

    With buffered=True this must reproduce `run.Ca_sat` -- that is the validation
    that our carbonate port matches the model's solver. With buffered=False it is
    the E1 ceiling.
    """
    return carb.ca_sat(run.soil_co2, run.del_alk, buffered=buffered)


def e1(run, ca_sat_thermo=None):
    """The E1 recompute: E0's dissolution, but with carbonate-consistent speciation.

    Returns E0 and E1 carbon numbers in mol/m2 and t CO2/ha. Pass `ca_sat_thermo`
    from `recompute_ca_sat(run, buffered=False)` to skip a second bisection.
    """
    if ca_sat_thermo is None:
        ca_sat_thermo = recompute_ca_sat(run, buffered=False)

    ca_total = run.Ca_sol + run.Ca_precip                 # conserved: only the split moves
    ca_sol_new = np.minimum(ca_sat_thermo, ca_total)
    ca_precip_new = np.maximum(ca_total - ca_sat_thermo, 0.0)

    out = {}
    out["ca_sat_thermo"] = ca_sat_thermo
    out["Ca_sol_e1"] = ca_sol_new
    out["Ca_precip_e1"] = ca_precip_new

    # Mg, K and Na never precipitate, so their export is untouched by E1.
    o = run.mat["totals"].Outflow
    mg, k, na = (np.atleast_1d(getattr(o, f)).astype(float) for f in ("Mg", "K", "Na"))
    ca_out_e0 = np.atleast_1d(o.Ca).astype(float)
    ca_out_e1 = run.export_mol_per_m2(ca_sol_new[-1, :])

    # Only the Ca export changes; scale the model's own accumulation by the ratio
    # our integration gives, so the E0 baseline stays exactly the model's number.
    ca_out_e0_ours = run.export_mol_per_m2(run.Ca_sol[-1, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ca_out_e0_ours > 0, ca_out_e1 / ca_out_e0_ours, 1.0)
    out["ca_export_ratio"] = ratio
    ca_out_e1_scaled = ca_out_e0 * ratio

    eta = eta_series(run)          # time-varying, as mass_balance.m:232

    # ---- Ca CONSERVATION ---------------------------------------------------
    # Dissolution fixes total Ca (precip + dissolved + exported), so the E1
    # precipitate must also absorb the Ca that E0 exported but E1 keeps.
    # Re-splitting only the in-column Ca would drop it and understate E1's CDR.
    sol_e0 = run.column_mol_per_m2(run.Ca_sol)
    sol_e1 = run.column_mol_per_m2(ca_sol_new)
    precip_e0 = run.column_mol_per_m2(run.Ca_precip)
    precip_e1 = precip_e0 + (sol_e0 - sol_e1) + (ca_out_e0 - ca_out_e1_scaled)

    tot_e0 = precip_e0 + sol_e0 + ca_out_e0
    tot_e1 = precip_e1 + sol_e1 + ca_out_e1_scaled
    resid = np.max(np.abs(tot_e1 - tot_e0) / np.maximum(tot_e0, 1e-12))
    if resid > 1e-10:
        raise AssertionError(f"Ca not conserved between E0 and E1: {resid:.2e}")
    out["ca_conservation_resid"] = resid

    # Relative to the Ca budget, not zero: the high-delAlk ceiling collapse (see
    # docstring) legitimately lets E1 precipitate a trace less than E0.
    deficit = float(np.min((precip_e1 - precip_e0) / np.maximum(tot_e0, 1e-12)))
    out["precip_deficit_rel"] = min(deficit, 0.0)
    if deficit < -PRECIP_RTOL:
        raise AssertionError(
            f"E1 precipitates {-deficit:.2%} less Ca than E0 -- too large to be the "
            "high-delAlk ceiling flip; check the redistribution in `e1`")

    out["cdr_e0_mol"] = eta * (na + k + 2 * (mg + ca_out_e0)) + precip_e0
    out["cdr_e1_mol"] = eta * (na + k + 2 * (mg + ca_out_e1_scaled)) + precip_e1
    out["sic_e0_mol"] = precip_e0
    out["sic_e1_mol"] = precip_e1
    out["eta"] = eta
    for key in ("cdr_e0", "cdr_e1", "sic_e0", "sic_e1"):
        out[key + "_t_ha"] = out[key + "_mol"] * M_CO2 * 0.01
    return out


#: Base node (z = 0.15 m), where `fd15.m:495` accumulates the outflow.
BASE = -1


def e2(run):
    """The E2 recompute: E0's pH and pCO2, carbonate-consistent alkalinity.

    The mirror of `e1`. At fixed pH and pCO2 the carbonate alkalinity is fixed
    (`carb.carbonate_alk`), so water leaving the base node can carry at most

        dAlk = Alk(pH_model, pCO2) - Alk(pH_ctrl, pCO2)

    of added alkalinity; times q, integrated, that caps the export term. Returns
    E0 and E2 carbon numbers plus diagnostics.
    """
    ph, co2 = run.pH, run.soil_co2
    cat = run.cation_alk
    ph_ctrl = carb.buffered_pH(run.alk_ctrl, co2)

    # The first stored timestep has soil CO2 = 0, which makes both alkalinities
    # meaningless. `carb.ca_sat` zeroes that column too.
    live = co2 > 0
    alk_cap = np.where(live, carb.carbonate_alk(ph, co2)
                             - carb.carbonate_alk(ph_ctrl, co2), 0.0)

    out = {}
    out["alk_cap"] = alk_cap
    out["alk_added"] = cat
    out["pH_ctrl"] = ph_ctrl
    out["omega_buffered"] = carb.omega(run.Ca_sol, ph, co2)

    # The cap is not sign-constrained; clipped below, and reported in case it is.
    out["frac_cap_negative"] = float(np.mean(alk_cap[live & (cat > 0)] < 0)) \
        if np.any(live & (cat > 0)) else 0.0

    # ---- the export ratio ---------------------------------------------------
    # As E1's `f_export`: a ratio of our own integrations, so their bias against
    # the model's accumulation cancels, as does eta.
    ours_e0 = run.export_mol_per_m2(cat[BASE])
    ours_e2 = run.export_mol_per_m2(np.maximum(alk_cap[BASE], 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(ours_e0 > 0, ours_e2 / np.maximum(ours_e0, 1e-300), 1.0)   # t=0 is 0/0
    if np.any(f < 0):
        raise AssertionError("E2 export ratio went negative")
    out["f_export_e2"] = f

    o = run.mat["totals"].Outflow
    mg, k, na = (np.atleast_1d(getattr(o, fld)).astype(float) for fld in ("Mg", "K", "Na"))
    ca = np.atleast_1d(o.Ca).astype(float)
    charge_e0 = na + k + 2 * (mg + ca)          # the model's own accumulation, eq/m2
    out["alk_export_e0_mol"] = charge_e0
    out["alk_export_e2_mol"] = charge_e0 * f

    # ---- carbon -------------------------------------------------------------
    eta = eta_series(run)          # time-varying, as mass_balance.m:232
    precip = run.column_mol_per_m2(run.Ca_precip)     # E2 does not move this
    out["eta"] = eta
    out["cdr_e0_mol"] = eta * charge_e0 + precip
    out["cdr_e2_mol"] = eta * charge_e0 * f + precip
    out["sic_e0_mol"] = precip
    # Export-only: drops the calcite credit, whose durability is a judgement call.
    out["cdr_export_e0_mol"] = eta * charge_e0
    out["cdr_export_e2_mol"] = eta * charge_e0 * f
    for key in ("cdr_e0", "cdr_e2", "sic_e0", "cdr_export_e0", "cdr_export_e2"):
        out[key + "_t_ha"] = out[key + "_mol"] * M_CO2 * 0.01
    return out


def closed_background(alk_ctrl, alk_added, soil_co2, pH):
    """Figure 2's two projections, closing the untreated background porewater
    carbonate system first.

    This closes the untreated porewater:

        pH_ctrl     = buffered_pH(alk_ctrl, soil_co2)        # as in `e2`
        alk_ctrl_cb = carbonate_alk(pH_ctrl, soil_co2)
        alk_tot     = alk_ctrl_cb + alk_added

        pH_cb       keep soil CO2, solve for pH         partner of pH_thermo
        co2_req_cb  keep the model's pH, solve for CO2  partner of co2_req

    No calcite step, so compare `pH_cb` with `pH_thermo`, not `pH_e1`. Takes
    arrays, so it runs from a .mat or from `state.zarr`.
    """
    co2 = np.asarray(soil_co2, dtype=float)
    ph_ctrl = carb.buffered_pH(alk_ctrl, co2)
    alk_ctrl_cb = carb.carbonate_alk(ph_ctrl, co2)
    alk_tot = alk_ctrl_cb + np.asarray(alk_added, dtype=float)
    co2_req_cb = carb.alk_pH_to_co2(alk_tot, pH)
    # Formed on the native grid, as co2_req_ratio; t=0 has soil_co2 = 0 -> NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(co2 > 0, co2_req_cb / co2, np.nan)
    return dict(alk_ctrl_cb=alk_ctrl_cb,
                pH_cb=carb.alk_to_pH(alk_tot, co2),
                co2_req_cb=co2_req_cb,
                co2_req_cb_ratio=ratio)


def carbonate_fields(run, with_e1=True, with_e2=True, with_cb=True):
    """Every carbonate diagnostic and E1/E2/CB field for one run, on the run's own grid.

    What `postprocess-narrow` reduces into `full.zarr`. Nothing is reduced in time: 
    fields are 31 x nt or nt on fd15's 100/yr grid, and the caller reduces to annual per key (below).

    `alk_cap` is exponential in pH, so capping annual means understates it by
    22-67 percent due to jensen's inequality. `f_export_e2` is accumulated here before 
    reduction. The annual-mean `alk_cap`/`alk_added` in `full.zarr` are for plotting.
    
    E1, E2 and CB re-close what E0's buffered configuration leaves open, so turn
    them off (`with_e1`, `with_e2`, `with_cb`) for any other run.

    KEYS, AND HOW EACH REDUCES TO ANNUAL

    diagnostics (any experiment)
      Ca_sat_thermo  31 x nt  mol/L   calcite ceiling at equilibrium.      MEAN
      pH_thermo      31 x nt          pH implied by E0's alkalinity.       MEAN
      pH_diag        31 x nt          E0's stored pH on this grid. Compare
                                      `pH_e1` to this, not to full.zarr's
                                      monthly-reduced `pH`.                MEAN
      omega          31 x nt          Omega at E0's Ca, alk and pCO2; spans
                                      decades. Over values > 0.       GEOMETRIC
      co2_req        31 x nt   ppm    pCO2 implied by E0's pH and alk.
                                      Can be negative: never geometric.    MEAN
      co2_req_ratio  31 x nt          co2_req / soil_co2, formed on this grid.
                                      Over values > 0.                GEOMETRIC

    E1 (with_e1)
      Ca_sol_e1      31 x nt  mol/L   dissolved Ca after the re-split.     MEAN
      pH_e1          31 x nt          pH at E1's alkalinity.               MEAN
      f_export       nt               multiply E0's cumulative `cdr_export`
                                      by this.                      END OF YEAR
      Ca_export_e1   nt      mol/m2   cumulative Ca export.         END OF YEAR
      cdr_precip_e1  nt    t CO2/ha   cumulative pedogenic calcite, an
                                      absolute.                     END OF YEAR
      sic_e0_t_ha    nt    t CO2/ha   our E0 precipitate; assert it against the
                                      model's `CO2_mass_precip`.    END OF YEAR
      e1_over_e0     nt               E1/E0 from absolutes; agrees with
                                      `f_export` to 0.1-0.2%.       END OF YEAR

    E2 (with_e2)
      alk_cap        31 x nt   eq/L   added alkalinity E0's pH and pCO2
                                      permit. Plotting only (see above).   MEAN
      alk_added      31 x nt   eq/L   what it actually carries, 2Ca+2Mg+K+Na;
                                      alk_added/alk_cap is the overcharge. MEAN
      pH_ctrl        31 x nt          the same soil, feedstock removed.    MEAN
      omega_buffered 31 x nt          Omega at E0's Ca and buffered pH; ~1,
                                      so E2 leaves calcite alone.     GEOMETRIC
      f_export_e2    nt               multiply E0's cumulative `cdr_export`
                                      by this.                      END OF YEAR
      alk_export_e2  nt      eq/m2    cumulative capped export.     END OF YEAR

    CB, closed background (with_cb)
      alk_ctrl_cb      31 x nt  eq/L  alk_ctrl closed at the untreated pH
                                      and soil CO2.                        MEAN
      pH_cb            31 x nt        pH of alk_ctrl_cb + alk_added; partner
                                      to pH_thermo.                        MEAN
      co2_req_cb       31 x nt   ppm  pCO2 of alk_ctrl_cb + alk_added at
                                      the model's pH; partner to co2_req.  MEAN
      co2_req_cb_ratio 31 x nt        co2_req_cb / soil_co2, on this grid.
                                      Over values > 0.                GEOMETRIC

    scalars
      eta, phi                        to re-derive the accounting from
                                      `state.zarr` without the .mat.
      ca_conservation_resid           from `e1`; ~4e-16.
      precip_deficit_rel              from `e1`; 0 except where the ceiling
                                      collapses at high delAlk.
      frac_cap_negative               from `e2`; 0 on every site run.
      frac_ceiling_collapsed          fraction of live node-times where
                                      buffered Ca_sat has collapsed below
                                      the thermodynamic one.

    `Ca_precip_e1` is omitted (it is `Ca_sol + Ca_precip - Ca_sol_e1`, but does
    not integrate to `cdr_precip_e1`, which also reclaims Ca E0 exported).
    `cdr_precip_e2` is omitted because E2 does not move it.
    """
    ca_sat_thermo = recompute_ca_sat(run, buffered=False)
    ph_thermo = carb.alk_to_pH(run.alk_total, run.soil_co2)

    # The third projection: the pCO2 the run's own pH and alkalinity imply.
    co2_req = carb.alk_pH_to_co2(run.alk_total, run.pH)
    # Exact inverse, so this is an algebra check, not a tolerance.
    scale = float(np.nanmax(np.abs(run.alk_total)))
    resid = float(np.nanmax(np.abs(carb.carbonate_alk(run.pH, co2_req) - run.alk_total))) / scale
    assert resid < 1e-9, f"co2_req does not invert carbonate_alk: {resid:.2e}"

    # Formed on the 100/yr grid: a ratio of annual means is not the annual mean
    # of the ratio. t=0 has soil_co2 = 0 -> NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        co2_req_ratio = np.where(run.soil_co2 > 0, co2_req / run.soil_co2, np.nan)

    out = dict(Ca_sat_thermo=ca_sat_thermo,
               pH_thermo=ph_thermo,
               pH_diag=run.pH,
               omega=carb.omega(run.Ca_sol, ph_thermo, run.soil_co2),
               co2_req=co2_req,
               co2_req_ratio=co2_req_ratio,
               phi=run.phi,
               eta=float(np.mean(eta_series(run))))

    if with_e2:
        r2 = e2(run)
        for k in ("alk_cap", "alk_added", "pH_ctrl", "omega_buffered",
                  "f_export_e2", "frac_cap_negative"):
            out[k] = r2[k]
        out["alk_export_e2"] = r2["alk_export_e2_mol"]
        out["e2_over_e0"] = np.where(r2["cdr_e0_t_ha"] > 0,
                                     r2["cdr_e2_t_ha"] / r2["cdr_e0_t_ha"], 1.0)

        # E2 passes the precipitate through untouched, valid only while Omega at
        # the model's pH is ~1 where it precipitates. Two checks:
        #  - the median, excluding node-times where the buffered ceiling has
        #    collapsed (see `e1`): Omega is ~3e-5 there by construction. That
        #    undersaturation is harmless to other outputs so we report it but 
        #    don't raise an error / warning.
        #  - no sustained supersaturation. Fresh rock causes brief Omega > 1
        #    transients (longest 32 substeps over e0-conus-1gt), so test how
        #    long Omega stays above 1.05, not a tail quantile.
        live = run.soil_co2 > 0
        collapsed = ca_sat_thermo > run.Ca_sat
        out["frac_ceiling_collapsed"] = float(np.mean(collapsed[live])) if live.any() else 0.0

        PERSIST = 50   # substeps (fd15's 100/yr grid); observed max is 32
        om = r2["omega_buffered"]
        g = (run.Ca_precip > 0) & live & np.isfinite(om)
        intact = g & ~collapsed
        if intact.any():
            med = float(np.median(om[intact]))
            assert 0.99 < med < 1.01, f"Omega at the model's own pH is {med:.4f}, not ~1"

        high_t = (g & (om > 1.05)).any(axis=0)   # any node exceeding, per substep
        if high_t.any():
            idx = np.flatnonzero(high_t)
            run_len = max(len(s) for s in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1))
            assert run_len <= PERSIST, (
                f"Omega stays above 1.05 for {run_len} consecutive substeps -- "
                "not a brief post-application transient")

    if with_cb:
        # run.pH, not a rebuild, so co2_req_cb is built on exactly the pH co2_req is.
        out.update(closed_background(run.alk_ctrl, run.cation_alk, run.soil_co2, run.pH))

    if not with_e1:
        return out

    r1 = e1(run, ca_sat_thermo=ca_sat_thermo)
    out["Ca_sol_e1"] = r1["Ca_sol_e1"]
    out["pH_e1"] = carb.alk_to_pH(run.del_alk + 2 * r1["Ca_sol_e1"], run.soil_co2)

    # The export credit is a charge sum; only the Ca term moves, so eta cancels.
    o = run.mat["totals"].Outflow
    mg, k, na = (np.atleast_1d(getattr(o, f)).astype(float) for f in ("Mg", "K", "Na"))
    ca = np.atleast_1d(o.Ca).astype(float)
    charge_e0 = na + k + 2 * (mg + ca)
    charge_e1 = na + k + 2 * (mg + ca * r1["ca_export_ratio"])
    with np.errstate(divide="ignore", invalid="ignore"):
        # t=0 is 0/0 for every term; 1.0 leaves E0's own zero untouched.
        out["f_export"] = np.where(charge_e0 > 0, charge_e1 / charge_e0, 1.0)
        out["e1_over_e0"] = np.where(r1["cdr_e0_t_ha"] > 0,
                                     r1["cdr_e1_t_ha"] / r1["cdr_e0_t_ha"], 1.0)
    out["Ca_export_e1"] = ca * r1["ca_export_ratio"]
    out["cdr_precip_e1"] = r1["sic_e1_t_ha"]
    out["sic_e0_t_ha"] = r1["sic_e0_t_ha"]
    out["ca_conservation_resid"] = r1["ca_conservation_resid"]
    out["precip_deficit_rel"] = r1["precip_deficit_rel"]
    return out
