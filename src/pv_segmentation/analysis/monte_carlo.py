"""Monte Carlo uncertainty propagation + Sobol decomposition (paper §Monte Carlo, H2)."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.io_utils import load_config, write_csv, UNITS
from analysis.economics import lcoe, npv

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze


def _draw_param(dist_cfg: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw samples from a configured distribution with optional clipping."""
    d = dist_cfg.get("distribution", "normal")
    if d == "normal":
        x = rng.normal(dist_cfg["mean"], dist_cfg["sd"], size=n)
    elif d == "lognormal":
        # Zero-mean lognormal so the multiplier has a median of 1.0
        x = np.exp(rng.normal(0.0, dist_cfg["sigma"], size=n))
    elif d == "beta":
        x = rng.beta(dist_cfg["a"], dist_cfg["b"], size=n)
    elif d == "uniform_int":
        x = rng.integers(dist_cfg["low"], dist_cfg["high"] + 1, size=n).astype(float)
    else:
        raise ValueError(f"Unknown distribution type: {d}")

    if "clip" in dist_cfg:
        lo, hi = dist_cfg["clip"]
        x = np.clip(x, lo, hi)
    return x


def _kappa_for_unit(mc_cfg: dict, unit: str) -> dict:
    """Retrieve unit-specific kappa parameters or fall back to district mean/sd."""
    kappa_cfg = mc_cfg.get("kappa", {})
    per_unit = kappa_cfg.get("per_unit") or {}
    if unit in per_unit:
        return per_unit[unit]
    return {"mean": kappa_cfg.get("mean", 0.93), "sd": kappa_cfg.get("sd", 0.04)}


def _sample_inputs(rng: np.random.Generator, n: int, mc_cfg: dict, unit: str | None = None) -> np.ndarray:
    """Return an (n, 6) array of canonical parameter draws in order: kappa, rsuf, G, capex, tariff, r."""
    # Resolve per-unit kappa if unit name is provided
    kappa_params = _kappa_for_unit(mc_cfg, unit) if unit else mc_cfg["kappa"]
    kappa_cfg_resolved = {**mc_cfg["kappa"], **kappa_params}

    kappa = _draw_param(kappa_cfg_resolved, n, rng)
    rsuf = _draw_param(mc_cfg["rsuf"], n, rng)
    G = _draw_param(mc_cfg["irradiance"], n, rng)
    capex = _draw_param(mc_cfg["capex"], n, rng)
    tau = _draw_param(mc_cfg["tariff"], n, rng)
    r = _draw_param(mc_cfg["discount"], n, rng)

    return np.column_stack([kappa, rsuf, G, capex, tau, r])


def _simulate(
    draws: np.ndarray,
    unit_capacity_mwp: float,
    unit_yield_kwh_per_kwp: float,
    eco_cfg: dict,
    module_cfg: dict,
) -> dict[str, np.ndarray]:
    """Vectorized simulation for a single unit. Returns dictionary of (n,) arrays."""
    kappa, rsuf, G, capex, tau, r = draws.T
    cap_mwp = unit_capacity_mwp * kappa * (rsuf / module_cfg["rsuf_conservative"])
    cap_kwp = cap_mwp * 1000.0
    e1 = unit_yield_kwh_per_kwp * cap_kwp * G

    lcoes = np.array([
        lcoe(
            eco_cfg["capex_per_kwp"] * cx, eco_cfg["opex_per_kwp_yr"],
            e1[i], eco_cfg["lifetime"], r[i], eco_cfg["degradation"], cap_kwp[i],
        )
        for i, cx in enumerate(capex)
    ])
    
    npvs = np.array([
        npv(
            capex[i] * eco_cfg["capex_per_kwp"], eco_cfg["opex_per_kwp_yr"],
            unit_yield_kwh_per_kwp * G[i], eco_cfg["tariff"] * tau[i],
            eco_cfg["lifetime"], r[i], eco_cfg["degradation"], cap_kwp[i],
        )
        for i in range(len(capex))
    ])

    return {
        "capacity_mwp": cap_mwp,
        "generation_gwh": e1 / 1e6,
        "lcoe": lcoes,
        "npv": npvs,
        "avoided_kt": e1 * eco_cfg["emission_factor_kg_per_kwh"] / 1e6,
    }


def sobol_indices(sim_fn, n_base: int, mc_cfg: dict, seed: int = 0, unit: str | None = None) -> dict:
    """SALib Saltelli sampling; returns first- and total-order indices per input."""
    problem = {
        "num_vars": 6,
        "names": ["kappa", "rsuf", "G", "capex", "tariff", "r"],
        "bounds": [[0, 1]] * 6,
    }
    rng = np.random.default_rng(seed)
    X = _sample_inputs(rng, 2 * n_base, mc_cfg, unit=unit)
    Y = sim_fn(X)
    Si = sobol_analyze.analyze(problem, Y, print_to_console=False)
    return {
        "S1": dict(zip(problem["names"], Si["S1"])),
        "ST": dict(zip(problem["names"], Si["ST"])),
        "S1_conf": dict(zip(problem["names"], Si["S1_conf"])),
        "ST_conf": dict(zip(problem["names"], Si["ST_conf"])),
    }


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo uncertainty propagation and Sobol decomposition.")
    parser.add_argument("--economics", type=Path, default=Path("configs/economics.yaml"))
    parser.add_argument("--mc", type=Path, default=Path("configs/monte_carlo.yaml"))
    parser.add_argument("--generation", type=Path, default=Path("results/analysis/generation_by_unit.csv"))
    parser.add_argument("--capacity", type=Path, default=Path("results/analysis/capacity_by_unit.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis/mc"))
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--n-sobol", type=int, default=1024)
    args = parser.parse_args()

    if not args.economics.exists():
        raise FileNotFoundError(f"Economics config not found: {args.economics}")
    if not args.mc.exists():
        raise FileNotFoundError(f"MC config not found: {args.mc}")
    if not args.generation.exists():
        raise FileNotFoundError(f"Generation CSV not found: {args.generation}")
    if not args.capacity.exists():
        raise FileNotFoundError(f"Capacity CSV not found: {args.capacity}")

    eco_cfg = load_config(args.economics)
    mc_cfg = load_config(args.mc)

    gen = pd.read_csv(args.generation).set_index("unit")
    cap = pd.read_csv(args.capacity).set_index("unit")

    rng = np.random.default_rng(mc_cfg.get("seed", 42))
    args.out.mkdir(parents=True, exist_ok=True)

    dist_rows, sobol_rows = [], []
    
    units_to_process = [u for u in UNITS if u in cap.index and u in gen.index]
    if not units_to_process:
        units_to_process = list(cap.index.intersection(gen.index))

    print(f"=== Running Monte Carlo Simulation for {len(units_to_process)} units (N={args.n}) ===")
    for unit in units_to_process:
        cap_mwp = float(cap.loc[unit, "capacity_mwp"])
        y = float(gen.loc[unit, "yield_kwh_per_kwp"])
        
        draws = _sample_inputs(rng, args.n, mc_cfg, unit=unit)
        sims = _simulate(draws, cap_mwp, y, eco_cfg["economics"], eco_cfg["module"])
        
        for k, v in sims.items():
            dist_rows.append({
                "unit": unit,
                "variable": k,
                "p2_5": float(np.percentile(v, 2.5)),
                "median": float(np.median(v)),
                "p97_5": float(np.percentile(v, 97.5)),
            })

        def sim_fn_cap(X, _c=cap_mwp, _y=y):
            return _simulate(X, _c, _y, eco_cfg["economics"], eco_cfg["module"])["capacity_mwp"]

        def sim_fn_gen(X, _c=cap_mwp, _y=y):
            return _simulate(X, _c, _y, eco_cfg["economics"], eco_cfg["module"])["generation_gwh"]

        seed_val = mc_cfg.get("seed", 42)
        S_cap = sobol_indices(sim_fn_cap, args.n_sobol, mc_cfg, seed=seed_val, unit=unit)
        S_gen = sobol_indices(sim_fn_gen, args.n_sobol, mc_cfg, seed=seed_val, unit=unit)

        for name in ("kappa", "rsuf", "G", "capex", "tariff", "r"):
            sobol_rows.append({
                "unit": unit,
                "input": name,
                "ST_capacity": S_cap["ST"][name],
                "ST_capacity_ci": S_cap["ST_conf"][name],
                "ST_generation": S_gen["ST"][name],
                "ST_generation_ci": S_gen["ST_conf"][name],
            })

    write_csv(pd.DataFrame(dist_rows), args.out / "distributions.csv")
    sobol_df = pd.DataFrame(sobol_rows)
    write_csv(sobol_df, args.out / "sobol.csv")

    print("\n=== Sobol Decomposition Summary (kappa & rsuf) ===")
    print(sobol_df.query("input in ['kappa', 'rsuf']").to_string(index=False))
    print(f"\nResults successfully written to {args.out}/")


if __name__ == "__main__":
    main()