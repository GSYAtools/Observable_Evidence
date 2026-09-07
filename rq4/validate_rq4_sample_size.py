#!/usr/bin/env python3
"""
RQ4 - Sequential cross-system sample-size stability analysis

Automatically detects the largest balanced perturbed sample size available
across providers/cells and compares the previous predefined checkpoint against
that new reference. Designed for iterative rounds 10 -> 15 -> 20 -> 25 -> 30.

Calibration uses two disjoint nominal samples of size n. Therefore a reference
size n requires at least 2*n nominal generations per provider/scenario.
The script stops rather than silently changing the calibration procedure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

PROVIDERS = ["openai", "gemini", "qwen"]
CHECKPOINTS = [3, 5, 7, 10, 15, 20, 25, 30]
PRIMARY_CHECKPOINTS = [10, 15, 20, 25, 30]
ALPHA = 0.05
B_CALIBRATION = 2000
B_PERTURBATION = 1000
BASE_SEED = 20260829
EXPECTED_EMBEDDING_DIM = 384
EXPECTED_SCENARIOS = 20
PERTURBATION_FAMILIES = ["semantic", "context_removal", "surface"]
LAMBDA_VALUES = [0.25, 0.50, 0.75, 1.00]
MAX_MEDIAN_RELATIVE_DEVIATION = 0.10
MIN_EXCEEDANCE_AGREEMENT = 0.90
MIN_RANK_STABILITY = 0.90
EPS = 1e-12


def condition_seed(provider, scenario_id, family, lam, n):
    text = f"{provider}|{scenario_id}|{family}|{lam}|{n}"
    value = sum((i + 1) * ord(c) for i, c in enumerate(text))
    return (BASE_SEED + value) % (2**32 - 1)


def calibration_seed(provider, scenario_id, n):
    text = f"cal|{provider}|{scenario_id}|{n}"
    value = sum((i + 1) * ord(c) for i, c in enumerate(text))
    return (BASE_SEED + value) % (2**32 - 1)


def load_provider_data(embeddings_dir: Path, provider: str):
    ep = embeddings_dir / f"rq4_{provider}_embeddings.npy"
    ip = embeddings_dir / f"rq4_{provider}_embedding_index.csv"
    if not ep.exists() or not ip.exists():
        raise FileNotFoundError(f"Missing embedding state for {provider}: {ep} / {ip}")
    emb = np.load(ep)
    idx = pd.read_csv(ip)
    if emb.ndim != 2 or emb.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(f"{provider}: unexpected embedding shape {emb.shape}")
    if len(idx) != len(emb):
        raise RuntimeError(f"{provider}: index/vector row mismatch")
    if idx["job_id"].duplicated().any():
        raise RuntimeError(f"{provider}: duplicate job_ids")
    rows = idx["embedding_row"].to_numpy(dtype=int)
    if not np.array_equal(rows, np.arange(len(idx))):
        raise RuntimeError(f"{provider}: embedding_row mismatch")
    return emb.astype(np.float64, copy=False), idx


def inspect_design(provider: str, idx: pd.DataFrame):
    scenarios = sorted(idx["scenario_id"].unique())
    if len(scenarios) != EXPECTED_SCENARIOS:
        raise RuntimeError(f"{provider}: expected {EXPECTED_SCENARIOS} scenarios, found {len(scenarios)}")
    nominal_counts, pert_counts = [], []
    details = []
    for sid in scenarios:
        s = idx[idx["scenario_id"] == sid]
        n0 = len(s[s["condition"] == "nominal"])
        nominal_counts.append(n0)
        for family in PERTURBATION_FAMILIES:
            for lam in LAMBDA_VALUES:
                cell = s[(s["perturbation_family"] == family) & np.isclose(s["lambda"], lam)]
                count = len(cell)
                pert_counts.append(count)
                details.append((sid, family, lam, count))
    return {
        "provider": provider,
        "min_nominal": min(nominal_counts),
        "max_nominal": max(nominal_counts),
        "min_perturbed": min(pert_counts),
        "max_perturbed": max(pert_counts),
        "balanced_nominal": len(set(nominal_counts)) == 1,
        "balanced_perturbed": len(set(pert_counts)) == 1,
        "details": details,
    }


def choose_reference(designs):
    min_available = min(d["min_perturbed"] for d in designs)
    eligible = [n for n in PRIMARY_CHECKPOINTS if n <= min_available]
    if not eligible:
        raise RuntimeError(f"Only {min_available} perturbed repetitions are jointly available; need at least 10.")
    reference_n = max(eligible)
    previous = [n for n in PRIMARY_CHECKPOINTS if n < reference_n]
    previous_n = max(previous) if previous else None
    sample_sizes = [n for n in CHECKPOINTS if n <= reference_n]
    return reference_n, previous_n, sample_sizes, min_available


def estimate_bandwidth(nominal):
    d = pdist(nominal, metric="euclidean")
    d = d[d > 0]
    if len(d) == 0:
        raise RuntimeError("Cannot estimate positive RBF bandwidth")
    bw = float(np.median(d))
    if not np.isfinite(bw) or bw <= 0:
        raise RuntimeError("Invalid RBF bandwidth")
    return bw


def rbf_kernel(x, y, bw):
    x2 = np.sum(x*x, axis=1)[:, None]
    y2 = np.sum(y*y, axis=1)[None, :]
    d2 = np.maximum(x2 + y2 - 2.0*(x @ y.T), 0.0)
    return np.exp(-d2 / (2.0*bw*bw))


def mmd2_biased(x, y, bw):
    v = np.mean(rbf_kernel(x,x,bw)) + np.mean(rbf_kernel(y,y,bw)) - 2.0*np.mean(rbf_kernel(x,y,bw))
    return float(max(v, 0.0))


def sample_without_replacement(rng, data, n):
    return data[rng.choice(len(data), size=n, replace=False)]


def sample_disjoint_pair(rng, nominal, n):
    if 2*n > len(nominal):
        raise RuntimeError(f"Reference n={n} requires at least {2*n} nominal generations for disjoint calibration; only {len(nominal)} are available.")
    ii = rng.choice(len(nominal), size=2*n, replace=False)
    return nominal[ii[:n]], nominal[ii[n:]]


def calibrate(provider, sid, nominal, bw, sample_sizes):
    thresholds, records = {}, []
    for n in sample_sizes:
        rng = np.random.default_rng(calibration_seed(provider, sid, n))
        vals = np.empty(B_CALIBRATION)
        for b in range(B_CALIBRATION):
            x, y = sample_disjoint_pair(rng, nominal, n)
            vals[b] = mmd2_biased(x, y, bw)
        threshold = float(np.quantile(vals, 1.0-ALPHA))
        thresholds[n] = threshold
        records.append({
            "provider": provider, "scenario_id": sid, "sample_size": n,
            "bandwidth": bw, "threshold": threshold,
            "calibration_mean": float(np.mean(vals)),
            "calibration_median": float(np.median(vals)),
            "calibration_sd": float(np.std(vals, ddof=1)), "B": B_CALIBRATION,
        })
    return thresholds, records


def evaluate_condition(provider, sid, family, lam, nominal, perturbed, bw, thresholds, sample_sizes, reference_n):
    distributions = {}
    raw = []
    # Every n gets repeated nominal sampling. For n < reference, perturbed samples are also subsampled.
    for n in sample_sizes:
        rng = np.random.default_rng(condition_seed(provider, sid, family, lam, n))
        rhos = np.empty(B_PERTURBATION)
        mmds = np.empty(B_PERTURBATION)
        for b in range(B_PERTURBATION):
            nom = sample_without_replacement(rng, nominal, n)
            per = perturbed if n == reference_n else sample_without_replacement(rng, perturbed, n)
            value = mmd2_biased(nom, per, bw)
            rho = value / max(thresholds[n], EPS)
            mmds[b], rhos[b] = value, rho
            raw.append({"provider": provider, "scenario_id": sid, "perturbation_family": family,
                        "lambda": lam, "sample_size": n, "iteration": b, "mmd2": value,
                        "threshold": thresholds[n], "rho": rho, "exceeded": bool(rho > 1.0)})
        distributions[n] = (mmds, rhos)

    ref_rhos = distributions[reference_n][1]
    ref_median = float(np.median(ref_rhos))
    rows = []
    for n in sample_sizes:
        mmds, rhos = distributions[n]
        med = float(np.median(rhos))
        rel = 0.0 if n == reference_n else float(abs(med-ref_median)/max(abs(ref_median), EPS))
        # Iteration-wise agreement is descriptive; RNG streams differ by n.
        agree = 1.0 if n == reference_n else float(np.mean((rhos > 1.0) == (ref_rhos > 1.0)))
        rows.append({
            "provider": provider, "scenario_id": sid, "perturbation_family": family, "lambda": lam,
            "sample_size": n, "reference_n": reference_n,
            "median_rho": med, "mean_rho": float(np.mean(rhos)), "sd_rho": float(np.std(rhos, ddof=1)),
            "median_mmd2": float(np.median(mmds)), "exceedance_rate": float(np.mean(rhos > 1.0)),
            "relative_deviation_vs_reference": rel, "exceedance_agreement_vs_reference": agree,
        })
    return raw, rows


def calculate_rank_stability(condition_df, reference_n):
    records = []
    keys = ["provider", "scenario_id", "perturbation_family"]
    for key, g in condition_df.groupby(keys):
        ref = g[g["sample_size"] == reference_n].sort_values("lambda")["median_rho"].to_numpy()
        for n, gn in g.groupby("sample_size"):
            vals = gn.sort_values("lambda")["median_rho"].to_numpy()
            if n == reference_n:
                rho = 1.0
            else:
                stat = spearmanr(vals, ref).statistic
                rho = float(stat) if np.isfinite(stat) else np.nan
            records.append(dict(zip(keys, key)) | {"sample_size": int(n), "reference_n": reference_n,
                                                        "rank_stability_vs_reference": rho})
    return pd.DataFrame(records)


def aggregate(condition_df, rank_df):
    group = ["provider", "perturbation_family", "sample_size", "reference_n"]
    a = condition_df.groupby(group, as_index=False).agg(
        median_relative_deviation=("relative_deviation_vs_reference", "median"),
        q90_relative_deviation=("relative_deviation_vs_reference", lambda x: np.quantile(x, .90)),
        mean_exceedance_agreement=("exceedance_agreement_vs_reference", "mean"),
        median_exceedance_agreement=("exceedance_agreement_vs_reference", "median"),
        mean_exceedance_rate=("exceedance_rate", "mean"),
    )
    r = rank_df.groupby(group, as_index=False).agg(
        median_rank_stability=("rank_stability_vs_reference", "median"),
        mean_rank_stability=("rank_stability_vs_reference", "mean"),
    )
    return a.merge(r, on=group, how="left")


def global_summary(agg, reference_n, previous_n):
    rows = []
    for n in sorted(agg["sample_size"].unique()):
        s = agg[agg["sample_size"] == n]
        med_dev = float(np.median(s["median_relative_deviation"]))
        min_agree = float(np.min(s["mean_exceedance_agreement"]))
        med_rank = float(np.nanmedian(s["median_rank_stability"]))
        stable = (n == reference_n) or (med_dev <= MAX_MEDIAN_RELATIVE_DEVIATION and
                                        min_agree >= MIN_EXCEEDANCE_AGREEMENT and
                                        med_rank >= MIN_RANK_STABILITY)
        rows.append({"sample_size": int(n), "reference_n": reference_n,
                     "global_median_relative_deviation": med_dev,
                     "minimum_mean_exceedance_agreement": min_agree,
                     "global_median_rank_stability": med_rank,
                     "meets_stability_criteria": bool(stable),
                     "is_previous_checkpoint": bool(previous_n is not None and n == previous_n),
                     "is_reference_checkpoint": bool(n == reference_n)})
    return pd.DataFrame(rows)


def decision(global_df, reference_n, previous_n):
    if previous_n is None:
        return {"reference_n": reference_n, "previous_n": None,
                "decision": "BASELINE_ONLY", "recommended_n": reference_n,
                "message": "No earlier primary checkpoint is available for sequential comparison."}
    row = global_df[global_df["sample_size"] == previous_n].iloc[0]
    keep = bool(row["meets_stability_criteria"])
    return {
        "reference_n": reference_n, "previous_n": previous_n,
        "previous_median_relative_deviation": float(row["global_median_relative_deviation"]),
        "previous_minimum_mean_exceedance_agreement": float(row["minimum_mean_exceedance_agreement"]),
        "previous_global_median_rank_stability": float(row["global_median_rank_stability"]),
        "decision": "KEEP_PREVIOUS" if keep else "INCREASE_TO_REFERENCE",
        "recommended_n": previous_n if keep else reference_n,
        "message": (f"n={previous_n} is stable against n={reference_n}." if keep
                    else f"n={previous_n} is not stable against n={reference_n}; use n={reference_n} and, if needed, acquire the next checkpoint."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings-dir", default="embeddings")
    ap.add_argument("--output-dir", default="sample_size_analysis")
    ap.add_argument("--providers", nargs="+", default=PROVIDERS, choices=PROVIDERS)
    args = ap.parse_args()
    edir, odir = Path(args.embeddings_dir), Path(args.output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    loaded, designs = {}, []
    print("="*72); print("RQ4 SEQUENTIAL SAMPLE-SIZE STABILITY"); print("="*72)
    for p in args.providers:
        emb, idx = load_provider_data(edir, p)
        design = inspect_design(p, idx)
        loaded[p] = (emb, idx); designs.append(design)
        print(f"{p:8s} nominal={design['min_nominal']}..{design['max_nominal']} perturbed={design['min_perturbed']}..{design['max_perturbed']}")

    reference_n, previous_n, sample_sizes, min_available = choose_reference(designs)
    min_nominal = min(d["min_nominal"] for d in designs)
    if 2*reference_n > min_nominal:
        raise RuntimeError(f"Reference n={reference_n} requires at least {2*reference_n} nominal generations for disjoint calibration; minimum available across providers/scenarios is {min_nominal}. Expand nominal pools before validating this checkpoint.")

    print(f"\nDetected balanced reference checkpoint: n={reference_n}")
    print(f"Previous primary checkpoint:            n={previous_n}")
    print(f"Sample sizes evaluated:                 {sample_sizes}")
    if min_available > reference_n:
        print(f"NOTE: {min_available} repetitions are available, but {reference_n} is the largest predefined checkpoint <= that count.")

    all_cal, all_raw, all_cond = [], [], []
    for p in args.providers:
        emb, idx = loaded[p]
        print(f"\nProvider: {p}")
        for sid in sorted(idx["scenario_id"].unique()):
            print(f"  {sid}", flush=True)
            sr = idx[idx["scenario_id"] == sid]
            nr = sr[sr["condition"] == "nominal"].sort_values("repetition")
            nominal = emb[nr["embedding_row"].to_numpy(dtype=int)]
            bw = estimate_bandwidth(nominal)
            thresholds, recs = calibrate(p, sid, nominal, bw, sample_sizes)
            all_cal.extend(recs)
            for family in PERTURBATION_FAMILIES:
                for lam in LAMBDA_VALUES:
                    cr = sr[(sr["perturbation_family"] == family) & np.isclose(sr["lambda"], lam)].sort_values("repetition")
                    # Use exactly the first reference_n repetitions, making sequential checkpoints nested.
                    cr = cr.iloc[:reference_n]
                    pert = emb[cr["embedding_row"].to_numpy(dtype=int)]
                    raw, rows = evaluate_condition(p, sid, family, lam, nominal, pert, bw, thresholds, sample_sizes, reference_n)
                    all_raw.extend(raw); all_cond.extend(rows)

    cal_df = pd.DataFrame(all_cal)
    raw_df = pd.DataFrame(all_raw)
    cond_df = pd.DataFrame(all_cond)
    rank_df = calculate_rank_stability(cond_df, reference_n)
    agg_df = aggregate(cond_df, rank_df)
    global_df = global_summary(agg_df, reference_n, previous_n)
    dec = decision(global_df, reference_n, previous_n)

    cal_df.to_csv(odir/"rq4_sample_size_calibration.csv", index=False)
    cond_df.to_csv(odir/"rq4_sample_size_condition_summary.csv", index=False)
    rank_df.to_csv(odir/"rq4_sample_size_rank_stability.csv", index=False)
    agg_df.to_csv(odir/"rq4_sample_size_aggregate.csv", index=False)
    global_df.to_csv(odir/"rq4_sample_size_global.csv", index=False)
    raw_df.to_csv(odir/"rq4_sample_size_raw.csv", index=False)
    with (odir/"rq4_sample_size_decision.json").open("w") as f:
        json.dump(dec, f, indent=2)
    metadata = {
        "providers": args.providers, "sample_sizes": sample_sizes,
        "reference_n": reference_n, "previous_n": previous_n,
        "available_perturbed_min": min_available, "minimum_nominal_available": min_nominal,
        "alpha": ALPHA, "B_calibration": B_CALIBRATION, "B_perturbation": B_PERTURBATION,
        "discrepancy": "biased MMD^2", "kernel": "RBF",
        "bandwidth": "median pairwise Euclidean distance over nominal embeddings, separately by provider and scenario",
        "calibration": "two disjoint nominal samples of size n; procedure is not changed across checkpoints",
        "representation": "R_E", "embedding_model": "BAAI/bge-small-en-v1.5",
        "stability_criteria": {"max_median_relative_deviation": MAX_MEDIAN_RELATIVE_DEVIATION,
                               "min_exceedance_agreement": MIN_EXCEEDANCE_AGREEMENT,
                               "min_rank_stability": MIN_RANK_STABILITY},
    }
    with (odir/"rq4_sample_size_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*72); print("GLOBAL SAMPLE-SIZE SUMMARY"); print("="*72)
    print(global_df.to_string(index=False))
    print("\n" + "="*72); print("SEQUENTIAL DECISION"); print("="*72)
    print(json.dumps(dec, indent=2))
    print(f"\nOutputs: {odir}")


if __name__ == "__main__":
    main()
