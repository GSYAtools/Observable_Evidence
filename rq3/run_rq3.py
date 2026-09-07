#!/usr/bin/env python3

"""
RQ3 - Calibration under Controlled Distributional Perturbation

Research question
-----------------
How does empirical calibration distinguish nominal stochastic
variability from controlled deviations in observable representations?

Design
------
Nominal distribution:
    P0 = N(0,1)

Perturbation families:
    location:
        N(lambda, 1)

    scale:
        N(0, (1+lambda)^2)

    contamination:
        (1-lambda) N(0,1) + lambda N(3, 0.5^2)

Discrepancies:
    Jensen-Shannon divergence
    Total Variation
    Wasserstein-1
    biased MMD^2 with RBF kernel

Calibration:
    empirical nominal-vs-nominal discrepancy distribution
    95th empirical quantile
    independent held-out nominal validation

Primary sample size:
    n = 100

Sample-size analysis:
    n = 50, 100, 250

Macro-seeds:
    10

Important
---------
The experiment measures exceedance relative to calibrated nominal
variability. Exceedance is not interpreted as diagnosis, anomaly,
failure, or statistical significance.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist
from scipy.stats import (
    wasserstein_distance,
    spearmanr,
    beta,
)


# ============================================================
# Configuration
# ============================================================

BASE_SEED = 20260829

SEEDS = [
    BASE_SEED + i
    for i in range(10)
]

ALPHA = 0.05

PRIMARY_N = 100

SAMPLE_SIZES = [
    50,
    100,
    250,
]

LAMBDA_VALUES = np.array(
    [
        0.00,
        0.05,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.80,
        1.00,
    ],
    dtype=float,
)

PERTURBATIONS = [
    "location",
    "scale",
    "contamination",
]

MEASURES = [
    "JS",
    "TV",
    "W1",
    "MMD2",
]

# Primary seed:
B_PRIMARY = 2000
R_NOMINAL_PRIMARY = 1000
R_PERTURB_PRIMARY = 1000

# Remaining robustness seeds:
B_ROBUST = 1000
R_NOMINAL_ROBUST = 500
R_PERTURB_ROBUST = 500

# Calibration pool
N_CAL_POOL = 20000

# Histogram representation for JS / TV
N_BINS = 40
LOW_QUANTILE = 0.001
HIGH_QUANTILE = 0.999

# MMD bandwidth estimation
MMD_BANDWIDTH_SUBSAMPLE = 3000

# Numerical stability
EPS = 1e-12

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# Utility
# ============================================================

def get_run_sizes(seed_index: int) -> tuple[int, int, int]:
    """
    Primary seed receives the full run.
    Remaining seeds use the robustness configuration.
    """

    if seed_index == 0:
        return (
            B_PRIMARY,
            R_NOMINAL_PRIMARY,
            R_PERTURB_PRIMARY,
        )

    return (
        B_ROBUST,
        R_NOMINAL_ROBUST,
        R_PERTURB_ROBUST,
    )


# ============================================================
# Distribution generators
# ============================================================

def sample_nominal(
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:

    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=n,
    )


def sample_perturbed(
    rng: np.random.Generator,
    n: int,
    perturbation: str,
    lam: float,
) -> np.ndarray:

    if perturbation == "location":

        return rng.normal(
            loc=lam,
            scale=1.0,
            size=n,
        )

    if perturbation == "scale":

        return rng.normal(
            loc=0.0,
            scale=1.0 + lam,
            size=n,
        )

    if perturbation == "contamination":

        component = rng.random(n) < lam

        x = rng.normal(
            loc=0.0,
            scale=1.0,
            size=n,
        )

        n_contaminated = int(
            np.sum(component)
        )

        if n_contaminated > 0:

            x[component] = rng.normal(
                loc=3.0,
                scale=0.5,
                size=n_contaminated,
            )

        return x

    raise ValueError(
        f"Unknown perturbation: {perturbation}"
    )


# ============================================================
# Fixed histogram representation
# ============================================================

def build_histogram_edges(
    calibration_pool: np.ndarray,
) -> np.ndarray:
    """
    Histogram bins are fixed once from nominal calibration data.

    Two overflow bins are represented through -inf and +inf.
    """

    low = float(
        np.quantile(
            calibration_pool,
            LOW_QUANTILE,
        )
    )

    high = float(
        np.quantile(
            calibration_pool,
            HIGH_QUANTILE,
        )
    )

    internal_edges = np.linspace(
        low,
        high,
        N_BINS - 1,
    )

    edges = np.concatenate(
        [
            [-np.inf],
            internal_edges,
            [np.inf],
        ]
    )

    return edges


def empirical_histogram(
    x: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:

    counts, _ = np.histogram(
        x,
        bins=edges,
    )

    probabilities = (
        counts.astype(float)
        / len(x)
    )

    return probabilities


# ============================================================
# JS divergence
# ============================================================

def js_divergence(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
) -> float:

    p = empirical_histogram(
        x,
        edges,
    )

    q = empirical_histogram(
        y,
        edges,
    )

    m = 0.5 * (p + q)

    p_mask = p > 0
    q_mask = q > 0

    kl_pm = np.sum(
        p[p_mask]
        * np.log(
            p[p_mask]
            / m[p_mask]
        )
    )

    kl_qm = np.sum(
        q[q_mask]
        * np.log(
            q[q_mask]
            / m[q_mask]
        )
    )

    return float(
        0.5 * kl_pm
        + 0.5 * kl_qm
    )


# ============================================================
# Total Variation
# ============================================================

def total_variation(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
) -> float:

    p = empirical_histogram(
        x,
        edges,
    )

    q = empirical_histogram(
        y,
        edges,
    )

    return float(
        0.5
        * np.sum(
            np.abs(p - q)
        )
    )


# ============================================================
# MMD
# ============================================================

def estimate_mmd_bandwidth(
    rng: np.random.Generator,
    calibration_pool: np.ndarray,
) -> float:
    """
    Median heuristic estimated once from nominal calibration data.
    """

    n = min(
        MMD_BANDWIDTH_SUBSAMPLE,
        len(calibration_pool),
    )

    subset = rng.choice(
        calibration_pool,
        size=n,
        replace=False,
    )

    distances = pdist(
        subset.reshape(-1, 1),
        metric="euclidean",
    )

    positive = distances[
        distances > 0
    ]

    bandwidth = float(
        np.median(positive)
    )

    if (
        not np.isfinite(bandwidth)
        or bandwidth <= 0
    ):
        raise RuntimeError(
            "Invalid MMD bandwidth"
        )

    return bandwidth


def rbf_kernel_matrix(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
) -> np.ndarray:

    squared_distance = (
        x[:, None] - y[None, :]
    ) ** 2

    return np.exp(
        -squared_distance
        / (2.0 * bandwidth**2)
    )


def mmd2_biased(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
) -> float:

    k_xx = rbf_kernel_matrix(
        x,
        x,
        bandwidth,
    )

    k_yy = rbf_kernel_matrix(
        y,
        y,
        bandwidth,
    )

    k_xy = rbf_kernel_matrix(
        x,
        y,
        bandwidth,
    )

    value = (
        np.mean(k_xx)
        + np.mean(k_yy)
        - 2.0 * np.mean(k_xy)
    )

    # Numerical roundoff can produce tiny negatives.
    return float(
        max(value, 0.0)
    )


# ============================================================
# Unified discrepancy
# ============================================================

def discrepancy(
    measure: str,
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    mmd_bandwidth: float,
) -> float:

    if measure == "JS":

        return js_divergence(
            x,
            y,
            edges,
        )

    if measure == "TV":

        return total_variation(
            x,
            y,
            edges,
        )

    if measure == "W1":

        return float(
            wasserstein_distance(
                x,
                y,
            )
        )

    if measure == "MMD2":

        return mmd2_biased(
            x,
            y,
            mmd_bandwidth,
        )

    raise ValueError(
        f"Unknown measure: {measure}"
    )


# ============================================================
# Calibration
# ============================================================

def calibrate_measure(
    rng: np.random.Generator,
    measure: str,
    calibration_pool: np.ndarray,
    n: int,
    B: int,
    edges: np.ndarray,
    mmd_bandwidth: float,
) -> tuple[np.ndarray, float]:

    values = np.empty(
        B,
        dtype=float,
    )

    for b in range(B):

        # Independent resamples from the nominal pool.
        #
        # Sampling with replacement approximates repeated
        # draws from the nominal reference distribution.

        x = rng.choice(
            calibration_pool,
            size=n,
            replace=True,
        )

        y = rng.choice(
            calibration_pool,
            size=n,
            replace=True,
        )

        values[b] = discrepancy(
            measure,
            x,
            y,
            edges,
            mmd_bandwidth,
        )

    threshold = float(
        np.quantile(
            values,
            1.0 - ALPHA,
        )
    )

    return values, threshold


# ============================================================
# Binomial confidence interval
# ============================================================

def clopper_pearson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:

    alpha_ci = (
        1.0 - confidence
    )

    if successes == 0:
        lower = 0.0
    else:
        lower = float(
            beta.ppf(
                alpha_ci / 2.0,
                successes,
                trials - successes + 1,
            )
        )

    if successes == trials:
        upper = 1.0
    else:
        upper = float(
            beta.ppf(
                1.0 - alpha_ci / 2.0,
                successes + 1,
                trials - successes,
            )
        )

    return lower, upper


# ============================================================
# Held-out nominal validation
# ============================================================

def validate_nominal_calibration(
    rng: np.random.Generator,
    seed: int,
    measure: str,
    n: int,
    R: int,
    threshold: float,
    edges: np.ndarray,
    mmd_bandwidth: float,
) -> tuple[pd.DataFrame, dict]:

    records = []

    exceedances = 0

    for repetition in range(R):

        # Fresh draws, independent of calibration pool.
        x = sample_nominal(
            rng,
            n,
        )

        y = sample_nominal(
            rng,
            n,
        )

        value = discrepancy(
            measure,
            x,
            y,
            edges,
            mmd_bandwidth,
        )

        if threshold > 0:

            ratio = (
                value / threshold
            )

        else:

            ratio = np.nan

        exceeded = bool(
            value > threshold
        )

        exceedances += int(
            exceeded
        )

        records.append(
            {
                "seed": seed,
                "sample_size": n,
                "measure": measure,
                "repetition": repetition,
                "discrepancy": value,
                "threshold": threshold,
                "rho": ratio,
                "exceeded": exceeded,
            }
        )

    fer = (
        exceedances / R
    )

    ci_low, ci_high = (
        clopper_pearson_interval(
            exceedances,
            R,
        )
    )

    summary = {
        "seed": seed,
        "sample_size": n,
        "measure": measure,
        "n_trials": R,
        "exceedances": exceedances,
        "false_exceedance_rate": fer,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "target_alpha": ALPHA,
        "threshold": threshold,
    }

    return (
        pd.DataFrame(records),
        summary,
    )


# ============================================================
# Controlled perturbations
# ============================================================

def run_perturbation_experiment(
    rng: np.random.Generator,
    seed: int,
    measure: str,
    perturbation: str,
    lam: float,
    n: int,
    R: int,
    threshold: float,
    edges: np.ndarray,
    mmd_bandwidth: float,
) -> pd.DataFrame:

    records = []

    for repetition in range(R):

        x = sample_nominal(
            rng,
            n,
        )

        y = sample_perturbed(
            rng,
            n,
            perturbation,
            lam,
        )

        value = discrepancy(
            measure,
            x,
            y,
            edges,
            mmd_bandwidth,
        )

        if threshold > 0:

            ratio = (
                value / threshold
            )

        else:

            ratio = np.nan

        records.append(
            {
                "seed": seed,
                "sample_size": n,
                "measure": measure,
                "perturbation":
                    perturbation,
                "lambda": lam,
                "repetition":
                    repetition,
                "discrepancy":
                    value,
                "threshold":
                    threshold,
                "rho":
                    ratio,
                "exceeded":
                    bool(
                        value > threshold
                    ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Perturbation summaries
# ============================================================

def summarise_perturbations(
    df: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        df
        .groupby(
            [
                "seed",
                "sample_size",
                "measure",
                "perturbation",
                "lambda",
            ],
            as_index=False,
        )
        .agg(
            median_discrepancy=(
                "discrepancy",
                "median",
            ),
            mean_discrepancy=(
                "discrepancy",
                "mean",
            ),
            median_rho=(
                "rho",
                "median",
            ),
            mean_rho=(
                "rho",
                "mean",
            ),
            q025_rho=(
                "rho",
                lambda x:
                    np.quantile(
                        x,
                        0.025,
                    ),
            ),
            q975_rho=(
                "rho",
                lambda x:
                    np.quantile(
                        x,
                        0.975,
                    ),
            ),
            exceedance_rate=(
                "exceeded",
                "mean",
            ),
            n_repetitions=(
                "repetition",
                "count",
            ),
        )
    )

    return grouped


# ============================================================
# Spearman response
# ============================================================

def calculate_spearman(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    groups = summary_df.groupby(
        [
            "seed",
            "sample_size",
            "measure",
            "perturbation",
        ]
    )

    for keys, group in groups:

        (
            seed,
            sample_size,
            measure,
            perturbation,
        ) = keys

        group = group.sort_values(
            "lambda"
        )

        result = spearmanr(
            group["lambda"],
            group["median_rho"],
        )

        records.append(
            {
                "seed": seed,
                "sample_size":
                    sample_size,
                "measure":
                    measure,
                "perturbation":
                    perturbation,
                "spearman_rho":
                    float(
                        result.statistic
                    ),
                "spearman_p":
                    float(
                        result.pvalue
                    ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# Aggregate across macro-seeds
# ============================================================

def aggregate_nominal_across_seeds(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df
        .groupby(
            [
                "sample_size",
                "measure",
            ],
            as_index=False,
        )
        .agg(
            mean_threshold=(
                "threshold",
                "mean",
            ),
            sd_threshold=(
                "threshold",
                "std",
            ),
            mean_fer=(
                "false_exceedance_rate",
                "mean",
            ),
            sd_fer=(
                "false_exceedance_rate",
                "std",
            ),
            min_fer=(
                "false_exceedance_rate",
                "min",
            ),
            max_fer=(
                "false_exceedance_rate",
                "max",
            ),
        )
    )


def aggregate_perturbations_across_seeds(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        summary_df
        .groupby(
            [
                "sample_size",
                "measure",
                "perturbation",
                "lambda",
            ],
            as_index=False,
        )
        .agg(
            mean_median_rho=(
                "median_rho",
                "mean",
            ),
            sd_median_rho=(
                "median_rho",
                "std",
            ),
            mean_exceedance_rate=(
                "exceedance_rate",
                "mean",
            ),
            sd_exceedance_rate=(
                "exceedance_rate",
                "std",
            ),
            mean_median_discrepancy=(
                "median_discrepancy",
                "mean",
            ),
        )
    )


def aggregate_spearman(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df
        .groupby(
            [
                "sample_size",
                "measure",
                "perturbation",
            ],
            as_index=False,
        )
        .agg(
            mean_spearman_rho=(
                "spearman_rho",
                "mean",
            ),
            sd_spearman_rho=(
                "spearman_rho",
                "std",
            ),
            min_spearman_rho=(
                "spearman_rho",
                "min",
            ),
            max_spearman_rho=(
                "spearman_rho",
                "max",
            ),
        )
    )


# ============================================================
# Figures
# ============================================================

def plot_calibrated_response(
    aggregate_df: pd.DataFrame,
) -> None:

    primary = aggregate_df[
        aggregate_df["sample_size"]
        == PRIMARY_N
    ]

    for perturbation in PERTURBATIONS:

        fig, ax = plt.subplots(
            figsize=(8, 5.5)
        )

        subset = primary[
            primary["perturbation"]
            == perturbation
        ]

        for measure in MEASURES:

            data = (
                subset[
                    subset["measure"]
                    == measure
                ]
                .sort_values("lambda")
            )

            ax.plot(
                data["lambda"],
                data[
                    "mean_median_rho"
                ],
                marker="o",
                label=measure,
            )

        ax.axhline(
            1.0,
            linestyle="--",
            label="calibrated reference bound",
        )

        ax.set_xlabel(
            r"Perturbation intensity $\lambda$"
        )

        ax.set_ylabel(
            r"Mean across seeds of median "
            r"calibrated discrepancy $\rho_D$"
        )

        ax.set_title(
            f"RQ3 calibrated response: "
            f"{perturbation}"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR
            / (
                "rq3_calibrated_response_"
                f"{perturbation}.pdf"
            ),
            bbox_inches="tight",
        )

        fig.savefig(
            FIGURES_DIR
            / (
                "rq3_calibrated_response_"
                f"{perturbation}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_exceedance_rate(
    aggregate_df: pd.DataFrame,
) -> None:

    primary = aggregate_df[
        aggregate_df["sample_size"]
        == PRIMARY_N
    ]

    for perturbation in PERTURBATIONS:

        fig, ax = plt.subplots(
            figsize=(8, 5.5)
        )

        subset = primary[
            primary["perturbation"]
            == perturbation
        ]

        for measure in MEASURES:

            data = (
                subset[
                    subset["measure"]
                    == measure
                ]
                .sort_values("lambda")
            )

            ax.plot(
                data["lambda"],
                data[
                    "mean_exceedance_rate"
                ],
                marker="o",
                label=measure,
            )

        ax.axhline(
            ALPHA,
            linestyle="--",
            label=(
                f"nominal alpha={ALPHA:.2f}"
            ),
        )

        ax.set_xlabel(
            r"Perturbation intensity $\lambda$"
        )

        ax.set_ylabel(
            "Mean exceedance rate"
        )

        ax.set_ylim(
            -0.02,
            1.02,
        )

        ax.set_title(
            f"RQ3 exceedance response: "
            f"{perturbation}"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR
            / (
                "rq3_exceedance_rate_"
                f"{perturbation}.pdf"
            ),
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_sample_size_thresholds(
    nominal_aggregate: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    for measure in MEASURES:

        data = (
            nominal_aggregate[
                nominal_aggregate[
                    "measure"
                ]
                == measure
            ]
            .sort_values(
                "sample_size"
            )
        )

        ax.plot(
            data["sample_size"],
            data["mean_threshold"],
            marker="o",
            label=measure,
        )

    ax.set_xlabel(
        "Sample size"
    )

    ax.set_ylabel(
        "Mean calibrated reference bound"
    )

    ax.set_title(
        "RQ3 sample-size dependence "
        "of nominal calibration"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "rq3_sample_size_thresholds.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Metadata
# ============================================================

def save_metadata() -> None:

    metadata = {
        "experiment":
            "RQ3 calibration under controlled perturbation",

        "base_seed":
            BASE_SEED,

        "seeds":
            SEEDS,

        "n_macro_seeds":
            len(SEEDS),

        "nominal_distribution":
            "Normal(0,1)",

        "perturbations": {
            "location":
                "Normal(lambda,1)",
            "scale":
                "Normal(0,(1+lambda)^2)",
            "contamination":
                "(1-lambda)Normal(0,1) "
                "+ lambda Normal(3,0.5^2)",
        },

        "lambda_values":
            LAMBDA_VALUES.tolist(),

        "measures":
            MEASURES,

        "primary_sample_size":
            PRIMARY_N,

        "sample_sizes":
            SAMPLE_SIZES,

        "alpha":
            ALPHA,

        "calibration_quantile":
            1.0 - ALPHA,

        "calibration_pool_size":
            N_CAL_POOL,

        "histogram_bins":
            N_BINS,

        "histogram_range_quantiles": [
            LOW_QUANTILE,
            HIGH_QUANTILE,
        ],

        "mmd":
            "biased MMD^2 with RBF kernel",

        "mmd_bandwidth":
            "median heuristic estimated once "
            "per macro-seed from nominal "
            "calibration data",

        "primary_iterations": {
            "B_calibration":
                B_PRIMARY,
            "R_nominal":
                R_NOMINAL_PRIMARY,
            "R_perturbation":
                R_PERTURB_PRIMARY,
        },

        "robustness_iterations": {
            "B_calibration":
                B_ROBUST,
            "R_nominal":
                R_NOMINAL_ROBUST,
            "R_perturbation":
                R_PERTURB_ROBUST,
        },

        "interpretation":
            "rho > 1 indicates exceedance of "
            "the calibrated nominal reference "
            "bound; no diagnostic meaning is "
            "assigned",

        "python_version":
            platform.python_version(),

        "numpy_version":
            np.__version__,

        "scipy_version":
            scipy.__version__,

        "pandas_version":
            pd.__version__,
    }

    with open(
        RESULTS_DIR
        / "rq3_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 72)
    print(
        "RQ3 - CALIBRATION UNDER "
        "CONTROLLED PERTURBATION"
    )
    print("=" * 72)

    all_nominal_summaries = []
    all_nominal_raw = []

    all_perturbation_raw = []

    calibration_records = []

    # ========================================================
    # Macro-seeds
    # ========================================================

    for seed_index, seed in enumerate(
        SEEDS
    ):

        print()
        print("=" * 72)
        print(
            f"Macro-seed "
            f"{seed_index + 1}/{len(SEEDS)}: "
            f"{seed}"
        )
        print("=" * 72)

        rng = np.random.default_rng(
            seed
        )

        (
            B,
            R_nominal,
            R_perturb,
        ) = get_run_sizes(
            seed_index
        )

        # ----------------------------------------------------
        # Independent nominal calibration pool
        # ----------------------------------------------------

        calibration_pool = sample_nominal(
            rng,
            N_CAL_POOL,
        )

        histogram_edges = (
            build_histogram_edges(
                calibration_pool
            )
        )

        mmd_bandwidth = (
            estimate_mmd_bandwidth(
                rng,
                calibration_pool,
            )
        )

        print(
            f"MMD bandwidth: "
            f"{mmd_bandwidth:.6f}"
        )

        # ====================================================
        # Sample sizes
        # ====================================================

        for n in SAMPLE_SIZES:

            print()
            print(
                f"Sample size n={n}"
            )

            thresholds = {}

            # ------------------------------------------------
            # Calibration per measure
            # ------------------------------------------------

            for measure in MEASURES:

                print(
                    f"  Calibrating {measure}...",
                    end="",
                    flush=True,
                )

                (
                    calibration_values,
                    threshold,
                ) = calibrate_measure(
                    rng=rng,
                    measure=measure,
                    calibration_pool=
                        calibration_pool,
                    n=n,
                    B=B,
                    edges=histogram_edges,
                    mmd_bandwidth=
                        mmd_bandwidth,
                )

                thresholds[
                    measure
                ] = threshold

                calibration_records.append(
                    {
                        "seed": seed,
                        "sample_size": n,
                        "measure": measure,
                        "B": B,
                        "threshold":
                            threshold,
                        "calibration_mean":
                            float(
                                np.mean(
                                    calibration_values
                                )
                            ),
                        "calibration_median":
                            float(
                                np.median(
                                    calibration_values
                                )
                            ),
                        "calibration_sd":
                            float(
                                np.std(
                                    calibration_values,
                                    ddof=1,
                                )
                            ),
                        "mmd_bandwidth":
                            mmd_bandwidth,
                    }
                )

                print(
                    f" threshold="
                    f"{threshold:.6g}"
                )

                # --------------------------------------------
                # Held-out nominal validation
                # --------------------------------------------

                nominal_raw, nominal_summary = (
                    validate_nominal_calibration(
                        rng=rng,
                        seed=seed,
                        measure=measure,
                        n=n,
                        R=R_nominal,
                        threshold=threshold,
                        edges=histogram_edges,
                        mmd_bandwidth=
                            mmd_bandwidth,
                    )
                )

                all_nominal_raw.append(
                    nominal_raw
                )

                all_nominal_summaries.append(
                    nominal_summary
                )

            # ------------------------------------------------
            # Perturbations only for primary n=100
            #
            # Sample-size experiment concerns calibration
            # thresholds. Full perturbation factorial is kept
            # at the primary sample size to avoid unnecessary
            # duplication.
            # ------------------------------------------------

            if n != PRIMARY_N:
                continue

            for perturbation in PERTURBATIONS:

                print()
                print(
                    f"  Perturbation: "
                    f"{perturbation}"
                )

                for lam in LAMBDA_VALUES:

                    print(
                        f"    lambda={lam:.2f}",
                        end="",
                        flush=True,
                    )

                    for measure in MEASURES:

                        perturb_df = (
                            run_perturbation_experiment(
                                rng=rng,
                                seed=seed,
                                measure=measure,
                                perturbation=
                                    perturbation,
                                lam=float(lam),
                                n=n,
                                R=R_perturb,
                                threshold=
                                    thresholds[
                                        measure
                                    ],
                                edges=
                                    histogram_edges,
                                mmd_bandwidth=
                                    mmd_bandwidth,
                            )
                        )

                        all_perturbation_raw.append(
                            perturb_df
                        )

                    print(" done")

    # ========================================================
    # Combine raw outputs
    # ========================================================

    calibration_df = pd.DataFrame(
        calibration_records
    )

    nominal_summary_df = pd.DataFrame(
        all_nominal_summaries
    )

    nominal_raw_df = pd.concat(
        all_nominal_raw,
        ignore_index=True,
    )

    perturbation_raw_df = pd.concat(
        all_perturbation_raw,
        ignore_index=True,
    )

    # ========================================================
    # Summaries
    # ========================================================

    perturbation_summary_df = (
        summarise_perturbations(
            perturbation_raw_df
        )
    )

    spearman_df = (
        calculate_spearman(
            perturbation_summary_df
        )
    )

    nominal_aggregate_df = (
        aggregate_nominal_across_seeds(
            nominal_summary_df
        )
    )

    perturbation_aggregate_df = (
        aggregate_perturbations_across_seeds(
            perturbation_summary_df
        )
    )

    spearman_aggregate_df = (
        aggregate_spearman(
            spearman_df
        )
    )

    # ========================================================
    # Save results
    # ========================================================

    calibration_df.to_csv(
        RESULTS_DIR
        / "rq3_calibration_summary.csv",
        index=False,
    )

    nominal_summary_df.to_csv(
        RESULTS_DIR
        / "rq3_nominal_validation_by_seed.csv",
        index=False,
    )

    nominal_aggregate_df.to_csv(
        RESULTS_DIR
        / "rq3_nominal_validation_aggregate.csv",
        index=False,
    )

    perturbation_summary_df.to_csv(
        RESULTS_DIR
        / "rq3_perturbation_summary_by_seed.csv",
        index=False,
    )

    perturbation_aggregate_df.to_csv(
        RESULTS_DIR
        / "rq3_perturbation_aggregate.csv",
        index=False,
    )

    spearman_df.to_csv(
        RESULTS_DIR
        / "rq3_spearman_by_seed.csv",
        index=False,
    )

    spearman_aggregate_df.to_csv(
        RESULTS_DIR
        / "rq3_spearman_aggregate.csv",
        index=False,
    )

    # Raw outputs are useful for reproducibility but larger.
    nominal_raw_df.to_csv(
        RESULTS_DIR
        / "rq3_nominal_raw.csv",
        index=False,
    )

    perturbation_raw_df.to_csv(
        RESULTS_DIR
        / "rq3_perturbation_raw.csv",
        index=False,
    )

    # ========================================================
    # Figures
    # ========================================================

    plot_calibrated_response(
        perturbation_aggregate_df
    )

    plot_exceedance_rate(
        perturbation_aggregate_df
    )

    plot_sample_size_thresholds(
        nominal_aggregate_df
    )

    # ========================================================
    # Metadata
    # ========================================================

    save_metadata()

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 72)
    print("RQ3 COMPLETED")
    print("=" * 72)

    print()
    print(
        "Nominal validation "
        "(aggregate across seeds):"
    )

    print(
        nominal_aggregate_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Spearman response "
        "(aggregate across seeds):"
    )

    print(
        spearman_aggregate_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results written to: "
        f"{RESULTS_DIR}"
    )

    print(
        f"Figures written to: "
        f"{FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()
