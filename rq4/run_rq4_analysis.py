#!/usr/bin/env python3

"""
RQ4 - Definitive analysis over semantic representations R_E.

This script must be executed only after the sequential sample-size
validation has selected the final number of perturbed generations.

It reads:
    - semantic embeddings for OpenAI, Gemini and Qwen
    - embedding indexes
    - rq4_sample_size_decision.json

It computes:
    1. model/scenario-specific nominal calibration
    2. held-out nominal calibration behaviour
    3. calibrated MMD^2 response for every perturbation condition
    4. exceedance rates
    5. cross-scenario aggregation
    6. figures for semantic, context-removal and surface perturbations

No model generation or embedding computation occurs here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


# ============================================================
# Frozen configuration
# ============================================================

PROVIDERS = [
    "openai",
    "gemini",
    "qwen",
]

PROVIDER_LABELS = {
    "openai": "GPT-5.4 Mini",
    "gemini": "Gemini 2.5 Flash",
    "qwen": "Qwen3-4B",
}

PERTURBATION_FAMILIES = [
    "semantic",
    "context_removal",
    "surface",
]

LAMBDA_VALUES = [
    0.25,
    0.50,
    0.75,
    1.00,
]

ALPHA = 0.05

B_CALIBRATION = 5000
B_EVALUATION = 2000
B_NOMINAL_VALIDATION = 2000

BASE_SEED = 20260830

EXPECTED_DIMENSION = 384

EPS = 1e-12


# ============================================================
# Utilities
# ============================================================

def deterministic_seed(*parts) -> int:

    text = "|".join(
        str(part)
        for part in parts
    )

    value = sum(
        (i + 1) * ord(char)
        for i, char in enumerate(text)
    )

    return (
        BASE_SEED + value
    ) % (2**32 - 1)


# ============================================================
# Input
# ============================================================

def load_final_n(
    decision_path: Path,
) -> int:

    with decision_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        decision = json.load(f)

    for key in [
        "recommended_n",
        "selected_n",
        "final_n",
    ]:

        if key in decision:

            n = int(
                decision[key]
            )

            if n < 2:
                raise RuntimeError(
                    f"Invalid final sample size: {n}"
                )

            return n

    raise RuntimeError(
        "Could not find recommended_n, "
        "selected_n or final_n in "
        f"{decision_path}"
    )


def load_provider(
    embeddings_dir: Path,
    provider: str,
) -> tuple[np.ndarray, pd.DataFrame]:

    embedding_path = (
        embeddings_dir
        / f"rq4_{provider}_embeddings.npy"
    )

    index_path = (
        embeddings_dir
        / f"rq4_{provider}_embedding_index.csv"
    )

    embeddings = np.load(
        embedding_path
    )

    index_df = pd.read_csv(
        index_path
    )

    if embeddings.ndim != 2:

        raise RuntimeError(
            f"{provider}: invalid embedding matrix"
        )

    if (
        embeddings.shape[1]
        != EXPECTED_DIMENSION
    ):

        raise RuntimeError(
            f"{provider}: expected "
            f"{EXPECTED_DIMENSION} dimensions, "
            f"found {embeddings.shape[1]}"
        )

    if len(index_df) != len(embeddings):

        raise RuntimeError(
            f"{provider}: index/embedding mismatch"
        )

    if (
        index_df["job_id"]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{provider}: duplicate job_ids"
        )

    return (
        embeddings.astype(
            np.float64,
            copy=False,
        ),
        index_df,
    )


# ============================================================
# MMD
# ============================================================

def estimate_bandwidth(
    nominal: np.ndarray,
) -> float:

    distances = pdist(
        nominal,
        metric="euclidean",
    )

    distances = distances[
        distances > 0
    ]

    if len(distances) == 0:

        raise RuntimeError(
            "Cannot estimate RBF bandwidth."
        )

    bandwidth = float(
        np.median(
            distances
        )
    )

    if (
        not np.isfinite(bandwidth)
        or bandwidth <= 0
    ):

        raise RuntimeError(
            "Invalid RBF bandwidth."
        )

    return bandwidth


def rbf_kernel(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
) -> np.ndarray:

    x_norm = np.sum(
        x * x,
        axis=1,
    )[:, None]

    y_norm = np.sum(
        y * y,
        axis=1,
    )[None, :]

    squared_distance = (
        x_norm
        + y_norm
        - 2.0 * (
            x @ y.T
        )
    )

    squared_distance = np.maximum(
        squared_distance,
        0.0,
    )

    return np.exp(
        -squared_distance
        / (
            2.0
            * bandwidth**2
        )
    )


def mmd2_biased(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
) -> float:

    value = (
        np.mean(
            rbf_kernel(
                x,
                x,
                bandwidth,
            )
        )
        +
        np.mean(
            rbf_kernel(
                y,
                y,
                bandwidth,
            )
        )
        -
        2.0
        * np.mean(
            rbf_kernel(
                x,
                y,
                bandwidth,
            )
        )
    )

    return float(
        max(
            value,
            0.0,
        )
    )


# ============================================================
# Sampling
# ============================================================

def sample_indices(
    rng: np.random.Generator,
    population_size: int,
    n: int,
) -> np.ndarray:

    return rng.choice(
        population_size,
        size=n,
        replace=False,
    )


def sample_disjoint_pair(
    rng: np.random.Generator,
    population_size: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:

    if 2 * n > population_size:

        raise RuntimeError(
            f"Need {2*n} nominal observations "
            f"for disjoint calibration, "
            f"but only {population_size} exist."
        )

    indices = rng.choice(
        population_size,
        size=2 * n,
        replace=False,
    )

    return (
        indices[:n],
        indices[n:],
    )


# ============================================================
# Calibration
# ============================================================

def calibrate(
    nominal: np.ndarray,
    n: int,
    bandwidth: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:

    values = np.empty(
        B_CALIBRATION,
        dtype=float,
    )

    for b in range(
        B_CALIBRATION
    ):

        idx_a, idx_b = (
            sample_disjoint_pair(
                rng,
                len(nominal),
                n,
            )
        )

        values[b] = (
            mmd2_biased(
                nominal[idx_a],
                nominal[idx_b],
                bandwidth,
            )
        )

    threshold = float(
        np.quantile(
            values,
            1.0 - ALPHA,
        )
    )

    return (
        threshold,
        values,
    )


# ============================================================
# Nominal validation
# ============================================================

def nominal_validation(
    nominal: np.ndarray,
    n: int,
    bandwidth: float,
    threshold: float,
    rng: np.random.Generator,
) -> dict:

    values = np.empty(
        B_NOMINAL_VALIDATION,
        dtype=float,
    )

    for b in range(
        B_NOMINAL_VALIDATION
    ):

        idx_a, idx_b = (
            sample_disjoint_pair(
                rng,
                len(nominal),
                n,
            )
        )

        values[b] = (
            mmd2_biased(
                nominal[idx_a],
                nominal[idx_b],
                bandwidth,
            )
        )

    rhos = (
        values
        / max(
            threshold,
            EPS,
        )
    )

    return {
        "median_mmd2":
            float(
                np.median(
                    values
                )
            ),

        "median_rho":
            float(
                np.median(
                    rhos
                )
            ),

        "exceedance_rate":
            float(
                np.mean(
                    rhos > 1.0
                )
            ),
    }


# ============================================================
# Perturbation evaluation
# ============================================================

def evaluate_condition(
    nominal: np.ndarray,
    perturbed: np.ndarray,
    n: int,
    bandwidth: float,
    threshold: float,
    rng: np.random.Generator,
) -> dict:

    if len(perturbed) < n:

        raise RuntimeError(
            f"Perturbed cell contains "
            f"{len(perturbed)} observations, "
            f"but final n={n}."
        )

    # Freeze the first n repetitions as the final
    # experimental cell.
    perturbed = (
        perturbed[:n]
    )

    values = np.empty(
        B_EVALUATION,
        dtype=float,
    )

    for b in range(
        B_EVALUATION
    ):

        nominal_idx = (
            sample_indices(
                rng,
                len(nominal),
                n,
            )
        )

        values[b] = (
            mmd2_biased(
                nominal[
                    nominal_idx
                ],
                perturbed,
                bandwidth,
            )
        )

    rhos = (
        values
        / max(
            threshold,
            EPS,
        )
    )

    return {
        "median_mmd2":
            float(
                np.median(
                    values
                )
            ),

        "mean_mmd2":
            float(
                np.mean(
                    values
                )
            ),

        "q025_mmd2":
            float(
                np.quantile(
                    values,
                    0.025,
                )
            ),

        "q975_mmd2":
            float(
                np.quantile(
                    values,
                    0.975,
                )
            ),

        "median_rho":
            float(
                np.median(
                    rhos
                )
            ),

        "mean_rho":
            float(
                np.mean(
                    rhos
                )
            ),

        "q025_rho":
            float(
                np.quantile(
                    rhos,
                    0.025,
                )
            ),

        "q975_rho":
            float(
                np.quantile(
                    rhos,
                    0.975,
                )
            ),

        "exceedance_rate":
            float(
                np.mean(
                    rhos > 1.0
                )
            ),
    }


# ============================================================
# Cross-scenario aggregation
# ============================================================

def aggregate_conditions(
    condition_df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        condition_df
        .groupby(
            [
                "provider",
                "perturbation_family",
                "lambda",
            ],
            as_index=False,
        )
        .agg(
            median_rho=(
                "median_rho",
                "median",
            ),

            q25_rho=(
                "median_rho",
                lambda x:
                    np.quantile(
                        x,
                        0.25,
                    ),
            ),

            q75_rho=(
                "median_rho",
                lambda x:
                    np.quantile(
                        x,
                        0.75,
                    ),
            ),

            mean_rho=(
                "median_rho",
                "mean",
            ),

            mean_exceedance_rate=(
                "exceedance_rate",
                "mean",
            ),

            median_exceedance_rate=(
                "exceedance_rate",
                "median",
            ),

            n_scenarios=(
                "scenario_id",
                "nunique",
            ),
        )
    )


# ============================================================
# Provider summary
# ============================================================

def provider_summary(
    condition_df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        condition_df
        .groupby(
            [
                "provider",
                "perturbation_family",
            ],
            as_index=False,
        )
        .agg(
            median_calibrated_response=(
                "median_rho",
                "median",
            ),

            mean_calibrated_response=(
                "median_rho",
                "mean",
            ),

            mean_exceedance_rate=(
                "exceedance_rate",
                "mean",
            ),
        )
    )


# ============================================================
# Figures
# ============================================================

def plot_family(
    aggregate_df: pd.DataFrame,
    family: str,
    figures_dir: Path,
) -> None:

    subset = aggregate_df[
        aggregate_df[
            "perturbation_family"
        ]
        == family
    ]

    fig, ax = plt.subplots(
        figsize=(7.5, 5.2)
    )

    for provider in PROVIDERS:

        data = (
            subset[
                subset[
                    "provider"
                ]
                == provider
            ]
            .sort_values(
                "lambda"
            )
        )

        if data.empty:
            continue

        ax.plot(
            data["lambda"],
            data["median_rho"],
            marker="o",
            label=(
                PROVIDER_LABELS[
                    provider
                ]
            ),
        )

        ax.fill_between(
            data["lambda"],
            data["q25_rho"],
            data["q75_rho"],
            alpha=0.15,
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="calibrated reference bound",
    )

    ax.set_xlabel(
        r"Perturbation intensity $\lambda$"
    )

    ax.set_ylabel(
        r"Calibrated response $\rho$"
    )

    title = (
        family
        .replace(
            "_",
            " ",
        )
        .title()
    )

    ax.set_title(
        title
    )

    ax.legend()

    fig.tight_layout()

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        figures_dir
        / f"rq4_{family}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        figures_dir
        / f"rq4_{family}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings-dir",
        default="embeddings",
    )

    parser.add_argument(
        "--decision",
        default=(
            "sample_size_analysis/"
            "rq4_sample_size_decision.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="rq4_analysis",
    )

    parser.add_argument(
        "--figures-dir",
        default="figures",
    )

    args = parser.parse_args()

    embeddings_dir = Path(
        args.embeddings_dir
    )

    decision_path = Path(
        args.decision
    )

    output_dir = Path(
        args.output_dir
    )

    figures_dir = Path(
        args.figures_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_n = load_final_n(
        decision_path
    )

    print("=" * 72)
    print("RQ4 DEFINITIVE ANALYSIS")
    print("=" * 72)

    print(
        f"Validated sample size: n={final_n}"
    )

    calibration_records = []
    nominal_records = []
    condition_records = []

    for provider in PROVIDERS:

        print()
        print(
            f"Provider: {provider}"
        )

        embeddings, index_df = (
            load_provider(
                embeddings_dir,
                provider,
            )
        )

        scenarios = sorted(
            index_df[
                "scenario_id"
            ]
            .unique()
        )

        if len(scenarios) != 20:

            raise RuntimeError(
                f"{provider}: expected 20 scenarios, "
                f"found {len(scenarios)}"
            )

        for scenario_id in scenarios:

            print(
                f"  {scenario_id}",
                flush=True,
            )

            scenario_rows = index_df[
                index_df[
                    "scenario_id"
                ]
                == scenario_id
            ]

            nominal_rows = (
                scenario_rows[
                    scenario_rows[
                        "condition"
                    ]
                    == "nominal"
                ]
                .sort_values(
                    "repetition"
                )
            )

            nominal = embeddings[
                nominal_rows[
                    "embedding_row"
                ]
                .to_numpy(
                    dtype=int
                )
            ]

            if (
                len(nominal)
                < 2 * final_n
            ):

                raise RuntimeError(
                    f"{provider}/{scenario_id}: "
                    f"final n={final_n} requires "
                    f"at least {2*final_n} nominal "
                    f"outputs; found {len(nominal)}."
                )

            bandwidth = (
                estimate_bandwidth(
                    nominal
                )
            )

            calibration_rng = (
                np.random.default_rng(
                    deterministic_seed(
                        provider,
                        scenario_id,
                        "calibration",
                        final_n,
                    )
                )
            )

            (
                threshold,
                calibration_values,
            ) = calibrate(
                nominal=nominal,
                n=final_n,
                bandwidth=bandwidth,
                rng=calibration_rng,
            )

            calibration_records.append(
                {
                    "provider":
                        provider,

                    "scenario_id":
                        scenario_id,

                    "sample_size":
                        final_n,

                    "bandwidth":
                        bandwidth,

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
                }
            )

            nominal_rng = (
                np.random.default_rng(
                    deterministic_seed(
                        provider,
                        scenario_id,
                        "nominal-validation",
                        final_n,
                    )
                )
            )

            nominal_result = (
                nominal_validation(
                    nominal=nominal,
                    n=final_n,
                    bandwidth=bandwidth,
                    threshold=threshold,
                    rng=nominal_rng,
                )
            )

            nominal_records.append(
                {
                    "provider":
                        provider,

                    "scenario_id":
                        scenario_id,

                    "sample_size":
                        final_n,

                    "threshold":
                        threshold,

                    **nominal_result,
                }
            )

            for family in (
                PERTURBATION_FAMILIES
            ):

                for lam in (
                    LAMBDA_VALUES
                ):

                    cell_rows = (
                        scenario_rows[
                            (
                                scenario_rows[
                                    "perturbation_family"
                                ]
                                == family
                            )
                            &
                            (
                                np.isclose(
                                    scenario_rows[
                                        "lambda"
                                    ],
                                    lam,
                                )
                            )
                        ]
                        .sort_values(
                            "repetition"
                        )
                    )

                    if (
                        len(cell_rows)
                        < final_n
                    ):

                        raise RuntimeError(
                            f"{provider}/"
                            f"{scenario_id}/"
                            f"{family}/{lam}: "
                            f"only {len(cell_rows)} "
                            f"outputs for final n="
                            f"{final_n}."
                        )

                    cell_rows = (
                        cell_rows.iloc[
                            :final_n
                        ]
                    )

                    perturbed = embeddings[
                        cell_rows[
                            "embedding_row"
                        ]
                        .to_numpy(
                            dtype=int
                        )
                    ]

                    evaluation_rng = (
                        np.random.default_rng(
                            deterministic_seed(
                                provider,
                                scenario_id,
                                family,
                                lam,
                                final_n,
                            )
                        )
                    )

                    result = (
                        evaluate_condition(
                            nominal=nominal,
                            perturbed=perturbed,
                            n=final_n,
                            bandwidth=bandwidth,
                            threshold=threshold,
                            rng=evaluation_rng,
                        )
                    )

                    task_family = (
                        scenario_rows[
                            "task_family"
                        ]
                        .iloc[0]
                    )

                    condition_records.append(
                        {
                            "provider":
                                provider,

                            "scenario_id":
                                scenario_id,

                            "task_family":
                                task_family,

                            "perturbation_family":
                                family,

                            "lambda":
                                lam,

                            "sample_size":
                                final_n,

                            "threshold":
                                threshold,

                            "bandwidth":
                                bandwidth,

                            **result,
                        }
                    )

    # ========================================================
    # Tables
    # ========================================================

    calibration_df = pd.DataFrame(
        calibration_records
    )

    nominal_df = pd.DataFrame(
        nominal_records
    )

    condition_df = pd.DataFrame(
        condition_records
    )

    aggregate_df = (
        aggregate_conditions(
            condition_df
        )
    )

    model_summary_df = (
        provider_summary(
            condition_df
        )
    )

    calibration_df.to_csv(
        output_dir
        / "rq4_calibration.csv",
        index=False,
    )

    nominal_df.to_csv(
        output_dir
        / "rq4_nominal_validation.csv",
        index=False,
    )

    condition_df.to_csv(
        output_dir
        / "rq4_condition_results.csv",
        index=False,
    )

    aggregate_df.to_csv(
        output_dir
        / "rq4_aggregate.csv",
        index=False,
    )

    model_summary_df.to_csv(
        output_dir
        / "rq4_model_summary.csv",
        index=False,
    )

    # ========================================================
    # Figures
    # ========================================================

    for family in (
        PERTURBATION_FAMILIES
    ):

        plot_family(
            aggregate_df,
            family,
            figures_dir,
        )

    # ========================================================
    # Metadata
    # ========================================================

    metadata = {
        "rq":
            "RQ4",

        "representation":
            "R_E",

        "embedding_model":
            "BAAI/bge-small-en-v1.5",

        "embedding_dimension":
            EXPECTED_DIMENSION,

        "final_sample_size":
            final_n,

        "alpha":
            ALPHA,

        "B_calibration":
            B_CALIBRATION,

        "B_evaluation":
            B_EVALUATION,

        "B_nominal_validation":
            B_NOMINAL_VALIDATION,

        "discrepancy":
            "biased MMD^2",

        "kernel":
            "RBF",

        "bandwidth":
            (
                "median pairwise Euclidean "
                "distance over nominal "
                "embeddings, estimated "
                "separately by provider "
                "and scenario"
            ),

        "providers":
            PROVIDERS,

        "perturbation_families":
            PERTURBATION_FAMILIES,

        "lambda_values":
            LAMBDA_VALUES,
    }

    with (
        output_dir
        / "rq4_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 72)
    print("RQ4 ANALYSIS COMPLETED")
    print("=" * 72)

    nominal_summary = (
        nominal_df
        .groupby(
            "provider"
        )
        .agg(
            mean_nominal_exceedance=(
                "exceedance_rate",
                "mean",
            ),

            median_nominal_exceedance=(
                "exceedance_rate",
                "median",
            ),
        )
    )

    print()
    print(
        "Nominal calibration:"
    )

    print(
        nominal_summary.to_string()
    )

    print()
    print(
        "Perturbation summary:"
    )

    print(
        model_summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results: {output_dir}"
    )

    print(
        f"Figures: {figures_dir}"
    )


if __name__ == "__main__":
    main()
