#!/usr/bin/env python3

"""
RQ2 - Controlled Evaluation of Representational Indistinguishability

This experiment examines two consequences of Proposition 1:

A. Pairwise irreducible approximation bound.
B. Population error floor under increasing downstream capacity.

Three regimes are compared:

1. Preserved information
2. Observational indistinguishability
3. Representation-induced indistinguishability

The experiment illustrates observable consequences of Proposition 1.
It does not constitute an empirical proof of the proposition.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from numpy.polynomial.legendre import legvander


# ============================================================
# Configuration
# ============================================================

SEED = 20260829

N_TRAIN = 5000
N_TEST = 10000

MIN_DEGREE = 1
MAX_DEGREE = 16

PAIR_D_VALUES = [
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

N_PAIR_X = 500

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)


# ============================================================
# Regimes
# ============================================================

REGIMES = [
    "preserved",
    "observational_loss",
    "representational_loss",
]


def build_representation(
    x: np.ndarray,
    z: np.ndarray,
    regime: str,
) -> np.ndarray:
    """
    Construct the downstream representation.

    preserved:
        observable (x,z), representation (x,z)

    observational_loss:
        observable x only

    representational_loss:
        observable (x,z), representation x only
    """

    if regime == "preserved":
        return np.column_stack([x, z])

    if regime in {
        "observational_loss",
        "representational_loss",
    }:
        return x.reshape(-1, 1)

    raise ValueError(
        f"Unknown regime: {regime}"
    )


# ============================================================
# Legendre polynomial features
# ============================================================

def scale_to_legendre(
    x: np.ndarray,
) -> np.ndarray:
    """
    Inputs are already in [-1,1].
    """
    return x


def polynomial_features_1d(
    X: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Polynomial features for the one-dimensional
    indistinguishable regimes.
    """

    x = scale_to_legendre(
        X[:, 0]
    )

    return legvander(
        x,
        degree,
    )


def total_degree_indices_2d(
    degree: int,
) -> list[tuple[int, int]]:

    return [
        (i, j)
        for i in range(degree + 1)
        for j in range(degree + 1 - i)
    ]


def polynomial_features_2d(
    X: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Total-degree Legendre basis for preserved regime.
    """

    x = scale_to_legendre(
        X[:, 0]
    )

    z = scale_to_legendre(
        X[:, 1]
    )

    lx = legvander(
        x,
        degree,
    )

    lz = legvander(
        z,
        degree,
    )

    indices = total_degree_indices_2d(
        degree
    )

    columns = [
        lx[:, i] * lz[:, j]
        for i, j in indices
    ]

    return np.column_stack(columns)


def build_features(
    representation: np.ndarray,
    degree: int,
) -> np.ndarray:

    if representation.shape[1] == 1:
        return polynomial_features_1d(
            representation,
            degree,
        )

    if representation.shape[1] == 2:
        return polynomial_features_2d(
            representation,
            degree,
        )

    raise ValueError(
        "Unexpected representation dimension"
    )


# ============================================================
# Linear fitting
# ============================================================

def fit_regression(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:

    beta, _, _, _ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    return beta


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    error = y_true - y_pred

    abs_error = np.abs(error)

    return {
        "rmse": float(
            np.sqrt(
                np.mean(error**2)
            )
        ),
        "mae": float(
            np.mean(abs_error)
        ),
        "max_error": float(
            np.max(abs_error)
        ),
        "bias": float(
            np.mean(y_pred - y_true)
        ),
    }


# ============================================================
# RQ2-A Pairwise bound
# ============================================================

def run_pairwise_experiment() -> pd.DataFrame:

    records = []

    # Same x values are used for both members
    # of every indistinguishable pair.

    x_values = rng.uniform(
        -1.0,
        1.0,
        size=N_PAIR_X,
    )

    for d in PAIR_D_VALUES:

        z_a = np.full(
            N_PAIR_X,
            -d,
        )

        z_b = np.full(
            N_PAIR_X,
            +d,
        )

        theoretical_bound = d

        for regime in REGIMES:

            rep_a = build_representation(
                x_values,
                z_a,
                regime,
            )

            rep_b = build_representation(
                x_values,
                z_b,
                regime,
            )

            representation_distance = np.linalg.norm(
                rep_a - rep_b,
                axis=1,
            )

            indistinguishable = np.isclose(
                representation_distance,
                0.0,
                atol=1e-12,
            )

            for idx in range(N_PAIR_X):

                records.append(
                    {
                        "d": d,
                        "x": x_values[idx],
                        "regime": regime,
                        "phi_a": -d,
                        "phi_b": +d,
                        "target_difference":
                            2.0 * d,
                        "theoretical_bound":
                            theoretical_bound,
                        "representation_distance":
                            representation_distance[idx],
                        "indistinguishable":
                            bool(indistinguishable[idx]),
                    }
                )

    return pd.DataFrame(records)


# ============================================================
# Pairwise predictor test
# ============================================================

def run_pairwise_prediction_test() -> pd.DataFrame:
    """
    Train downstream approximators on population data and
    evaluate paired cases.

    In indistinguishable regimes, paired cases must receive
    identical predictions because their representations are
    identical.
    """

    train_x = rng.uniform(
        -1.0,
        1.0,
        size=N_TRAIN,
    )

    train_z = rng.uniform(
        -1.0,
        1.0,
        size=N_TRAIN,
    )

    y_train = train_z.copy()

    x_pairs = rng.uniform(
        -1.0,
        1.0,
        size=N_PAIR_X,
    )

    records = []

    for degree in range(
        MIN_DEGREE,
        MAX_DEGREE + 1,
    ):

        for regime in REGIMES:

            train_rep = build_representation(
                train_x,
                train_z,
                regime,
            )

            X_train = build_features(
                train_rep,
                degree,
            )

            beta = fit_regression(
                X_train,
                y_train,
            )

            for d in PAIR_D_VALUES:

                z_a = np.full(
                    N_PAIR_X,
                    -d,
                )

                z_b = np.full(
                    N_PAIR_X,
                    +d,
                )

                rep_a = build_representation(
                    x_pairs,
                    z_a,
                    regime,
                )

                rep_b = build_representation(
                    x_pairs,
                    z_b,
                    regime,
                )

                X_a = build_features(
                    rep_a,
                    degree,
                )

                X_b = build_features(
                    rep_b,
                    degree,
                )

                pred_a = X_a @ beta
                pred_b = X_b @ beta

                error_a = np.abs(
                    -d - pred_a
                )

                error_b = np.abs(
                    +d - pred_b
                )

                pair_max_error = np.maximum(
                    error_a,
                    error_b,
                )

                prediction_difference = np.abs(
                    pred_a - pred_b
                )

                representation_distance = np.linalg.norm(
                    rep_a - rep_b,
                    axis=1,
                )

                for idx in range(N_PAIR_X):

                    records.append(
                        {
                            "degree":
                                degree,
                            "regime":
                                regime,
                            "d":
                                d,
                            "theoretical_bound":
                                d,
                            "pair_max_error":
                                pair_max_error[idx],
                            "bound_satisfied":
                                bool(
                                    pair_max_error[idx]
                                    >= d - 1e-10
                                ),
                            "prediction_difference":
                                prediction_difference[idx],
                            "representation_distance":
                                representation_distance[idx],
                        }
                    )

    return pd.DataFrame(records)


# ============================================================
# RQ2-B Population experiment
# ============================================================

def run_population_experiment() -> pd.DataFrame:

    train_x = rng.uniform(
        -1.0,
        1.0,
        size=N_TRAIN,
    )

    train_z = rng.uniform(
        -1.0,
        1.0,
        size=N_TRAIN,
    )

    test_x = rng.uniform(
        -1.0,
        1.0,
        size=N_TEST,
    )

    test_z = rng.uniform(
        -1.0,
        1.0,
        size=N_TEST,
    )

    y_train = train_z.copy()
    y_test = test_z.copy()

    records = []

    for regime in REGIMES:

        print()
        print(f"Regime: {regime}")

        train_rep = build_representation(
            train_x,
            train_z,
            regime,
        )

        test_rep = build_representation(
            test_x,
            test_z,
            regime,
        )

        for degree in range(
            MIN_DEGREE,
            MAX_DEGREE + 1,
        ):

            print(
                f"  degree {degree:2d}",
                end="",
                flush=True,
            )

            X_train = build_features(
                train_rep,
                degree,
            )

            X_test = build_features(
                test_rep,
                degree,
            )

            beta = fit_regression(
                X_train,
                y_train,
            )

            train_pred = (
                X_train @ beta
            )

            test_pred = (
                X_test @ beta
            )

            train_metrics = calculate_metrics(
                y_train,
                train_pred,
            )

            test_metrics = calculate_metrics(
                y_test,
                test_pred,
            )

            condition_number = float(
                np.linalg.cond(X_train)
            )

            records.append(
                {
                    "regime":
                        regime,
                    "degree":
                        degree,
                    "n_features":
                        X_train.shape[1],
                    "train_rmse":
                        train_metrics["rmse"],
                    "train_mae":
                        train_metrics["mae"],
                    "train_max_error":
                        train_metrics["max_error"],
                    "test_rmse":
                        test_metrics["rmse"],
                    "test_mae":
                        test_metrics["mae"],
                    "test_max_error":
                        test_metrics["max_error"],
                    "test_bias":
                        test_metrics["bias"],
                    "condition_number":
                        condition_number,
                }
            )

            print(
                f"  test RMSE="
                f"{test_metrics['rmse']:.6f}"
            )

    return pd.DataFrame(records)


# ============================================================
# Figures
# ============================================================

def plot_population_rmse(
    df: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    for regime in REGIMES:

        subset = (
            df[
                df["regime"] == regime
            ]
            .sort_values("degree")
        )

        ax.plot(
            subset["degree"],
            subset["test_rmse"],
            marker="o",
            label=regime,
        )

    theoretical_floor = (
        1.0 / np.sqrt(3.0)
    )

    ax.axhline(
        theoretical_floor,
        linestyle="--",
        label=(
            r"$1/\sqrt{3}$ "
            "uninformative RMSE"
        ),
    )

    ax.set_xlabel(
        "Maximum polynomial degree"
    )

    ax.set_ylabel(
        "Held-out RMSE"
    )

    ax.set_title(
        "RQ2 downstream approximation under "
        "indistinguishability"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "rq2_population_rmse.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURES_DIR
        / "rq2_population_rmse.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_pairwise_bound(
    pair_df: pd.DataFrame,
) -> None:

    summary = (
        pair_df
        .groupby(
            [
                "regime",
                "d",
            ],
            as_index=False,
        )
        .agg(
            mean_pair_max_error=(
                "pair_max_error",
                "mean",
            ),
            min_pair_max_error=(
                "pair_max_error",
                "min",
            ),
            mean_prediction_difference=(
                "prediction_difference",
                "mean",
            ),
        )
    )

    # Use highest downstream degree for visualisation.
    # Rebuild summary from degree MAX_DEGREE only.

    degree_data = pair_df[
        pair_df["degree"]
        == MAX_DEGREE
    ]

    summary = (
        degree_data
        .groupby(
            [
                "regime",
                "d",
            ],
            as_index=False,
        )
        .agg(
            mean_pair_max_error=(
                "pair_max_error",
                "mean",
            ),
            min_pair_max_error=(
                "pair_max_error",
                "min",
            ),
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    for regime in REGIMES:

        subset = summary[
            summary["regime"] == regime
        ]

        ax.plot(
            subset["d"],
            subset["mean_pair_max_error"],
            marker="o",
            label=regime,
        )

    d_values = np.array(
        PAIR_D_VALUES
    )

    ax.plot(
        d_values,
        d_values,
        linestyle="--",
        label="Proposition 1 lower bound",
    )

    ax.set_xlabel(
        r"Pair separation $d$"
    )

    ax.set_ylabel(
        "Mean pairwise maximum error"
    )

    ax.set_title(
        "RQ2 pairwise approximation bound"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "rq2_pairwise_bound.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURES_DIR
        / "rq2_pairwise_bound.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Summaries
# ============================================================

def build_pairwise_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df
        .groupby(
            [
                "degree",
                "regime",
                "d",
            ],
            as_index=False,
        )
        .agg(
            mean_pair_max_error=(
                "pair_max_error",
                "mean",
            ),
            min_pair_max_error=(
                "pair_max_error",
                "min",
            ),
            max_pair_max_error=(
                "pair_max_error",
                "max",
            ),
            mean_prediction_difference=(
                "prediction_difference",
                "mean",
            ),
            bound_satisfaction_rate=(
                "bound_satisfied",
                "mean",
            ),
        )
    )


# ============================================================
# Metadata
# ============================================================

def save_metadata() -> None:

    metadata = {
        "experiment":
            "RQ2 controlled representational indistinguishability",

        "seed":
            SEED,

        "situation_space":
            "[-1,1]^2",

        "target":
            "phi(x,z)=z",

        "regimes": {
            "preserved":
                "omega(x,z)=(x,z), psi(x,z)=(x,z)",
            "observational_loss":
                "omega(x,z)=x",
            "representational_loss":
                "omega(x,z)=(x,z), psi(x,z)=x",
        },

        "population": {
            "n_train":
                N_TRAIN,
            "n_test":
                N_TEST,
            "distribution":
                "x,z independent Uniform(-1,1)",
            "degrees":
                [
                    MIN_DEGREE,
                    MAX_DEGREE,
                ],
            "theoretical_uninformative_rmse":
                float(
                    1.0 / np.sqrt(3.0)
                ),
        },

        "pairwise": {
            "d_values":
                PAIR_D_VALUES,
            "n_x_per_d":
                N_PAIR_X,
            "theoretical_bound":
                "|phi(sa)-phi(sb)|/2 = d",
        },

        "basis":
            "Legendre polynomial basis",

        "interpretation":
            "controlled empirical illustration "
            "of Proposition 1; not empirical proof",

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
        / "rq2_metadata.json",
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

    print("=" * 70)
    print(
        "RQ2 - CONTROLLED REPRESENTATIONAL "
        "INDISTINGUISHABILITY"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Structural pairwise check
    # --------------------------------------------------------

    structural_df = (
        run_pairwise_experiment()
    )

    structural_df.to_csv(
        RESULTS_DIR
        / "rq2_pairwise_structure.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Pairwise bound with trained approximators
    # --------------------------------------------------------

    pair_prediction_df = (
        run_pairwise_prediction_test()
    )

    pair_prediction_df.to_csv(
        RESULTS_DIR
        / "rq2_pairwise_predictions.csv",
        index=False,
    )

    pair_summary_df = (
        build_pairwise_summary(
            pair_prediction_df
        )
    )

    pair_summary_df.to_csv(
        RESULTS_DIR
        / "rq2_pairwise_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Population error floor
    # --------------------------------------------------------

    population_df = (
        run_population_experiment()
    )

    population_df.to_csv(
        RESULTS_DIR
        / "rq2_population_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    plot_population_rmse(
        population_df
    )

    plot_pairwise_bound(
        pair_prediction_df
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    save_metadata()

    print()
    print("=" * 70)
    print("RQ2 COMPLETED")
    print("=" * 70)

    print()
    print("Results:")
    print(
        RESULTS_DIR
        / "rq2_pairwise_structure.csv"
    )
    print(
        RESULTS_DIR
        / "rq2_pairwise_predictions.csv"
    )
    print(
        RESULTS_DIR
        / "rq2_pairwise_summary.csv"
    )
    print(
        RESULTS_DIR
        / "rq2_population_results.csv"
    )

    print()
    print("Figures:")
    print(
        FIGURES_DIR
        / "rq2_population_rmse.pdf"
    )
    print(
        FIGURES_DIR
        / "rq2_pairwise_bound.pdf"
    )


if __name__ == "__main__":
    main()
