#!/usr/bin/env python3

"""
RQ1 - Final Controlled Evaluation of Expressive Approximation

Primary objective
-----------------
Examine how empirical maximum approximation error evolves as the
expressive capacity of a polynomial algebra increases.

This experiment is a numerical illustration of Theorem 1.
It is not an empirical proof of the theorem.

Primary fitting method
----------------------
Grid-based L-infinity-oriented approximation.

Control method
--------------
L2 least-squares approximation.

Final target functionals
------------------------
1. Smooth periodic structure
2. Non-separable interaction
3. Sharp continuous transition

Robustness
----------
The complete experiment is repeated using a denser training and
evaluation discretisation.
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
from scipy.optimize import linprog


# ============================================================
# Configuration
# ============================================================

SEED = 20260829

MIN_DEGREE = 1
MAX_DEGREE = 16

# Primary experiment
PRIMARY_TRAIN_GRID = 31
PRIMARY_EVAL_GRID = 201

# Robustness experiment
ROBUST_TRAIN_GRID = 41
ROBUST_EVAL_GRID = 251

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# Final target functionals
# ============================================================

def phi_periodic(
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Smooth periodic target.
    """
    return np.sin(np.pi * u) * np.cos(np.pi * v)


def phi_interaction(
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Non-separable interaction target.
    """
    return np.sin(3.0 * np.pi * u * v)


def phi_transition(
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Sharp but continuous transition.
    """
    return np.tanh(8.0 * (u + v - 1.0))


TARGETS = {
    "periodic": phi_periodic,
    "interaction": phi_interaction,
    "transition": phi_transition,
}


# ============================================================
# Grid construction
# ============================================================

def make_training_grid(
    n: int,
) -> tuple[np.ndarray, np.ndarray]:

    axis = np.linspace(
        0.0,
        1.0,
        n,
    )

    uu, vv = np.meshgrid(
        axis,
        axis,
        indexing="xy",
    )

    return uu.ravel(), vv.ravel()


def make_evaluation_grid(
    n: int,
) -> tuple[np.ndarray, np.ndarray]:

    """
    Construct a dense evaluation grid shifted relative to
    the training grid.

    The evaluation locations therefore do not systematically
    coincide with the fitting locations.
    """

    step = 1.0 / n

    axis = np.linspace(
        step / 2.0,
        1.0 - step / 2.0,
        n,
    )

    uu, vv = np.meshgrid(
        axis,
        axis,
        indexing="xy",
    )

    return uu.ravel(), vv.ravel()


# ============================================================
# Legendre polynomial basis
# ============================================================

def scale_to_legendre_domain(
    x: np.ndarray,
) -> np.ndarray:

    """
    Map [0,1] to [-1,1].
    """

    return 2.0 * x - 1.0


def total_degree_indices(
    degree: int,
) -> list[tuple[int, int]]:

    """
    Indices (i,j) satisfying i+j <= degree.
    """

    return [
        (i, j)
        for i in range(degree + 1)
        for j in range(degree + 1 - i)
    ]


def design_matrix(
    u: np.ndarray,
    v: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:

    us = scale_to_legendre_domain(u)
    vs = scale_to_legendre_domain(v)

    lu = legvander(
        us,
        degree,
    )

    lv = legvander(
        vs,
        degree,
    )

    indices = total_degree_indices(degree)

    columns = [
        lu[:, i] * lv[:, j]
        for i, j in indices
    ]

    X = np.column_stack(columns)

    return X, indices


# ============================================================
# L2 fitting
# ============================================================

def fit_l2(
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
# L-infinity-oriented fitting
# ============================================================

def fit_linf(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:

    """
    Solve the discrete minimax problem

        min t

    subject to

        |X beta - y| <= t

    on the training grid.

    Returns
    -------
    beta
        Polynomial coefficients.

    t
        Optimal maximum error on the training grid.
    """

    n_samples, n_features = X.shape

    # Decision variables:
    #
    # beta_1, ..., beta_p, t

    c = np.zeros(n_features + 1)
    c[-1] = 1.0

    # X beta - y <= t
    #
    # X beta - t <= y

    A1 = np.column_stack(
        [
            X,
            -np.ones(n_samples),
        ]
    )

    b1 = y

    # y - X beta <= t
    #
    # -X beta - t <= -y

    A2 = np.column_stack(
        [
            -X,
            -np.ones(n_samples),
        ]
    )

    b2 = -y

    A_ub = np.vstack(
        [
            A1,
            A2,
        ]
    )

    b_ub = np.concatenate(
        [
            b1,
            b2,
        ]
    )

    bounds = [
        (None, None)
        for _ in range(n_features)
    ]

    # t >= 0
    bounds.append(
        (0.0, None)
    )

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            "L-infinity optimisation failed: "
            + result.message
        )

    beta = result.x[:-1]
    optimum = float(result.x[-1])

    return beta, optimum


# ============================================================
# Evaluation metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    error = y_true - y_pred

    absolute_error = np.abs(error)

    return {
        "rmse": float(
            np.sqrt(
                np.mean(error**2)
            )
        ),
        "mae": float(
            np.mean(absolute_error)
        ),
        "max_error": float(
            np.max(absolute_error)
        ),
        "p95_error": float(
            np.quantile(
                absolute_error,
                0.95,
            )
        ),
        "p99_error": float(
            np.quantile(
                absolute_error,
                0.99,
            )
        ),
    }


# ============================================================
# One experimental configuration
# ============================================================

def run_configuration(
    experiment_name: str,
    train_grid_size: int,
    eval_grid_size: int,
) -> pd.DataFrame:

    train_u, train_v = make_training_grid(
        train_grid_size
    )

    eval_u, eval_v = make_evaluation_grid(
        eval_grid_size
    )

    records: list[dict] = []

    print()
    print("=" * 70)
    print(f"Configuration: {experiment_name}")
    print(
        f"Training grid: "
        f"{train_grid_size} x {train_grid_size}"
    )
    print(
        f"Evaluation grid: "
        f"{eval_grid_size} x {eval_grid_size}"
    )
    print("=" * 70)

    for target_name, target_function in TARGETS.items():

        print()
        print(f"Target: {target_name}")

        y_train = target_function(
            train_u,
            train_v,
        )

        y_eval = target_function(
            eval_u,
            eval_v,
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

            X_train, indices = design_matrix(
                train_u,
                train_v,
                degree,
            )

            X_eval, _ = design_matrix(
                eval_u,
                eval_v,
                degree,
            )

            n_terms = len(indices)

            condition_number = float(
                np.linalg.cond(X_train)
            )

            # =================================================
            # L2 control
            # =================================================

            beta_l2 = fit_l2(
                X_train,
                y_train,
            )

            pred_train_l2 = (
                X_train @ beta_l2
            )

            pred_eval_l2 = (
                X_eval @ beta_l2
            )

            train_l2 = calculate_metrics(
                y_train,
                pred_train_l2,
            )

            eval_l2 = calculate_metrics(
                y_eval,
                pred_eval_l2,
            )

            records.append(
                {
                    "experiment":
                        experiment_name,
                    "target":
                        target_name,
                    "degree":
                        degree,
                    "n_terms":
                        n_terms,
                    "method":
                        "L2",
                    "train_grid_size":
                        train_grid_size,
                    "eval_grid_size":
                        eval_grid_size,
                    "train_rmse":
                        train_l2["rmse"],
                    "train_mae":
                        train_l2["mae"],
                    "train_p95_error":
                        train_l2["p95_error"],
                    "train_p99_error":
                        train_l2["p99_error"],
                    "train_max_error":
                        train_l2["max_error"],
                    "eval_rmse":
                        eval_l2["rmse"],
                    "eval_mae":
                        eval_l2["mae"],
                    "eval_p95_error":
                        eval_l2["p95_error"],
                    "eval_p99_error":
                        eval_l2["p99_error"],
                    "eval_max_error":
                        eval_l2["max_error"],
                    "condition_number":
                        condition_number,
                    "linf_training_optimum":
                        np.nan,
                }
            )

            # =================================================
            # L-infinity primary method
            # =================================================

            beta_linf, linf_optimum = fit_linf(
                X_train,
                y_train,
            )

            pred_train_linf = (
                X_train @ beta_linf
            )

            pred_eval_linf = (
                X_eval @ beta_linf
            )

            train_linf = calculate_metrics(
                y_train,
                pred_train_linf,
            )

            eval_linf = calculate_metrics(
                y_eval,
                pred_eval_linf,
            )

            records.append(
                {
                    "experiment":
                        experiment_name,
                    "target":
                        target_name,
                    "degree":
                        degree,
                    "n_terms":
                        n_terms,
                    "method":
                        "Linf",
                    "train_grid_size":
                        train_grid_size,
                    "eval_grid_size":
                        eval_grid_size,
                    "train_rmse":
                        train_linf["rmse"],
                    "train_mae":
                        train_linf["mae"],
                    "train_p95_error":
                        train_linf["p95_error"],
                    "train_p99_error":
                        train_linf["p99_error"],
                    "train_max_error":
                        train_linf["max_error"],
                    "eval_rmse":
                        eval_linf["rmse"],
                    "eval_mae":
                        eval_linf["mae"],
                    "eval_p95_error":
                        eval_linf["p95_error"],
                    "eval_p99_error":
                        eval_linf["p99_error"],
                    "eval_max_error":
                        eval_linf["max_error"],
                    "condition_number":
                        condition_number,
                    "linf_training_optimum":
                        linf_optimum,
                }
            )

            print(
                "  "
                f"Linf eval max="
                f"{eval_linf['max_error']:.6e}"
            )

    return pd.DataFrame(records)


# ============================================================
# Robustness comparison
# ============================================================

def build_robustness_table(
    primary_df: pd.DataFrame,
    robust_df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Compare the primary L-infinity results with the denser-grid
    robustness execution.
    """

    primary = (
        primary_df[
            primary_df["method"] == "Linf"
        ][
            [
                "target",
                "degree",
                "eval_max_error",
            ]
        ]
        .rename(
            columns={
                "eval_max_error":
                    "primary_eval_max_error"
            }
        )
    )

    robust = (
        robust_df[
            robust_df["method"] == "Linf"
        ][
            [
                "target",
                "degree",
                "eval_max_error",
            ]
        ]
        .rename(
            columns={
                "eval_max_error":
                    "robust_eval_max_error"
            }
        )
    )

    comparison = primary.merge(
        robust,
        on=[
            "target",
            "degree",
        ],
        how="inner",
    )

    comparison[
        "absolute_difference"
    ] = np.abs(
        comparison[
            "primary_eval_max_error"
        ]
        -
        comparison[
            "robust_eval_max_error"
        ]
    )

    denominator = np.maximum(
        comparison[
            "primary_eval_max_error"
        ],
        np.finfo(float).eps,
    )

    comparison[
        "relative_difference"
    ] = (
        comparison[
            "absolute_difference"
        ]
        / denominator
    )

    return comparison


# ============================================================
# Main RQ1 figure
# ============================================================

def plot_primary_max_error(
    primary_df: pd.DataFrame,
) -> None:

    data = primary_df[
        primary_df["method"] == "Linf"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    for target in TARGETS:

        target_data = (
            data[
                data["target"] == target
            ]
            .sort_values("degree")
        )

        ax.plot(
            target_data["degree"],
            target_data["eval_max_error"],
            marker="o",
            label=target,
        )

    ax.set_xlabel(
        "Maximum polynomial degree"
    )

    ax.set_ylabel(
        "Empirical maximum approximation error"
    )

    ax.set_yscale("log")

    ax.set_title(
        "RQ1 expressive approximation"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "rq1_final_max_error.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURES_DIR
        / "rq1_final_max_error.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# L2 vs L-infinity control figure
# ============================================================

def plot_method_comparison(
    primary_df: pd.DataFrame,
) -> None:

    for target in TARGETS:

        subset = (
            primary_df[
                primary_df["target"] == target
            ]
            .sort_values("degree")
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        for method in [
            "L2",
            "Linf",
        ]:

            data = subset[
                subset["method"] == method
            ]

            ax.plot(
                data["degree"],
                data["eval_max_error"],
                marker="o",
                label=method,
            )

        ax.set_xlabel(
            "Maximum polynomial degree"
        )

        ax.set_ylabel(
            "Empirical maximum approximation error"
        )

        ax.set_yscale("log")

        ax.set_title(
            f"RQ1 fitting criterion: {target}"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR
            / f"rq1_method_comparison_{target}.pdf",
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# Grid robustness figure
# ============================================================

def plot_grid_robustness(
    primary_df: pd.DataFrame,
    robust_df: pd.DataFrame,
) -> None:

    primary = primary_df[
        primary_df["method"] == "Linf"
    ]

    robust = robust_df[
        robust_df["method"] == "Linf"
    ]

    for target in TARGETS:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        p = (
            primary[
                primary["target"] == target
            ]
            .sort_values("degree")
        )

        r = (
            robust[
                robust["target"] == target
            ]
            .sort_values("degree")
        )

        ax.plot(
            p["degree"],
            p["eval_max_error"],
            marker="o",
            label=(
                f"{PRIMARY_TRAIN_GRID}x"
                f"{PRIMARY_TRAIN_GRID} / "
                f"{PRIMARY_EVAL_GRID}x"
                f"{PRIMARY_EVAL_GRID}"
            ),
        )

        ax.plot(
            r["degree"],
            r["eval_max_error"],
            marker="s",
            label=(
                f"{ROBUST_TRAIN_GRID}x"
                f"{ROBUST_TRAIN_GRID} / "
                f"{ROBUST_EVAL_GRID}x"
                f"{ROBUST_EVAL_GRID}"
            ),
        )

        ax.set_xlabel(
            "Maximum polynomial degree"
        )

        ax.set_ylabel(
            "Empirical maximum approximation error"
        )

        ax.set_yscale("log")

        ax.set_title(
            f"RQ1 grid robustness: {target}"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR
            / f"rq1_grid_robustness_{target}.pdf",
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# Metadata
# ============================================================

def save_metadata() -> None:

    metadata = {
        "experiment":
            "RQ1 final controlled expressive approximation",

        "date_design_frozen":
            "2026-08-29",

        "seed":
            SEED,

        "domain":
            "[0,1]^2",

        "base_functionals": {
            "T1": "u",
            "T2": "v",
        },

        "separation":
            "T1 and T2 separate points on K",

        "min_degree":
            MIN_DEGREE,

        "max_degree":
            MAX_DEGREE,

        "primary_configuration": {
            "training_grid_per_axis":
                PRIMARY_TRAIN_GRID,
            "evaluation_grid_per_axis":
                PRIMARY_EVAL_GRID,
        },

        "robustness_configuration": {
            "training_grid_per_axis":
                ROBUST_TRAIN_GRID,
            "evaluation_grid_per_axis":
                ROBUST_EVAL_GRID,
        },

        "basis":
            "total-degree tensor-product Legendre basis",

        "primary_fit":
            "grid-based L-infinity linear programming",

        "control_fit":
            "L2 least squares",

        "targets": {
            "periodic":
                "sin(pi*u)*cos(pi*v)",
            "interaction":
                "sin(3*pi*u*v)",
            "transition":
                "tanh(8*(u+v-1))",
        },

        "primary_outcome":
            "empirical maximum approximation error "
            "on independent evaluation grid",

        "interpretation":
            "numerical illustration of expressive "
            "approximation; not empirical proof of "
            "Theorem 1",

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
        / "rq1_final_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


# ============================================================
# Entry point
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        "RQ1 - FINAL CONTROLLED "
        "EXPRESSIVE APPROXIMATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Primary experiment
    # --------------------------------------------------------

    primary_df = run_configuration(
        experiment_name="primary",
        train_grid_size=PRIMARY_TRAIN_GRID,
        eval_grid_size=PRIMARY_EVAL_GRID,
    )

    primary_df.to_csv(
        RESULTS_DIR
        / "rq1_final_primary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Robustness experiment
    # --------------------------------------------------------

    robust_df = run_configuration(
        experiment_name="grid_robustness",
        train_grid_size=ROBUST_TRAIN_GRID,
        eval_grid_size=ROBUST_EVAL_GRID,
    )

    robust_df.to_csv(
        RESULTS_DIR
        / "rq1_final_grid_robustness.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Robustness comparison
    # --------------------------------------------------------

    comparison_df = build_robustness_table(
        primary_df,
        robust_df,
    )

    comparison_df.to_csv(
        RESULTS_DIR
        / "rq1_final_robustness_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    save_metadata()

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    plot_primary_max_error(
        primary_df
    )

    plot_method_comparison(
        primary_df
    )

    plot_grid_robustness(
        primary_df,
        robust_df,
    )

    print()
    print("=" * 70)
    print("RQ1 FINAL EXPERIMENT COMPLETED")
    print("=" * 70)

    print()
    print("Primary results:")
    print(
        RESULTS_DIR
        / "rq1_final_primary.csv"
    )

    print()
    print("Grid robustness:")
    print(
        RESULTS_DIR
        / "rq1_final_grid_robustness.csv"
    )

    print()
    print("Robustness comparison:")
    print(
        RESULTS_DIR
        / "rq1_final_robustness_comparison.csv"
    )

    print()
    print("Figures:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
