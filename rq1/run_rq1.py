#!/usr/bin/env python3

"""
RQ1 - Controlled Evaluation of Expressive Approximation

Purpose
-------
Numerically illustrate the approximation behaviour associated with
Theorem 1 under a controlled compact domain and a separating family.

The experiment compares increasingly expressive polynomial spaces
using two fitting criteria:

1. L2 least-squares approximation.
2. Grid-based L-infinity-oriented approximation using linear programming.

The experiment does NOT constitute an empirical proof of Theorem 1.
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
MAX_DEGREE = 12

TRAIN_GRID_SIZE = 31
EVAL_GRID_SIZE = 201

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)


# ============================================================
# Target functionals
# ============================================================

def phi_1(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Smooth periodic structure."""
    return np.sin(np.pi * u) * np.cos(np.pi * v)


def phi_2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Localised smooth structure."""
    return np.exp(
        -8.0 * ((u - 0.5) ** 2 + (v - 0.5) ** 2)
    )


def phi_3(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Smooth rational structure."""
    return 1.0 / (1.0 + 10.0 * (u**2 + v**2))


def phi_4(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Non-separable interaction."""
    return np.sin(3.0 * np.pi * u * v)


def phi_5(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sharp but continuous transition."""
    return np.tanh(8.0 * (u + v - 1.0))


TARGETS = {
    "phi1_periodic": phi_1,
    "phi2_localised": phi_2,
    "phi3_rational": phi_3,
    "phi4_interaction": phi_4,
    "phi5_transition": phi_5,
}


# ============================================================
# Grid construction
# ============================================================

def make_training_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, n)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    return uu.ravel(), vv.ravel()


def make_evaluation_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Dense grid shifted by half a grid interval so that the evaluation
    locations do not systematically coincide with the training grid.
    """
    step = 1.0 / n
    axis = np.linspace(
        step / 2.0,
        1.0 - step / 2.0,
        n
    )

    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    return uu.ravel(), vv.ravel()


# ============================================================
# Polynomial basis
# ============================================================

def scale_to_legendre_domain(x: np.ndarray) -> np.ndarray:
    """Map [0,1] to [-1,1]."""
    return 2.0 * x - 1.0


def total_degree_indices(degree: int) -> list[tuple[int, int]]:
    """
    Return (i,j) satisfying i+j <= degree.
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

    lu = legvander(us, degree)
    lv = legvander(vs, degree)

    indices = total_degree_indices(degree)

    columns = [
        lu[:, i] * lv[:, j]
        for i, j in indices
    ]

    X = np.column_stack(columns)

    return X, indices


# ============================================================
# L2 approximation
# ============================================================

def fit_l2(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:

    beta, _, _, _ = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )

    return beta


# ============================================================
# L-infinity-oriented approximation
# ============================================================

def fit_linf(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Solve

        min t

    subject to

        |X beta - y| <= t

    on the training grid.
    """

    n_samples, n_features = X.shape

    # Variables are [beta_1, ..., beta_p, t]
    c = np.zeros(n_features + 1)
    c[-1] = 1.0

    # X beta - y <= t
    # X beta - t <= y
    upper_1 = np.column_stack(
        [X, -np.ones(n_samples)]
    )
    bound_1 = y

    # y - X beta <= t
    # -X beta - t <= -y
    upper_2 = np.column_stack(
        [-X, -np.ones(n_samples)]
    )
    bound_2 = -y

    A_ub = np.vstack([upper_1, upper_2])
    b_ub = np.concatenate([bound_1, bound_2])

    bounds = [
        (None, None)
        for _ in range(n_features)
    ]
    bounds.append((0.0, None))

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(
            f"L-infinity optimisation failed: "
            f"{result.message}"
        )

    return result.x[:-1]


# ============================================================
# Evaluation metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:

    error = y_true - y_pred

    return {
        "rmse": float(
            np.sqrt(np.mean(error**2))
        ),
        "mae": float(
            np.mean(np.abs(error))
        ),
        "max_error": float(
            np.max(np.abs(error))
        ),
    }


# ============================================================
# Main experiment
# ============================================================

def run_experiment() -> pd.DataFrame:

    train_u, train_v = make_training_grid(
        TRAIN_GRID_SIZE
    )

    eval_u, eval_v = make_evaluation_grid(
        EVAL_GRID_SIZE
    )

    records = []

    for target_name, target_function in TARGETS.items():

        print(f"\nTarget: {target_name}")

        y_train = target_function(
            train_u,
            train_v
        )

        y_eval = target_function(
            eval_u,
            eval_v
        )

        for degree in range(
            MIN_DEGREE,
            MAX_DEGREE + 1
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

            condition_number = float(
                np.linalg.cond(X_train)
            )

            # ----------------------------------------
            # L2
            # ----------------------------------------

            beta_l2 = fit_l2(
                X_train,
                y_train
            )

            pred_train_l2 = X_train @ beta_l2
            pred_eval_l2 = X_eval @ beta_l2

            train_l2 = calculate_metrics(
                y_train,
                pred_train_l2
            )

            eval_l2 = calculate_metrics(
                y_eval,
                pred_eval_l2
            )

            records.append({
                "target": target_name,
                "degree": degree,
                "n_terms": len(indices),
                "method": "L2",
                "train_rmse": train_l2["rmse"],
                "train_mae": train_l2["mae"],
                "train_max_error": train_l2["max_error"],
                "eval_rmse": eval_l2["rmse"],
                "eval_mae": eval_l2["mae"],
                "eval_max_error": eval_l2["max_error"],
                "condition_number": condition_number,
            })

            # ----------------------------------------
            # L-infinity-oriented
            # ----------------------------------------

            beta_linf = fit_linf(
                X_train,
                y_train
            )

            pred_train_linf = X_train @ beta_linf
            pred_eval_linf = X_eval @ beta_linf

            train_linf = calculate_metrics(
                y_train,
                pred_train_linf
            )

            eval_linf = calculate_metrics(
                y_eval,
                pred_eval_linf
            )

            records.append({
                "target": target_name,
                "degree": degree,
                "n_terms": len(indices),
                "method": "Linf",
                "train_rmse": train_linf["rmse"],
                "train_mae": train_linf["mae"],
                "train_max_error": train_linf["max_error"],
                "eval_rmse": eval_linf["rmse"],
                "eval_mae": eval_linf["mae"],
                "eval_max_error": eval_linf["max_error"],
                "condition_number": condition_number,
            })

            print(" done")

    return pd.DataFrame(records)


# ============================================================
# Figures
# ============================================================

def plot_max_error(df: pd.DataFrame) -> None:

    for method in ["L2", "Linf"]:

        subset = df[df["method"] == method]

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        for target in TARGETS:

            target_data = subset[
                subset["target"] == target
            ].sort_values("degree")

            ax.plot(
                target_data["degree"],
                target_data["eval_max_error"],
                marker="o",
                label=target,
            )

        ax.set_xlabel("Maximum polynomial degree")
        ax.set_ylabel(
            "Empirical maximum approximation error"
        )

        ax.set_title(
            f"RQ1 expressive approximation ({method})"
        )

        ax.legend()
        ax.grid(True, alpha=0.25)

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR
            / f"rq1_max_error_{method.lower()}.pdf",
            bbox_inches="tight",
        )

        fig.savefig(
            FIGURES_DIR
            / f"rq1_max_error_{method.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_rmse(df: pd.DataFrame) -> None:

    for method in ["L2", "Linf"]:

        subset = df[df["method"] == method]

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        for target in TARGETS:

            target_data = subset[
                subset["target"] == target
            ].sort_values("degree")

            ax.plot(
                target_data["degree"],
                target_data["eval_rmse"],
                marker="o",
                label=target,
            )

        ax.set_xlabel("Maximum polynomial degree")
        ax.set_ylabel("Evaluation RMSE")

        ax.set_title(
            f"RQ1 RMSE ({method})"
        )

        ax.legend()
        ax.grid(True, alpha=0.25)

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR
            / f"rq1_rmse_{method.lower()}.pdf",
            bbox_inches="tight",
        )

        plt.close(fig)


# ============================================================
# Metadata
# ============================================================

def save_metadata() -> None:

    metadata = {
        "experiment": "RQ1",
        "seed": SEED,
        "domain": "[0,1]^2",
        "min_degree": MIN_DEGREE,
        "max_degree": MAX_DEGREE,
        "training_grid_size_per_axis":
            TRAIN_GRID_SIZE,
        "evaluation_grid_size_per_axis":
            EVAL_GRID_SIZE,
        "training_points":
            TRAIN_GRID_SIZE**2,
        "evaluation_points":
            EVAL_GRID_SIZE**2,
        "basis":
            "total-degree tensor-product Legendre",
        "fit_methods": [
            "L2 least squares",
            "grid-based L-infinity linear programming",
        ],
        "targets": list(TARGETS.keys()),
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
        RESULTS_DIR / "rq1_metadata.json",
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

    print("=" * 60)
    print("RQ1 - Controlled Expressive Approximation")
    print("=" * 60)

    df = run_experiment()

    output_csv = (
        RESULTS_DIR
        / "rq1_results.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    save_metadata()

    plot_max_error(df)
    plot_rmse(df)

    print("\nExperiment completed.")
    print(f"Results: {output_csv}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
