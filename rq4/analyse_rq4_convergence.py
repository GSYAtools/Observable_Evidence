#!/usr/bin/env python3

"""
RQ4 - Decomposition of sequential sample-size convergence

Purpose
-------
Determine whether changes in calibrated response rho across sample-size
checkpoints are primarily associated with:

    1. changes in the raw MMD^2 estimate,
    2. changes in the calibrated nominal reference bound theta,
    3. or both.

The script DOES NOT recompute embeddings or MMD.

Inputs
------
Produced by validate_rq4_sample_size_v2.py:

    rq4_sample_size_calibration.csv
    rq4_sample_size_condition_summary.csv
    rq4_sample_size_global.csv

Expected condition-level fields include:

    provider
    scenario_id
    perturbation_family
    lambda
    sample_size
    median_mmd2
    median_rho
    exceedance_rate

Expected calibration fields include:

    provider
    scenario_id
    sample_size
    threshold

Outputs
-------
rq4_convergence_condition.csv
rq4_convergence_transition.csv
rq4_convergence_provider.csv
rq4_convergence_family.csv
rq4_convergence_metadata.json

The main output is rq4_convergence_transition.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


# ============================================================
# Utilities
# ============================================================

def relative_change(
    previous: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """
    Relative difference using the newer checkpoint as denominator:

        |x_previous - x_current| / |x_current|

    This matches the sequential interpretation already used in the
    sample-size validation.
    """

    previous = np.asarray(
        previous,
        dtype=float,
    )

    current = np.asarray(
        current,
        dtype=float,
    )

    denominator = np.maximum(
        np.abs(current),
        EPS,
    )

    return (
        np.abs(
            previous - current
        )
        / denominator
    )


def signed_relative_change(
    previous: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """
    Signed change from previous checkpoint to current checkpoint:

        (current - previous) / |current|

    Positive:
        quantity increased at the newer checkpoint.

    Negative:
        quantity decreased.
    """

    previous = np.asarray(
        previous,
        dtype=float,
    )

    current = np.asarray(
        current,
        dtype=float,
    )

    denominator = np.maximum(
        np.abs(current),
        EPS,
    )

    return (
        current - previous
    ) / denominator


def q90(series: pd.Series) -> float:

    return float(
        np.quantile(
            series,
            0.90,
        )
    )


def q25(series: pd.Series) -> float:

    return float(
        np.quantile(
            series,
            0.25,
        )
    )


def q75(series: pd.Series) -> float:

    return float(
        np.quantile(
            series,
            0.75,
        )
    )


# ============================================================
# Load and validate
# ============================================================

def load_inputs(
    input_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame | None,
]:

    calibration_path = (
        input_dir
        / "rq4_sample_size_calibration.csv"
    )

    condition_path = (
        input_dir
        / "rq4_sample_size_condition_summary.csv"
    )

    global_path = (
        input_dir
        / "rq4_sample_size_global.csv"
    )

    if not calibration_path.exists():

        raise FileNotFoundError(
            calibration_path
        )

    if not condition_path.exists():

        raise FileNotFoundError(
            condition_path
        )

    calibration = pd.read_csv(
        calibration_path
    )

    conditions = pd.read_csv(
        condition_path
    )

    global_df = None

    if global_path.exists():

        global_df = pd.read_csv(
            global_path
        )

    required_calibration = {
        "provider",
        "scenario_id",
        "sample_size",
        "threshold",
    }

    required_conditions = {
        "provider",
        "scenario_id",
        "perturbation_family",
        "lambda",
        "sample_size",
        "median_mmd2",
        "median_rho",
        "exceedance_rate",
    }

    missing_calibration = (
        required_calibration
        - set(
            calibration.columns
        )
    )

    missing_conditions = (
        required_conditions
        - set(
            conditions.columns
        )
    )

    if missing_calibration:

        raise RuntimeError(
            "Calibration CSV missing fields: "
            + ", ".join(
                sorted(
                    missing_calibration
                )
            )
        )

    if missing_conditions:

        raise RuntimeError(
            "Condition CSV missing fields: "
            + ", ".join(
                sorted(
                    missing_conditions
                )
            )
        )

    return (
        calibration,
        conditions,
        global_df,
    )


# ============================================================
# Merge theta into condition-level results
# ============================================================

def build_condition_table(
    calibration: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:

    theta = (
        calibration[
            [
                "provider",
                "scenario_id",
                "sample_size",
                "threshold",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "threshold": "theta"
            }
        )
    )

    duplicates = (
        theta
        .duplicated(
            subset=[
                "provider",
                "scenario_id",
                "sample_size",
            ],
            keep=False,
        )
    )

    if duplicates.any():

        raise RuntimeError(
            "Calibration contains multiple "
            "thresholds for the same "
            "provider/scenario/sample_size."
        )

    merged = conditions.merge(
        theta,
        on=[
            "provider",
            "scenario_id",
            "sample_size",
        ],
        how="left",
        validate="many_to_one",
    )

    if merged["theta"].isna().any():

        bad = (
            merged[
                merged[
                    "theta"
                ].isna()
            ][
                [
                    "provider",
                    "scenario_id",
                    "sample_size",
                ]
            ]
            .drop_duplicates()
        )

        raise RuntimeError(
            "Missing calibration thresholds:\n"
            + bad.to_string(
                index=False
            )
        )

    # Internal consistency check.
    reconstructed_rho = (
        merged["median_mmd2"]
        / np.maximum(
            merged["theta"],
            EPS,
        )
    )

    relative_error = (
        np.abs(
            reconstructed_rho
            - merged["median_rho"]
        )
        / np.maximum(
            np.abs(
                merged["median_rho"]
            ),
            EPS,
        )
    )

    merged[
        "rho_reconstruction_relative_error"
    ] = relative_error

    return merged


# ============================================================
# Sequential transitions
# ============================================================

def build_transitions(
    condition_table: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[int],
]:

    sample_sizes = sorted(
        int(x)
        for x in condition_table[
            "sample_size"
        ].unique()
    )

    if len(sample_sizes) < 2:

        raise RuntimeError(
            "At least two sample sizes are required."
        )

    key_columns = [
        "provider",
        "scenario_id",
        "perturbation_family",
        "lambda",
    ]

    transition_records = []

    for previous_n, current_n in zip(
        sample_sizes[:-1],
        sample_sizes[1:],
    ):

        previous = (
            condition_table[
                condition_table[
                    "sample_size"
                ]
                == previous_n
            ][
                key_columns
                + [
                    "median_mmd2",
                    "median_rho",
                    "theta",
                    "exceedance_rate",
                ]
            ]
            .copy()
        )

        current = (
            condition_table[
                condition_table[
                    "sample_size"
                ]
                == current_n
            ][
                key_columns
                + [
                    "median_mmd2",
                    "median_rho",
                    "theta",
                    "exceedance_rate",
                ]
            ]
            .copy()
        )

        previous = previous.rename(
            columns={
                "median_mmd2":
                    "mmd2_previous",

                "median_rho":
                    "rho_previous",

                "theta":
                    "theta_previous",

                "exceedance_rate":
                    "exceedance_previous",
            }
        )

        current = current.rename(
            columns={
                "median_mmd2":
                    "mmd2_current",

                "median_rho":
                    "rho_current",

                "theta":
                    "theta_current",

                "exceedance_rate":
                    "exceedance_current",
            }
        )

        merged = previous.merge(
            current,
            on=key_columns,
            how="inner",
            validate="one_to_one",
        )

        expected_previous = len(
            previous
        )

        expected_current = len(
            current
        )

        if (
            len(merged)
            != expected_previous
            or len(merged)
            != expected_current
        ):

            raise RuntimeError(
                f"Condition mismatch between "
                f"n={previous_n} and "
                f"n={current_n}: "
                f"{expected_previous} vs "
                f"{expected_current} vs "
                f"{len(merged)} matched."
            )

        merged[
            "previous_n"
        ] = previous_n

        merged[
            "current_n"
        ] = current_n

        merged[
            "transition"
        ] = (
            f"{previous_n}->{current_n}"
        )

        # ----------------------------------------------------
        # Absolute relative changes
        # ----------------------------------------------------

        merged[
            "relative_change_mmd2"
        ] = relative_change(
            merged[
                "mmd2_previous"
            ],
            merged[
                "mmd2_current"
            ],
        )

        merged[
            "relative_change_theta"
        ] = relative_change(
            merged[
                "theta_previous"
            ],
            merged[
                "theta_current"
            ],
        )

        merged[
            "relative_change_rho"
        ] = relative_change(
            merged[
                "rho_previous"
            ],
            merged[
                "rho_current"
            ],
        )

        # ----------------------------------------------------
        # Signed changes
        # ----------------------------------------------------

        merged[
            "signed_change_mmd2"
        ] = signed_relative_change(
            merged[
                "mmd2_previous"
            ],
            merged[
                "mmd2_current"
            ],
        )

        merged[
            "signed_change_theta"
        ] = signed_relative_change(
            merged[
                "theta_previous"
            ],
            merged[
                "theta_current"
            ],
        )

        merged[
            "signed_change_rho"
        ] = signed_relative_change(
            merged[
                "rho_previous"
            ],
            merged[
                "rho_current"
            ],
        )

        # ----------------------------------------------------
        # Exceedance behaviour
        # ----------------------------------------------------

        merged[
            "absolute_change_exceedance_rate"
        ] = np.abs(
            merged[
                "exceedance_current"
            ]
            - merged[
                "exceedance_previous"
            ]
        )

        # ----------------------------------------------------
        # Direction indicators
        # ----------------------------------------------------

        merged[
            "theta_decreased"
        ] = (
            merged[
                "theta_current"
            ]
            <
            merged[
                "theta_previous"
            ]
        )

        merged[
            "mmd2_increased"
        ] = (
            merged[
                "mmd2_current"
            ]
            >
            merged[
                "mmd2_previous"
            ]
        )

        merged[
            "rho_increased"
        ] = (
            merged[
                "rho_current"
            ]
            >
            merged[
                "rho_previous"
            ]
        )

        transition_records.append(
            merged
        )

    transitions = pd.concat(
        transition_records,
        ignore_index=True,
    )

    return (
        transitions,
        sample_sizes,
    )


# ============================================================
# Aggregate summaries
# ============================================================

def aggregate_transition(
    transitions: pd.DataFrame,
) -> pd.DataFrame:

    result = (
        transitions
        .groupby(
            [
                "previous_n",
                "current_n",
                "transition",
            ],
            as_index=False,
        )
        .agg(
            median_relative_change_mmd2=(
                "relative_change_mmd2",
                "median",
            ),

            q90_relative_change_mmd2=(
                "relative_change_mmd2",
                q90,
            ),

            median_relative_change_theta=(
                "relative_change_theta",
                "median",
            ),

            q90_relative_change_theta=(
                "relative_change_theta",
                q90,
            ),

            median_relative_change_rho=(
                "relative_change_rho",
                "median",
            ),

            q90_relative_change_rho=(
                "relative_change_rho",
                q90,
            ),

            median_abs_change_exceedance=(
                "absolute_change_exceedance_rate",
                "median",
            ),

            q90_abs_change_exceedance=(
                "absolute_change_exceedance_rate",
                q90,
            ),

            median_signed_change_mmd2=(
                "signed_change_mmd2",
                "median",
            ),

            median_signed_change_theta=(
                "signed_change_theta",
                "median",
            ),

            median_signed_change_rho=(
                "signed_change_rho",
                "median",
            ),

            proportion_theta_decreased=(
                "theta_decreased",
                "mean",
            ),

            proportion_mmd2_increased=(
                "mmd2_increased",
                "mean",
            ),

            proportion_rho_increased=(
                "rho_increased",
                "mean",
            ),

            n_conditions=(
                "scenario_id",
                "size",
            ),
        )
    )

    return result


def aggregate_provider(
    transitions: pd.DataFrame,
) -> pd.DataFrame:

    return (
        transitions
        .groupby(
            [
                "provider",
                "previous_n",
                "current_n",
                "transition",
            ],
            as_index=False,
        )
        .agg(
            median_relative_change_mmd2=(
                "relative_change_mmd2",
                "median",
            ),

            median_relative_change_theta=(
                "relative_change_theta",
                "median",
            ),

            median_relative_change_rho=(
                "relative_change_rho",
                "median",
            ),

            median_abs_change_exceedance=(
                "absolute_change_exceedance_rate",
                "median",
            ),

            proportion_theta_decreased=(
                "theta_decreased",
                "mean",
            ),

            proportion_rho_increased=(
                "rho_increased",
                "mean",
            ),
        )
    )


def aggregate_family(
    transitions: pd.DataFrame,
) -> pd.DataFrame:

    return (
        transitions
        .groupby(
            [
                "perturbation_family",
                "previous_n",
                "current_n",
                "transition",
            ],
            as_index=False,
        )
        .agg(
            median_relative_change_mmd2=(
                "relative_change_mmd2",
                "median",
            ),

            median_relative_change_theta=(
                "relative_change_theta",
                "median",
            ),

            median_relative_change_rho=(
                "relative_change_rho",
                "median",
            ),

            median_abs_change_exceedance=(
                "absolute_change_exceedance_rate",
                "median",
            ),

            proportion_theta_decreased=(
                "theta_decreased",
                "mean",
            ),

            proportion_rho_increased=(
                "rho_increased",
                "mean",
            ),
        )
    )


# ============================================================
# Theta-only summary
# ============================================================

def aggregate_theta(
    calibration: pd.DataFrame,
) -> pd.DataFrame:

    return (
        calibration
        .groupby(
            "sample_size",
            as_index=False,
        )
        .agg(
            median_theta=(
                "threshold",
                "median",
            ),

            q25_theta=(
                "threshold",
                q25,
            ),

            q75_theta=(
                "threshold",
                q75,
            ),

            mean_theta=(
                "threshold",
                "mean",
            ),
        )
        .sort_values(
            "sample_size"
        )
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="sample_size_analysis",
    )

    parser.add_argument(
        "--output-dir",
        default="convergence_analysis",
    )

    args = parser.parse_args()

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("RQ4 SAMPLE-SIZE DECOMPOSITION")
    print("=" * 72)

    (
        calibration,
        conditions,
        global_df,
    ) = load_inputs(
        input_dir
    )

    condition_table = (
        build_condition_table(
            calibration,
            conditions,
        )
    )

    (
        transitions,
        sample_sizes,
    ) = build_transitions(
        condition_table
    )

    transition_summary = (
        aggregate_transition(
            transitions
        )
    )

    provider_summary = (
        aggregate_provider(
            transitions
        )
    )

    family_summary = (
        aggregate_family(
            transitions
        )
    )

    theta_summary = (
        aggregate_theta(
            calibration
        )
    )

    # ========================================================
    # Save
    # ========================================================

    condition_table.to_csv(
        output_dir
        / "rq4_convergence_condition.csv",
        index=False,
    )

    transitions.to_csv(
        output_dir
        / "rq4_convergence_transition_raw.csv",
        index=False,
    )

    transition_summary.to_csv(
        output_dir
        / "rq4_convergence_transition.csv",
        index=False,
    )

    provider_summary.to_csv(
        output_dir
        / "rq4_convergence_provider.csv",
        index=False,
    )

    family_summary.to_csv(
        output_dir
        / "rq4_convergence_family.csv",
        index=False,
    )

    theta_summary.to_csv(
        output_dir
        / "rq4_convergence_theta.csv",
        index=False,
    )

    metadata = {
        "sample_sizes":
            sample_sizes,

        "transitions": [
            f"{a}->{b}"
            for a, b in zip(
                sample_sizes[:-1],
                sample_sizes[1:],
            )
        ],

        "relative_change_definition":
            (
                "|previous-current|/"
                "|current|"
            ),

        "signed_change_definition":
            (
                "(current-previous)/"
                "|current|"
            ),

        "decomposition_variables": [
            "median_mmd2",
            "theta",
            "median_rho",
        ],

        "source_directory":
            str(input_dir),
    }

    with (
        output_dir
        / "rq4_convergence_metadata.json"
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
    # Console output
    # ========================================================

    print()
    print(
        "Detected sample sizes:",
        sample_sizes,
    )

    print()
    print("=" * 72)
    print("SEQUENTIAL DECOMPOSITION")
    print("=" * 72)

    display_columns = [
        "transition",
        "median_relative_change_mmd2",
        "median_relative_change_theta",
        "median_relative_change_rho",
        "median_abs_change_exceedance",
        "proportion_theta_decreased",
        "proportion_rho_increased",
    ]

    print(
        transition_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("THETA TRAJECTORY")
    print("=" * 72)

    print(
        theta_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Latest transition
    # --------------------------------------------------------

    latest = (
        transition_summary
        .sort_values(
            "current_n"
        )
        .iloc[-1]
    )

    print()
    print("=" * 72)
    print("LATEST CHECKPOINT")
    print("=" * 72)

    print(
        f"Transition: "
        f"{latest['transition']}"
    )

    print(
        "Median relative MMD2 change: "
        f"{latest['median_relative_change_mmd2']:.4f}"
    )

    print(
        "Median relative theta change: "
        f"{latest['median_relative_change_theta']:.4f}"
    )

    print(
        "Median relative rho change: "
        f"{latest['median_relative_change_rho']:.4f}"
    )

    print(
        "Median absolute exceedance-rate change: "
        f"{latest['median_abs_change_exceedance']:.4f}"
    )

    print(
        "Proportion of calibration bounds "
        "that decreased: "
        f"{latest['proportion_theta_decreased']:.4f}"
    )

    print(
        "Proportion of calibrated responses "
        "that increased: "
        f"{latest['proportion_rho_increased']:.4f}"
    )

    print()
    print(
        f"Outputs: {output_dir}"
    )


if __name__ == "__main__":
    main()

