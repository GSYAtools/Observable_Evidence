#!/usr/bin/env python3

"""
RQ4 - Incremental semantic embedding builder

Reads one or more generation JSONL files and computes embeddings only
for successful generations whose job_id has not previously been embedded.

Frozen semantic representation
------------------------------
Model:
    BAAI/bge-small-en-v1.5

Dimension:
    384

Normalisation:
    L2 normalised

The script is incremental:
    - existing job_ids are skipped
    - only new responses are encoded
    - existing embeddings are preserved
    - index and embedding matrix remain row-aligned

Typical usage
-------------
python embed_rq4_incremental.py \
    --input results/rq4_openai.jsonl \
    --provider openai

Run again later after new generations have been appended.
Only the new successful generations will be embedded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# Frozen representation configuration
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

EXPECTED_DIMENSION = 384

DEFAULT_BATCH_SIZE = 128

DEFAULT_OUTPUT_DIR = Path("embeddings")


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:

    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_text(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# Load successful generations
# ============================================================

def load_successful_generations(
    input_path: Path,
) -> pd.DataFrame:

    records = []

    malformed_lines = 0
    error_records = 0
    empty_successes = 0

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError:

                malformed_lines += 1

                print(
                    f"WARNING: malformed JSON "
                    f"at line {line_number}"
                )

                continue

            if record.get(
                "status"
            ) != "ok":

                error_records += 1
                continue

            text = record.get(
                "text",
                ""
            )

            if not isinstance(
                text,
                str,
            ) or not text.strip():

                empty_successes += 1

                print(
                    "WARNING: successful job "
                    "without text: "
                    f"{record.get('job_id')}"
                )

                continue

            records.append(
                record
            )

    if not records:

        raise RuntimeError(
            "No valid successful generations "
            f"found in {input_path}."
        )

    df = pd.DataFrame(
        records
    )

    required = [
        "job_id",
        "provider",
        "scenario_id",
        "condition",
        "lambda",
        "repetition",
        "text",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required fields: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # A successful job_id must occur only once.
    # --------------------------------------------------------

    duplicate_mask = (
        df["job_id"]
        .duplicated(
            keep=False
        )
    )

    if duplicate_mask.any():

        duplicate_ids = (
            df.loc[
                duplicate_mask,
                "job_id",
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Duplicate successful job IDs "
            "found in generation file: "
            f"{duplicate_ids[:10]}"
        )

    print(
        f"Successful records:   {len(df)}"
    )

    print(
        f"Provider errors:      {error_records}"
    )

    print(
        f"Malformed lines:      {malformed_lines}"
    )

    print(
        f"Empty successes:      {empty_successes}"
    )

    return (
        df
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Existing representation
# ============================================================

def load_existing_state(
    embedding_path: Path,
    index_path: Path,
) -> tuple[np.ndarray, pd.DataFrame]:

    embeddings_exist = (
        embedding_path.exists()
    )

    index_exists = (
        index_path.exists()
    )

    # They must either both exist or both not exist.

    if (
        embeddings_exist
        != index_exists
    ):

        raise RuntimeError(
            "Incremental state is inconsistent. "
            "Embedding matrix and index must "
            "either both exist or both be absent."
        )

    if not embeddings_exist:

        empty_embeddings = np.empty(
            (
                0,
                EXPECTED_DIMENSION,
            ),
            dtype=np.float32,
        )

        empty_index = pd.DataFrame()

        return (
            empty_embeddings,
            empty_index,
        )

    embeddings = np.load(
        embedding_path
    )

    index_df = pd.read_csv(
        index_path
    )

    if embeddings.ndim != 2:

        raise RuntimeError(
            "Existing embedding matrix "
            f"has shape {embeddings.shape}."
        )

    if (
        embeddings.shape[1]
        != EXPECTED_DIMENSION
    ):

        raise RuntimeError(
            "Existing embedding dimension "
            f"is {embeddings.shape[1]}, "
            f"expected {EXPECTED_DIMENSION}."
        )

    if (
        len(index_df)
        != embeddings.shape[0]
    ):

        raise RuntimeError(
            "Embedding/index row mismatch: "
            f"{embeddings.shape[0]} vectors "
            f"versus {len(index_df)} index rows."
        )

    if (
        not index_df.empty
        and index_df[
            "job_id"
        ].duplicated().any()
    ):

        raise RuntimeError(
            "Existing embedding index contains "
            "duplicate job_ids."
        )

    # Verify explicit row alignment if present.

    if (
        not index_df.empty
        and "embedding_row"
        in index_df.columns
    ):

        expected_rows = np.arange(
            len(index_df)
        )

        actual_rows = (
            index_df[
                "embedding_row"
            ]
            .to_numpy()
        )

        if not np.array_equal(
            expected_rows,
            actual_rows,
        ):

            raise RuntimeError(
                "Existing embedding_row values "
                "are not sequential."
            )

    return (
        embeddings.astype(
            np.float32,
            copy=False,
        ),
        index_df,
    )


# ============================================================
# Atomic writers
# ============================================================

def atomic_save_npy(
    path: Path,
    array: np.ndarray,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        path.parent
        / (
            "."
            + path.name
            + ".tmp.npy"
        )
    )

    np.save(
        temp_path,
        array,
    )

    os.replace(
        temp_path,
        path,
    )


def atomic_save_csv(
    path: Path,
    df: pd.DataFrame,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        path.parent
        / (
            "."
            + path.name
            + ".tmp"
        )
    )

    df.to_csv(
        temp_path,
        index=False,
    )

    os.replace(
        temp_path,
        path,
    )


def atomic_save_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        path.parent
        / (
            "."
            + path.name
            + ".tmp"
        )
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# Build index rows
# ============================================================

def build_index_rows(
    df: pd.DataFrame,
    start_row: int,
) -> pd.DataFrame:

    columns = [
        "job_id",
        "provider",
        "scenario_id",
        "task_family",
        "condition",
        "perturbation_family",
        "lambda",
        "repetition",
        "prompt_sha256",
        "output_word_count",
        "model_requested",
        "model_returned",
        "timestamp_utc",
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    result = (
        df[
            available_columns
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    result.insert(
        0,
        "embedding_row",
        np.arange(
            start_row,
            start_row + len(result),
        ),
    )

    result[
        "response_sha256"
    ] = [
        sha256_text(text)
        for text in df[
            "text"
        ].tolist()
    ]

    result[
        "embedded_at_utc"
    ] = utc_now()

    result[
        "embedding_model"
    ] = EMBEDDING_MODEL

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Provider generation JSONL file."
        ),
    )

    parser.add_argument(
        "--provider",
        required=True,
        choices=[
            "openai",
            "gemini",
            "qwen",
        ],
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional maximum number of NEW "
            "responses to embed in this run. "
            "Useful for testing incremental behaviour."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show how many responses require "
            "embedding without loading the model."
        ),
    )

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit < 1
    ):

        parser.error(
            "--limit must be >= 1"
        )

    input_path = Path(
        args.input
    )

    output_dir = Path(
        args.output_dir
    )

    prefix = (
        f"rq4_{args.provider}"
    )

    embedding_path = (
        output_dir
        / f"{prefix}_embeddings.npy"
    )

    index_path = (
        output_dir
        / f"{prefix}_embedding_index.csv"
    )

    metadata_path = (
        output_dir
        / f"{prefix}_embedding_metadata.json"
    )

    print("=" * 72)
    print("RQ4 INCREMENTAL SEMANTIC EMBEDDING")
    print("=" * 72)

    print(
        f"Provider:         {args.provider}"
    )

    print(
        f"Input:            {input_path}"
    )

    print(
        f"Embedding model:  {EMBEDDING_MODEL}"
    )

    # --------------------------------------------------------
    # Generation data
    # --------------------------------------------------------

    generation_df = (
        load_successful_generations(
            input_path
        )
    )

    provider_values = set(
        generation_df[
            "provider"
        ].astype(str)
    )

    if provider_values != {
        args.provider
    }:

        raise RuntimeError(
            "Input provider mismatch. "
            f"Found {provider_values}, "
            f"expected only "
            f"{args.provider}."
        )

    # --------------------------------------------------------
    # Existing representation
    # --------------------------------------------------------

    (
        existing_embeddings,
        existing_index,
    ) = load_existing_state(
        embedding_path,
        index_path,
    )

    if existing_index.empty:

        completed_ids = set()

    else:

        completed_ids = set(
            existing_index[
                "job_id"
            ].astype(str)
        )

    pending_df = (
        generation_df[
            ~generation_df[
                "job_id"
            ].isin(
                completed_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        f"Already embedded: {len(completed_ids)}"
    )

    print(
        f"Pending:          {len(pending_df)}"
    )

    if args.limit is not None:

        pending_df = (
            pending_df
            .iloc[
                :args.limit
            ]
            .copy()
        )

        print(
            f"Selected:         {len(pending_df)}"
        )

    if args.dry_run:

        if len(pending_df):

            print()
            print(
                pending_df[
                    [
                        "job_id",
                        "scenario_id",
                        "condition",
                        "perturbation_family",
                        "lambda",
                        "repetition",
                    ]
                ]
                .head(20)
                .to_string(
                    index=False
                )
            )

        return

    if pending_df.empty:

        print()
        print(
            "No new successful generations "
            "require embedding."
        )

        return

    # --------------------------------------------------------
    # Load frozen representation model
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device:           {device}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
    )

    # --------------------------------------------------------
    # Encode ONLY pending responses
    # --------------------------------------------------------

    texts = (
        pending_df[
            "text"
        ]
        .astype(str)
        .tolist()
    )

    print()
    print(
        f"Encoding {len(texts)} "
        f"new responses..."
    )

    new_embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    new_embeddings = np.asarray(
        new_embeddings,
        dtype=np.float32,
    )

    if (
        new_embeddings.ndim != 2
        or new_embeddings.shape[1]
        != EXPECTED_DIMENSION
    ):

        raise RuntimeError(
            "Unexpected new embedding shape: "
            f"{new_embeddings.shape}"
        )

    # --------------------------------------------------------
    # Append
    # --------------------------------------------------------

    start_row = (
        existing_embeddings.shape[0]
    )

    combined_embeddings = (
        np.concatenate(
            [
                existing_embeddings,
                new_embeddings,
            ],
            axis=0,
        )
    )

    new_index = build_index_rows(
        pending_df,
        start_row=start_row,
    )

    if existing_index.empty:

        combined_index = (
            new_index
        )

    else:

        combined_index = (
            pd.concat(
                [
                    existing_index,
                    new_index,
                ],
                ignore_index=True,
            )
        )

    # --------------------------------------------------------
    # Final integrity checks BEFORE writing
    # --------------------------------------------------------

    if (
        combined_embeddings.shape[0]
        != len(combined_index)
    ):

        raise RuntimeError(
            "Internal row-count mismatch."
        )

    if (
        combined_index[
            "job_id"
        ].duplicated().any()
    ):

        raise RuntimeError(
            "Duplicate job_ids after append."
        )

    expected_rows = np.arange(
        len(combined_index)
    )

    if not np.array_equal(
        combined_index[
            "embedding_row"
        ].to_numpy(),
        expected_rows,
    ):

        raise RuntimeError(
            "embedding_row alignment failure."
        )

    # --------------------------------------------------------
    # Atomic persistence
    # --------------------------------------------------------

    atomic_save_npy(
        embedding_path,
        combined_embeddings,
    )

    atomic_save_csv(
        index_path,
        combined_index,
    )

    metadata = {
        "representation":
            "R_E",

        "embedding_model":
            EMBEDDING_MODEL,

        "embedding_dimension":
            EXPECTED_DIMENSION,

        "normalised":
            True,

        "normalisation":
            "L2",

        "dtype":
            "float32",

        "provider":
            args.provider,

        "source_jsonl":
            str(input_path),

        "total_embeddings":
            int(
                combined_embeddings.shape[0]
            ),

        "new_embeddings_this_run":
            int(
                new_embeddings.shape[0]
            ),

        "device_last_run":
            device,

        "batch_size_last_run":
            args.batch_size,

        "updated_at_utc":
            utc_now(),

        "python_version":
            platform.python_version(),

        "numpy_version":
            np.__version__,

        "torch_version":
            torch.__version__,
    }

    atomic_save_json(
        metadata_path,
        metadata,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("EMBEDDING UPDATE COMPLETED")
    print("=" * 72)

    print(
        f"Previous:   "
        f"{start_row}"
    )

    print(
        f"Added:      "
        f"{len(new_embeddings)}"
    )

    print(
        f"Total:      "
        f"{len(combined_embeddings)}"
    )

    print(
        f"Shape:      "
        f"{combined_embeddings.shape}"
    )

    print()
    print(
        f"Embeddings: {embedding_path}"
    )

    print(
        f"Index:      {index_path}"
    )

    print(
        f"Metadata:   {metadata_path}"
    )


if __name__ == "__main__":
    main()
