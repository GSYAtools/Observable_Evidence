#!/usr/bin/env python3

"""
RQ4 - Heterogeneous Stochastic Intelligent Systems
Generation runner

Frozen experimental generation design
-------------------------------------
Scenarios:
    20

Nominal generations:
    60 per scenario and model

Perturbed generations:
    305 per scenario, perturbation family, and lambda

Perturbation families:
    semantic
    context_removal
    surface

Perturbation levels:
    0.25, 0.50, 0.75, 1.00

Commercial models:
    OpenAI gpt-5.4-mini-2026-03-17
        reasoning effort = none
        verbosity = low

    Google gemini-2.5-flash
        thinking budget = 0

    Qwen/Qwen3-4B
        thinking disabled
        stochastic sampling enabled

Output constraint:
    maximum 300 output tokens
    instruction to answer in <= 120 words

The runner is resumable. Every completed call is written
immediately to JSONL. --limit controls the maximum number
of NEW calls attempted in the current execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Frozen experimental configuration
# ============================================================

DEFAULT_CORPUS = "rq4_prompts.json"

RESULTS_DIR = Path("results")

OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"
GEMINI_MODEL = "gemini-2.5-flash"
QWEN_MODEL = "Qwen/Qwen3-4B"

qwen_tokenizer = None
qwen_model = None

N_NOMINAL = 60
N_PERTURBED = 30

MAX_OUTPUT_TOKENS = 300

SYSTEM_INSTRUCTION = (
    "Answer the user's task directly in no more than 120 words. "
    "Do not mention these instructions. "
    "Do not use external tools or assume information "
    "that is not contained in the prompt."
)

EXPECTED_SCENARIOS = 20

EXPECTED_PERTURBATION_FAMILIES = {
    "semantic",
    "context_removal",
    "surface",
}

EXPECTED_LAMBDAS = {
    0.25,
    0.50,
    0.75,
    1.00,
}


# ============================================================
# General utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def stable_id(*parts: Any) -> str:

    raw = "|".join(
        str(part)
        for part in parts
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:24]


def prompt_hash(prompt: str) -> str:

    return hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()


def load_corpus(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# Corpus safety checks
# ============================================================

def validate_corpus(
    corpus: dict,
) -> None:

    scenarios = corpus.get(
        "scenarios",
        []
    )

    if len(scenarios) != EXPECTED_SCENARIOS:

        raise RuntimeError(
            f"Corpus contains "
            f"{len(scenarios)} scenarios; "
            f"expected {EXPECTED_SCENARIOS}."
        )

    scenario_ids = set()

    for scenario in scenarios:

        scenario_id = scenario.get(
            "scenario_id"
        )

        if not scenario_id:

            raise RuntimeError(
                "Scenario without scenario_id."
            )

        if scenario_id in scenario_ids:

            raise RuntimeError(
                f"Duplicate scenario_id: "
                f"{scenario_id}"
            )

        scenario_ids.add(
            scenario_id
        )

        base = scenario.get(
            "base"
        )

        if (
            not isinstance(base, str)
            or not base.strip()
        ):

            raise RuntimeError(
                f"{scenario_id}: "
                "missing base prompt."
            )

        transformations = scenario.get(
            "transformations",
            {}
        )

        families = set(
            transformations.keys()
        )

        if (
            families
            != EXPECTED_PERTURBATION_FAMILIES
        ):

            raise RuntimeError(
                f"{scenario_id}: "
                f"unexpected perturbation families "
                f"{families}."
            )

        for family, items in (
            transformations.items()
        ):

            lambdas = {
                float(item["lambda"])
                for item in items
            }

            if lambdas != EXPECTED_LAMBDAS:

                raise RuntimeError(
                    f"{scenario_id}/{family}: "
                    f"unexpected lambda values "
                    f"{lambdas}."
                )


# ============================================================
# Job construction
# ============================================================

def build_jobs(
    corpus: dict,
    provider: str,
) -> list[dict]:

    jobs = []

    for scenario in corpus["scenarios"]:

        scenario_id = (
            scenario["scenario_id"]
        )

        task_family = (
            scenario["task_family"]
        )

        base_prompt = (
            scenario["base"]
        )

        # ----------------------------------------------------
        # Nominal pool
        # ----------------------------------------------------

        for repetition in range(
            N_NOMINAL
        ):

            job_id = stable_id(
                provider,
                scenario_id,
                "nominal",
                "none",
                0.0,
                repetition,
            )

            jobs.append(
                {
                    "job_id":
                        job_id,

                    "provider":
                        provider,

                    "scenario_id":
                        scenario_id,

                    "task_family":
                        task_family,

                    "condition":
                        "nominal",

                    "perturbation_family":
                        None,

                    "lambda":
                        0.0,

                    "repetition":
                        repetition,

                    "prompt":
                        base_prompt,

                    "prompt_sha256":
                        prompt_hash(
                            base_prompt
                        ),
                }
            )

        # ----------------------------------------------------
        # Perturbed conditions
        # ----------------------------------------------------

        for (
            family,
            transformations,
        ) in scenario[
            "transformations"
        ].items():

            for transformation in transformations:

                lam = float(
                    transformation[
                        "lambda"
                    ]
                )

                prompt = (
                    transformation[
                        "prompt"
                    ]
                )

                for repetition in range(
                    N_PERTURBED
                ):

                    job_id = stable_id(
                        provider,
                        scenario_id,
                        "perturbed",
                        family,
                        lam,
                        repetition,
                    )

                    jobs.append(
                        {
                            "job_id":
                                job_id,

                            "provider":
                                provider,

                            "scenario_id":
                                scenario_id,

                            "task_family":
                                task_family,

                            "condition":
                                "perturbed",

                            "perturbation_family":
                                family,

                            "lambda":
                                lam,

                            "repetition":
                                repetition,

                            "prompt":
                                prompt,

                            "prompt_sha256":
                                prompt_hash(
                                    prompt
                                ),
                        }
                    )

    return jobs


# ============================================================
# Resume support
# ============================================================

def load_completed_ids(
    output_path: Path,
) -> set[str]:

    completed = set()

    if not output_path.exists():

        return completed

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError:

                continue

            if (
                record.get("status")
                == "ok"
            ):

                job_id = record.get(
                    "job_id"
                )

                if job_id:
                    completed.add(
                        job_id
                    )

    return completed


def append_record(
    output_path: Path,
    record: dict,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()

        os.fsync(
            f.fileno()
        )


# ============================================================
# OpenAI
# ============================================================

def generate_openai(
    prompt: str,
) -> dict:

    from openai import OpenAI

    client = OpenAI()

    response = client.responses.create(
        model=OPENAI_MODEL,

        instructions=(
            SYSTEM_INSTRUCTION
        ),

        input=prompt,

        reasoning={
            "effort": "none",
        },
	text={
	    "verbosity": "low",
	},

        max_output_tokens=(
            MAX_OUTPUT_TOKENS
        ),

        store=False,
    )

    text = (
        response.output_text
        or ""
    )

    usage = getattr(
        response,
        "usage",
        None,
    )

    usage_dict = {}

    if usage is not None:

        usage_dict = {
            "input_tokens":
                getattr(
                    usage,
                    "input_tokens",
                    None,
                ),

            "output_tokens":
                getattr(
                    usage,
                    "output_tokens",
                    None,
                ),

            "total_tokens":
                getattr(
                    usage,
                    "total_tokens",
                    None,
                ),
        }

        output_details = getattr(
            usage,
            "output_tokens_details",
            None,
        )

        if output_details is not None:

            usage_dict[
                "reasoning_tokens"
            ] = getattr(
                output_details,
                "reasoning_tokens",
                None,
            )

    return {
        "text":
            text,

        "model_requested":
            OPENAI_MODEL,

        "model_returned":
            getattr(
                response,
                "model",
                None,
            ),

        "response_id":
            getattr(
                response,
                "id",
                None,
            ),

        "finish_reason":
            None,

        "usage":
            usage_dict,

        "provider_metadata": {
            "reasoning_effort":
                "none",
            "max_output_tokens":
                MAX_OUTPUT_TOKENS,
	    "verbosity":
		"low",
            "max_words_instruction":
                120,
        },
    }


# ============================================================
# Gemini
# ============================================================

def generate_gemini(
    prompt: str,
) -> dict:

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.environ[
            "GEMINI_API_KEY"
        ]
    )

    response = (
        client.models.generate_content(
            model=GEMINI_MODEL,

            contents=prompt,

            config=(
                types.GenerateContentConfig(
                    system_instruction=(
                        SYSTEM_INSTRUCTION
                    ),

                    max_output_tokens=(
                        MAX_OUTPUT_TOKENS
                    ),

                    thinking_config=(
                        types.ThinkingConfig(
                            thinking_budget=0
                        )
                    ),
                )
            ),
        )
    )

    text = (
        response.text
        or ""
    )

    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    usage_dict = {}

    if usage is not None:

        usage_dict = {
            "input_tokens":
                getattr(
                    usage,
                    "prompt_token_count",
                    None,
                ),

            "output_tokens":
                getattr(
                    usage,
                    "candidates_token_count",
                    None,
                ),

            "total_tokens":
                getattr(
                    usage,
                    "total_token_count",
                    None,
                ),

            "reasoning_tokens":
                getattr(
                    usage,
                    "thoughts_token_count",
                    None,
                ),
        }

    finish_reason = None

    candidates = getattr(
        response,
        "candidates",
        None,
    )

    if candidates:

        finish_reason_obj = getattr(
            candidates[0],
            "finish_reason",
            None,
        )

        if (
            finish_reason_obj
            is not None
        ):

            finish_reason = str(
                finish_reason_obj
            )

    return {
        "text":
            text,

        "model_requested":
            GEMINI_MODEL,

        "model_returned":
            GEMINI_MODEL,

        "response_id":
            getattr(
                response,
                "response_id",
                None,
            ),

        "finish_reason":
            finish_reason,

        "usage":
            usage_dict,

        "provider_metadata": {
            "thinking_budget":
                0,
            "max_output_tokens":
                MAX_OUTPUT_TOKENS,
            "max_words_instruction":
                120,
        },
    }


# ============================================================
# Qwen local provider
# ============================================================

def initialise_qwen() -> None:
    global qwen_tokenizer, qwen_model

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading local model: {QWEN_MODEL}", flush=True)

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype="auto",
        device_map="auto",
    )
    qwen_model.eval()

    print(
        f"Qwen loaded on {qwen_model.device}; "
        f"dtype={next(qwen_model.parameters()).dtype}",
        flush=True,
    )


def generate_qwen(prompt: str) -> dict:
    global qwen_tokenizer, qwen_model

    if qwen_tokenizer is None or qwen_model is None:
        raise RuntimeError("Qwen has not been initialised.")

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]

    formatted = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = qwen_tokenizer(
        formatted,
        return_tensors="pt",
    ).to(qwen_model.device)

    input_length = int(inputs["input_ids"].shape[1])

    outputs = qwen_model.generate(
        **inputs,
        max_new_tokens=MAX_OUTPUT_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=qwen_tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][input_length:]
    response_text = qwen_tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    output_tokens = int(generated_ids.shape[0])

    return {
        "text": response_text,
        "model_requested": QWEN_MODEL,
        "model_returned": QWEN_MODEL,
        "response_id": None,
        "finish_reason": None,
        "usage": {
            "input_tokens": input_length,
            "output_tokens": output_tokens,
            "total_tokens": input_length + output_tokens,
            "reasoning_tokens": 0,
        },
        "provider_metadata": {
            "thinking": False,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "max_words_instruction": 120,
        },
    }


# ============================================================
# Provider dispatcher
# ============================================================

def execute_job(
    job: dict,
) -> dict:

    started = time.time()

    try:

        if (
            job["provider"]
            == "openai"
        ):

            result = (
                generate_openai(
                    job["prompt"]
                )
            )

        elif (
            job["provider"]
            == "gemini"
        ):

            result = (
                generate_gemini(
                    job["prompt"]
                )
            )

        elif (
            job["provider"]
            == "qwen"
        ):

            result = (
                generate_qwen(
                    job["prompt"]
                )
            )

        else:

            raise ValueError(
                "Unsupported provider: "
                f"{job['provider']}"
            )

        elapsed = (
            time.time()
            - started
        )

        text = result.get(
            "text",
            ""
        )

        word_count = len(
            text.split()
        )

        return {
            **job,

            "status":
                "ok",

            "timestamp_utc":
                utc_now(),

            "elapsed_seconds":
                elapsed,

            "output_word_count":
                word_count,

            **result,
        }

    except Exception as exc:

        elapsed = (
            time.time()
            - started
        )

        return {
            **job,

            "status":
                "error",

            "timestamp_utc":
                utc_now(),

            "elapsed_seconds":
                elapsed,

            "error_type":
                type(exc).__name__,

            "error_message":
                str(exc),
        }


# ============================================================
# Environment
# ============================================================

def validate_environment(
    provider: str,
) -> None:

    if provider == "openai":

        if not os.getenv(
            "OPENAI_API_KEY"
        ):

            raise RuntimeError(
                "OPENAI_API_KEY "
                "is not defined."
            )

    elif provider == "gemini":

        if not os.getenv(
            "GEMINI_API_KEY"
        ):

            raise RuntimeError(
                "GEMINI_API_KEY "
                "is not defined."
            )

    elif provider == "qwen":
        pass


# ============================================================
# Default output
# ============================================================

def default_output_path(
    provider: str,
) -> Path:

    return (
        RESULTS_DIR
        / f"rq4_{provider}.jsonl"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

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
        "--corpus",
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of NEW calls "
            "attempted during this execution. "
            "--limit 1 performs one pending "
            "generation and exits."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Inspect pending jobs without "
            "calling the provider."
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

    corpus_path = Path(
        args.corpus
    )

    if args.output is None:

        output_path = (
            default_output_path(
                args.provider
            )
        )

    else:

        output_path = Path(
            args.output
        )

    # --------------------------------------------------------
    # Corpus
    # --------------------------------------------------------

    corpus = load_corpus(
        corpus_path
    )

    validate_corpus(
        corpus
    )

    # --------------------------------------------------------
    # Jobs
    # --------------------------------------------------------

    jobs = build_jobs(
        corpus,
        args.provider,
    )

    expected_jobs = (
        EXPECTED_SCENARIOS
        * (
            N_NOMINAL
            +
            len(
                EXPECTED_PERTURBATION_FAMILIES
            )
            * len(
                EXPECTED_LAMBDAS
            )
            * N_PERTURBED
        )
    )

    if len(jobs) != expected_jobs:

        raise RuntimeError(
            f"Built {len(jobs)} jobs; "
            f"expected {expected_jobs}."
        )

    completed_ids = (
        load_completed_ids(
            output_path
        )
    )

    pending_jobs = [
        job
        for job in jobs
        if (
            job["job_id"]
            not in completed_ids
        )
    ]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 72)
    print("RQ4 GENERATION RUNNER")
    print("=" * 72)

    print(
        f"Provider:          "
        f"{args.provider}"
    )

    print(
        f"Corpus:            "
        f"{corpus_path}"
    )

    print(
        f"Output:            "
        f"{output_path}"
    )

    print(
        f"Scenarios:         "
        f"{EXPECTED_SCENARIOS}"
    )

    print(
        f"Nominal/scenario:  "
        f"{N_NOMINAL}"
    )

    print(
        f"Perturbed/cell:    "
        f"{N_PERTURBED}"
    )

    print(
        f"Total jobs:        "
        f"{len(jobs)}"
    )

    print(
        f"Completed:         "
        f"{len(completed_ids)}"
    )

    print(
        f"Pending:           "
        f"{len(pending_jobs)}"
    )

    print(
        f"Limit:             "
        f"{args.limit}"
    )

    print("=" * 72)

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if args.dry_run:

        if args.limit is None:

            preview = pending_jobs

        else:

            preview = pending_jobs[
                :args.limit
            ]

        for job in preview[:20]:

            print(
                job["job_id"],
                job["scenario_id"],
                job["condition"],
                job[
                    "perturbation_family"
                ],
                job["lambda"],
                job["repetition"],
            )

        if len(preview) > 20:

            print(
                f"... "
                f"{len(preview) - 20} "
                f"more jobs"
            )

        return

    # --------------------------------------------------------
    # API credentials
    # --------------------------------------------------------

    validate_environment(
        args.provider
    )

    if args.provider == "qwen":
        initialise_qwen()

    # --------------------------------------------------------
    # Apply limit
    # --------------------------------------------------------

    if args.limit is None:

        selected_jobs = (
            pending_jobs
        )

    else:

        selected_jobs = (
            pending_jobs[
                :args.limit
            ]
        )

    if not selected_jobs:

        print(
            "No pending jobs."
        )

        return

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    successful = 0
    failed = 0

    for index, job in enumerate(
        selected_jobs,
        start=1,
    ):

        print(
            f"[{index}/"
            f"{len(selected_jobs)}] "
            f"{job['scenario_id']} "
            f"{job['condition']} "
            f"{job['perturbation_family']} "
            f"lambda={job['lambda']} "
            f"rep={job['repetition']}",
            flush=True,
        )

        record = execute_job(
            job
        )

        # Persist immediately.
        append_record(
            output_path,
            record,
        )

        if (
            record["status"]
            == "ok"
        ):

            successful += 1

            text_preview = (
                record.get(
                    "text",
                    ""
                )
                .replace(
                    "\n",
                    " "
                )
            )

            usage = record.get(
                "usage",
                {}
            )

            print(
                "  OK "
                f"({record['elapsed_seconds']:.2f}s, "
                f"{record['output_word_count']} words)"
            )

            print(
                "  Tokens: "
                f"in={usage.get('input_tokens')} "
                f"out={usage.get('output_tokens')} "
                f"reasoning="
                f"{usage.get('reasoning_tokens')}"
            )

            print(
                "  "
                + text_preview[:180]
            )

        else:

            failed += 1

            print(
                "  ERROR "
                f"{record['error_type']}: "
                f"{record['error_message']}",
                file=sys.stderr,
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RUN COMPLETED")
    print("=" * 72)

    print(
        f"New successful: "
        f"{successful}"
    )

    print(
        f"New failed:     "
        f"{failed}"
    )

    print(
        f"Output:         "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()

