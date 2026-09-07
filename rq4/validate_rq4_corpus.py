#!/usr/bin/env python3

"""
RQ4 corpus validation and human-review preparation.

This script performs structural and lexical diagnostics on the
controlled prompt-perturbation corpus.

It does NOT attempt to determine semantic equivalence automatically.
Semantic substitution, information removal, propositional preservation,
coherence, and task preservation require human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_FAMILIES = {
    "semantic",
    "context_removal",
    "surface",
}

EXPECTED_LEVELS = [
    0.25,
    0.50,
    0.75,
    1.00,
]

EXPECTED_TASK_FAMILIES = {
    "contextual_reasoning",
    "comparison",
    "explanation",
    "summarisation_interpretation",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_set(text: str) -> set[str]:
    return set(tokens(text))


def jaccard(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)

    if not sa and not sb:
        return 1.0

    return len(sa & sb) / len(sa | sb)


def token_retention(base: str, transformed: str) -> float:
    """
    Fraction of unique base tokens also appearing in transformation.
    Diagnostic only.
    """
    base_tokens = token_set(base)

    if not base_tokens:
        return 1.0

    return len(base_tokens & token_set(transformed)) / len(base_tokens)


def length_ratio(base: str, transformed: str) -> float:
    n_base = max(len(tokens(base)), 1)
    return len(tokens(transformed)) / n_base


def add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    scenario_id: str,
    family: str,
    lam: float | str,
    message: str,
) -> None:

    issues.append(
        {
            "severity": severity,
            "scenario_id": scenario_id,
            "family": family,
            "lambda": lam,
            "message": message,
        }
    )


def expected_metadata(family: str) -> dict[str, bool]:

    if family == "semantic":
        return {
            "semantic_change_expected": True,
            "semantic_information_reduced": False,
            "propositional_content_preserved": False,
        }

    if family == "context_removal":
        return {
            "semantic_change_expected": False,
            "semantic_information_reduced": True,
            "propositional_content_preserved": False,
        }

    if family == "surface":
        return {
            "semantic_change_expected": False,
            "semantic_information_reduced": False,
            "propositional_content_preserved": True,
        }

    raise ValueError(f"Unknown family: {family}")


def validate_metadata(
    item: dict[str, Any],
    family: str,
    scenario_id: str,
    lam: float,
    issues: list[dict[str, Any]],
) -> None:

    expected = expected_metadata(family)

    for field, expected_value in expected.items():

        if field not in item:
            add_issue(
                issues,
                "ERROR",
                scenario_id,
                family,
                lam,
                f"Missing metadata field '{field}'.",
            )
            continue

        if item[field] is not expected_value:
            add_issue(
                issues,
                "ERROR",
                scenario_id,
                family,
                lam,
                f"{field}={item[field]!r}, expected {expected_value!r}.",
            )

    changed = item.get("changed_facts", [])
    removed = item.get("removed_facts", [])

    if not isinstance(changed, list):
        add_issue(
            issues, "ERROR", scenario_id, family, lam,
            "changed_facts must be a list.",
        )
        changed = []

    if not isinstance(removed, list):
        add_issue(
            issues, "ERROR", scenario_id, family, lam,
            "removed_facts must be a list.",
        )
        removed = []

    if family == "semantic":
        if len(changed) == 0:
            add_issue(
                issues, "ERROR", scenario_id, family, lam,
                "Semantic transformation declares no changed facts.",
            )

        if len(removed) > 0:
            add_issue(
                issues, "WARNING", scenario_id, family, lam,
                "Semantic transformation also removes facts. "
                "Review whether this mixes perturbation families.",
            )

    elif family == "context_removal":
        if len(changed) > 0:
            add_issue(
                issues, "ERROR", scenario_id, family, lam,
                "Context-removal transformation declares changed facts.",
            )

        if len(removed) == 0:
            add_issue(
                issues, "ERROR", scenario_id, family, lam,
                "Context-removal transformation declares no removed facts.",
            )

    elif family == "surface":
        if changed:
            add_issue(
                issues, "ERROR", scenario_id, family, lam,
                "Surface transformation declares changed facts.",
            )

        if removed:
            add_issue(
                issues, "ERROR", scenario_id, family, lam,
                "Surface transformation declares removed facts.",
            )


def validate_scenario(
    scenario: dict[str, Any],
    issues: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    global_prompts: Counter,
) -> None:

    scenario_id = scenario.get("scenario_id", "")

    if not scenario_id:
        add_issue(
            issues, "ERROR", "", "", "",
            "Scenario without scenario_id.",
        )
        return

    task_family = scenario.get("task_family")

    if task_family not in EXPECTED_TASK_FAMILIES:
        add_issue(
            issues,
            "WARNING",
            scenario_id,
            "",
            "",
            f"Unexpected task_family '{task_family}'.",
        )

    base = scenario.get("base")

    if not isinstance(base, str) or not base.strip():
        add_issue(
            issues, "ERROR", scenario_id, "", "",
            "Missing or empty base prompt.",
        )
        return

    global_prompts[normalise_text(base)] += 1

    transformations = scenario.get("transformations")

    if not isinstance(transformations, dict):
        add_issue(
            issues, "ERROR", scenario_id, "", "",
            "Missing transformations object.",
        )
        return

    actual_families = set(transformations.keys())

    if actual_families != EXPECTED_FAMILIES:
        add_issue(
            issues,
            "ERROR",
            scenario_id,
            "",
            "",
            "Transformation families are "
            f"{sorted(actual_families)}, expected "
            f"{sorted(EXPECTED_FAMILIES)}.",
        )

    for family in EXPECTED_FAMILIES:

        family_items = transformations.get(family, [])

        if not isinstance(family_items, list):
            add_issue(
                issues,
                "ERROR",
                scenario_id,
                family,
                "",
                "Transformation family must be a list.",
            )
            continue

        levels = []

        previous_length = None
        previous_removed_count = None

        for item in family_items:

            try:
                lam = float(item["lambda"])
            except (KeyError, TypeError, ValueError):
                add_issue(
                    issues,
                    "ERROR",
                    scenario_id,
                    family,
                    "",
                    "Invalid or missing lambda.",
                )
                continue

            levels.append(lam)

            prompt = item.get("prompt")

            if not isinstance(prompt, str) or not prompt.strip():
                add_issue(
                    issues,
                    "ERROR",
                    scenario_id,
                    family,
                    lam,
                    "Missing or empty transformed prompt.",
                )
                continue

            global_prompts[normalise_text(prompt)] += 1

            if normalise_text(prompt) == normalise_text(base):
                add_issue(
                    issues,
                    "ERROR",
                    scenario_id,
                    family,
                    lam,
                    "Transformation is identical to base prompt.",
                )

            validate_metadata(
                item,
                family,
                scenario_id,
                lam,
                issues,
            )

            n_base = len(tokens(base))
            n_prompt = len(tokens(prompt))
            jac = jaccard(base, prompt)
            retention = token_retention(base, prompt)
            ratio = length_ratio(base, prompt)

            diagnostics.append(
                {
                    "scenario_id": scenario_id,
                    "task_family": task_family,
                    "family": family,
                    "lambda": lam,
                    "base_tokens": n_base,
                    "prompt_tokens": n_prompt,
                    "length_ratio": ratio,
                    "token_jaccard": jac,
                    "base_token_retention": retention,
                    "changed_fact_count":
                        len(item.get("changed_facts", [])),
                    "removed_fact_count":
                        len(item.get("removed_facts", [])),
                }
            )

            # Human-review sheet.
            review_rows.append(
                {
                    "scenario_id": scenario_id,
                    "task_family": task_family,
                    "family": family,
                    "lambda": lam,
                    "base_prompt": base,
                    "transformed_prompt": prompt,
                    "declared_changed_facts":
                        " | ".join(item.get("changed_facts", [])),
                    "declared_removed_facts":
                        " | ".join(item.get("removed_facts", [])),
                    "expected_meaning_changed":
                        family == "semantic",
                    "expected_information_removed":
                        family == "context_removal",
                    "expected_propositional_content_preserved":
                        family == "surface",
                    "reviewer_meaning_changed": "",
                    "reviewer_information_removed": "",
                    "reviewer_propositional_content_preserved": "",
                    "reviewer_prompt_coherent": "",
                    "reviewer_task_preserved": "",
                    "reviewer_notes": "",
                }
            )

            # These are diagnostics, not hard semantic criteria.
            if family == "surface":
                if ratio < 0.60 or ratio > 1.60:
                    add_issue(
                        issues,
                        "REVIEW",
                        scenario_id,
                        family,
                        lam,
                        f"Surface length ratio is {ratio:.2f}; "
                        "manual review recommended.",
                    )

            if family == "context_removal":
                removed_count = len(item.get("removed_facts", []))

                if (
                    previous_removed_count is not None
                    and removed_count < previous_removed_count
                ):
                    add_issue(
                        issues,
                        "REVIEW",
                        scenario_id,
                        family,
                        lam,
                        "Declared removed-fact count decreases as "
                        "lambda increases.",
                    )

                previous_removed_count = removed_count

                if (
                    previous_length is not None
                    and n_prompt > previous_length + 5
                ):
                    add_issue(
                        issues,
                        "REVIEW",
                        scenario_id,
                        family,
                        lam,
                        "Context-removal prompt becomes substantially "
                        "longer at a higher lambda.",
                    )

                previous_length = n_prompt

        if sorted(levels) != EXPECTED_LEVELS:
            add_issue(
                issues,
                "ERROR",
                scenario_id,
                family,
                "",
                f"Levels are {sorted(levels)}, expected {EXPECTED_LEVELS}.",
            )


def update_metadata_fields(data: dict[str, Any]) -> None:
    """
    Add semantic_information_reduced to the pilot JSON if missing.

    Existing declarations are not otherwise rewritten.
    """

    for scenario in data.get("scenarios", []):

        transformations = scenario.get("transformations", {})

        for family, items in transformations.items():

            expected = expected_metadata(family)

            for item in items:
                item.setdefault(
                    "semantic_information_reduced",
                    expected["semantic_information_reduced"],
                )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_json",
        help="RQ4 corpus JSON",
    )

    parser.add_argument(
        "--output-dir",
        default="validation",
    )

    parser.add_argument(
        "--write-normalised-json",
        action="store_true",
        help=(
            "Write a copy with the new "
            "semantic_information_reduced metadata field."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    update_metadata_fields(data)

    issues: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    global_prompts: Counter = Counter()

    scenarios = data.get("scenarios", [])

    if len(scenarios) != 20:
        add_issue(
            issues,
            "REVIEW",
            "",
            "",
            "",
            f"Pilot contains {len(scenarios)} scenarios; expected 4.",
        )

    scenario_ids = [
        s.get("scenario_id")
        for s in scenarios
    ]

    duplicate_ids = [
        sid
        for sid, count in Counter(scenario_ids).items()
        if sid and count > 1
    ]

    for sid in duplicate_ids:
        add_issue(
            issues,
            "ERROR",
            sid,
            "",
            "",
            "Duplicate scenario_id.",
        )

    for scenario in scenarios:
        validate_scenario(
            scenario,
            issues,
            diagnostics,
            review_rows,
            global_prompts,
        )

    # Global accidental duplicate prompts.
    duplicate_prompt_count = 0

    for prompt, count in global_prompts.items():

        if count > 1:
            duplicate_prompt_count += 1

            add_issue(
                issues,
                "REVIEW",
                "",
                "",
                "",
                "A normalised prompt occurs "
                f"{count} times: {prompt[:120]}",
            )

    write_csv(
        output_dir / "rq4_validation_issues.csv",
        issues,
    )

    write_csv(
        output_dir / "rq4_lexical_diagnostics.csv",
        diagnostics,
    )

    write_csv(
        output_dir / "rq4_human_review.csv",
        review_rows,
    )

    if args.write_normalised_json:

        with (
            output_dir
            / "rq4_prompts_pilot_normalised.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

    severity_counts = Counter(
        issue["severity"]
        for issue in issues
    )

    print("=" * 68)
    print("RQ4 PILOT CORPUS VALIDATION")
    print("=" * 68)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Transformations: {len(diagnostics)}")
    print(f"Human-review rows: {len(review_rows)}")
    print(f"Errors: {severity_counts['ERROR']}")
    print(f"Warnings: {severity_counts['WARNING']}")
    print(f"Manual-review flags: {severity_counts['REVIEW']}")
    print(f"Duplicate prompt groups: {duplicate_prompt_count}")

    print()
    print("Outputs:")
    print(output_dir / "rq4_validation_issues.csv")
    print(output_dir / "rq4_lexical_diagnostics.csv")
    print(output_dir / "rq4_human_review.csv")

    if args.write_normalised_json:
        print(
            output_dir
            / "rq4_prompts_pilot_normalised.json"
        )

    if severity_counts["ERROR"] > 0:
        raise SystemExit(
            "\nStructural validation failed. "
            "Resolve ERROR rows before using the corpus."
        )

    print(
        "\nStructural validation passed. "
        "Human semantic review is still required."
    )


if __name__ == "__main__":
    main()
