#!/usr/bin/env python3
"""Build a self-consistent Text-to-SQL prediction from repeated model runs."""

import argparse
import json
import os
import sqlite3
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


BIRD_SEPARATOR = "\t----- bird -----\t"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select each question's SQL by strict raw-string majority first, "
            "then by SQLite execution-result consensus."
        )
    )
    parser.add_argument("--model", help='Model name contained in filenames, e.g. "Kwai-AutoSQL-14B"')
    parser.add_argument(
        "--results-dir",
        default="./llm/exp_result/tokenhub_output_kg",
        help="Directory containing repeated prediction JSON files",
    )
    parser.add_argument(
        "--db-root",
        default="../koenshen_bird_evaluate/data_mini_dev/dev_databases",
        help="Directory containing <db_id>/<db_id>.sqlite",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-query timeout in seconds")
    parser.add_argument("--output", help="Consensus prediction JSON path")
    parser.add_argument("--report", help="Detailed analysis JSON path")
    return parser.parse_args()


def load_runs(results_dir, model):
    results_dir = Path(results_dir)
    files = sorted(
        path
        for path in results_dir.glob(f"*{model}*.json")
        if "_self_consistency_" not in path.name
    )
    if not files:
        raise FileNotFoundError(f"No JSON files matching model {model!r} in {results_dir}")

    runs = []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        runs.append((path, data))

    expected_keys = set(runs[0][1])
    for path, data in runs[1:]:
        if set(data) != expected_keys:
            missing = sorted(expected_keys - set(data))
            extra = sorted(set(data) - expected_keys)
            raise ValueError(f"Question IDs differ in {path}: missing={missing[:5]}, extra={extra[:5]}")
    return runs


def split_prediction(prediction):
    if not isinstance(prediction, str):
        return "", None
    if BIRD_SEPARATOR not in prediction:
        return prediction, None
    return prediction.rsplit(BIRD_SEPARATOR, 1)


def execute_sql(sql, db_path, timeout):
    if not sql.strip():
        return None, "empty SQL"

    started = time.monotonic()
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")

        def abort_if_timed_out():
            return 1 if time.monotonic() - started > timeout else 0

        connection.set_progress_handler(abort_if_timed_out, 10_000)
        rows = connection.execute(sql).fetchall()
        # This mirrors the repository's official EX metric, which compares sets
        # and therefore ignores row order and duplicate rows.
        result_key = frozenset(rows)
        return result_key, None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"
    finally:
        connection.close()


def choose_prediction(question_id, candidates, db_root, timeout):
    raw_counts = Counter(candidates)
    top_raw, top_raw_votes = raw_counts.most_common(1)[0]
    strict_majority = len(candidates) // 2 + 1

    base_report = {
        "question_id": question_id,
        "num_candidates": len(candidates),
        "num_raw_variants": len(raw_counts),
        "top_raw_votes": top_raw_votes,
        "strict_majority_required": strict_majority,
    }
    if top_raw_votes >= strict_majority:
        base_report.update({"method": "raw_string_majority", "selected_votes": top_raw_votes})
        return top_raw, base_report

    execution_groups = []
    candidate_details = []
    for raw_prediction, raw_votes in raw_counts.most_common():
        sql, db_id = split_prediction(raw_prediction)
        if not db_id:
            error = "missing BIRD database separator or database ID"
            result_key = None
        else:
            db_path = Path(db_root) / db_id / f"{db_id}.sqlite"
            if not db_path.is_file():
                result_key, error = None, f"database not found: {db_path}"
            else:
                result_key, error = execute_sql(sql, db_path, timeout)

        candidate_details.append(
            {
                "raw_prediction": raw_prediction,
                "raw_votes": raw_votes,
                "execution_ok": error is None,
                "execution_error": error,
            }
        )
        if error is not None:
            continue

        for group in execution_groups:
            if group["result_key"] == result_key:
                group["votes"] += raw_votes
                group["members"].append(raw_prediction)
                break
        else:
            execution_groups.append(
                {"result_key": result_key, "votes": raw_votes, "members": [raw_prediction]}
            )

    if execution_groups:
        # Stable sorting preserves the first candidate encountered when tied.
        winner = sorted(execution_groups, key=lambda group: group["votes"], reverse=True)[0]
        selected = winner["members"][0]
        method = "execution_consensus"
        selected_votes = winner["votes"]
    else:
        selected = top_raw
        method = "raw_plurality_fallback_all_execution_failed"
        selected_votes = top_raw_votes

    base_report.update(
        {
            "method": method,
            "selected_votes": selected_votes,
            "num_execution_result_variants": len(execution_groups),
            "candidates": candidate_details,
        }
    )
    return selected, base_report


def main():
    args = parse_args()
    runs = load_runs(args.results_dir, args.model)
    question_ids = sorted(runs[0][1], key=lambda value: int(value) if value.isdigit() else value)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_model = args.model.replace("/", "_")
    output = Path(args.output) if args.output else Path(args.results_dir) / (
        f"predict_mini_dev_{safe_model}_self_consistency_{timestamp}_pid{os.getpid()}.json"
    )
    report = Path(args.report) if args.report else output.with_name(output.stem + "_report.json")

    selected_predictions = {}
    question_reports = []
    for position, question_id in enumerate(question_ids, start=1):
        candidates = [data[question_id] for _, data in runs]
        selected, analysis = choose_prediction(question_id, candidates, args.db_root, args.timeout)
        selected_predictions[question_id] = selected
        question_reports.append(analysis)
        if position % 25 == 0 or position == len(question_ids):
            print(f"Processed {position}/{len(question_ids)} questions")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(selected_predictions, handle, ensure_ascii=False, indent=4)

    summary = Counter(item["method"] for item in question_reports)
    report_payload = {
        "model": args.model,
        "source_files": [str(path) for path, _ in runs],
        "num_runs": len(runs),
        "num_questions": len(question_ids),
        "method_counts": dict(summary),
        "output_file": str(output),
        "questions": question_reports,
    }
    with report.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, ensure_ascii=False, indent=2)

    print(f"Consensus output: {output}")
    print(f"Analysis report: {report}")
    print(f"Selection methods: {dict(summary)}")


if __name__ == "__main__":
    # python llm/src/self_consistency.py --model Kwai-AutoSQL-14B
    main()
