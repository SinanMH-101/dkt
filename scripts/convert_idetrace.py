"""Convert one IDETrace deployment into run-level DKT interactions.

Python 3 standard library only. One TEST_CASE_RESULT event becomes one
interaction: correct means all tests passed. Individual tests within the same
execution are not serialized into artificial sequential observations.
"""

import sys

if sys.version_info[0] < 3:
    sys.exit("Use Python 3: conda run -n dkt-convert python scripts/convert_idetrace.py --help")

import argparse
import collections
import csv
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_run(event):
    """Validate a test result and label the execution as a whole."""
    timestamp = event.get("timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ValueError("missing_or_invalid_timestamp")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid_payload")
    details = payload.get("detailed_results")
    total = payload.get("total_tests")
    passed_count = payload.get("passed_count")
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError("invalid_test_count")
    if not isinstance(details, list) or len(details) != total:
        raise ValueError("missing_or_incomplete_test_results")
    if not all(isinstance(item, dict) and type(item.get("passed")) is bool for item in details):
        raise ValueError("invalid_pass_labels")
    count = sum(item["passed"] for item in details)
    if type(passed_count) is not int or count != passed_count:
        raise ValueError("inconsistent_passed_count")
    rate = payload.get("passrate")
    expected_rate = 100.0 * count / total
    # The telemetry can round the reported pass percentage.
    if (isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or abs(rate - expected_rate) > 0.51):
        raise ValueError("inconsistent_passrate")
    return {"timestamp": timestamp, "session_id": event.get("sessionId"),
            "passed_tests": count, "total_tests": total, "correct": int(count == total)}


def convert(args):
    with args.input.open(encoding="utf-8") as source:
        raw = json.load(source)
    if not isinstance(raw, dict):
        raise ValueError("Expected a JSON object keyed by participant ID")
    exclusions = collections.Counter()
    sequences = []
    test_events = 0
    eligible_runs = 0
    for participant in sorted(raw):
        entry = raw[participant]
        if not isinstance(entry, dict) or not isinstance(entry.get("events"), list):
            raise ValueError("Expected an events list for each participant")
        runs = []
        for position, event in enumerate(entry["events"]):
            if event.get("type") != "TEST_CASE_RESULT":
                continue
            test_events += 1
            try:
                run = parse_run(event)
            except ValueError as error:
                exclusions[str(error)] += 1
                continue
            run["source_event_index"] = position
            runs.append(run)
        # Preserve actual repeated executions, even when their outcomes match.
        runs.sort(key=lambda run: (run["timestamp"], run["source_event_index"]))
        eligible_runs += len(runs)
        if len(runs) < args.min_length:
            exclusions["participants_below_min_length"] += 1
            exclusions["runs_below_min_length"] += len(runs)
            continue
        sequences.append(runs)
    if len(sequences) < 2:
        raise ValueError("Need at least two participants with sufficient valid executions")
    ids = list(range(len(sequences)))
    random.Random(args.seed).shuffle(ids)
    n_train = max(1, min(len(ids) - 1, int(len(ids) * args.train_fraction)))
    training = set(ids[:n_train])
    outcomes = {"train": set(), "test": set()}
    for student, runs in enumerate(sequences):
        outcomes["train" if student in training else "test"].update(run["correct"] for run in runs)
    if any(len(values) != 2 for values in outcomes.values()):
        raise ValueError("Both splits need passing and failing executions for AUC; change split or dataset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.output_dir / "idetrace.txt"
    audit = args.output_dir / "idetrace_runs.csv"
    # Keep execution details in the audit file. DKT gets one exercise ID for this task.
    with dataset.open("w", encoding="ascii", newline="") as data, audit.open("w", encoding="utf-8", newline="") as details:
        writer = csv.writer(details)
        writer.writerow(["student_id", "exercise_id", "task", "step", "timestamp_ms", "session_id", "source_event_index", "passed_tests", "total_tests", "correct"])
        for student, runs in enumerate(sequences):
            for step, run in enumerate(runs):
                data.write("%d 0 %d\n" % (student, run["correct"]))
                writer.writerow([student, 0, args.task, step, run["timestamp"], run["session_id"], run["source_event_index"], run["passed_tests"], run["total_tests"], run["correct"]])
    (args.output_dir / "idetrace_split.txt").write_text(
        " ".join("1" if student in training else "0" for student in range(len(sequences))), encoding="ascii")
    lengths = sorted(len(runs) for runs in sequences)
    middle = len(lengths) // 2
    median = lengths[middle] if len(lengths) % 2 else (lengths[middle - 1] + lengths[middle]) / 2.0
    correct = sum(run["correct"] for runs in sequences for run in runs)
    interactions = sum(lengths)
    summary = {
        "source": args.input.name, "task": args.task,
        "representation": "One interaction per test execution; exercise_id=0 for this deployment task; correct iff every test passes",
        "input_participants": len(raw), "test_result_events": test_events,
        "valid_runs_before_length_filter": eligible_runs, "exclusions": dict(exclusions),
        "students": len(sequences), "interactions": interactions,
        "exercises": 1, "correct_runs": correct, "incorrect_runs": interactions - correct,
        "train_students": n_train, "test_students": len(sequences) - n_train,
        "train_interactions": sum(len(runs) for student, runs in enumerate(sequences) if student in training),
        "test_interactions": sum(len(runs) for student, runs in enumerate(sequences) if student not in training),
        "sequence_lengths": {"min": min(lengths), "median": median, "max": max(lengths)},
        "same_timestamp_runs_after_first": sum(len(runs) - len({run["timestamp"] for run in runs}) for runs in sequences),
        "multi_session_students": sum(len({run["session_id"] for run in runs}) > 1 for runs in sequences),
        "ordering": "timestamp, then source event index for ties; ties do not prove within-timestamp chronology",
        "limitations": ["Predicts repeated attempts on one programming task, not cross-concept learning", "Task label is supplied by the user; TASK_SELECT is text selection, not reliable task switching", "No original participant IDs, names, code or chat text are exported"],
        "seed": args.seed, "train_fraction": args.train_fraction, "min_length": args.min_length,
    }
    (args.output_dir / "idetrace_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Output: " + str(args.output_dir.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/tutortrace/deployment_1.json")
    parser.add_argument("--task", default="GradeBook", help="Verified task label for this single-task deployment")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/idetrace")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=8240)
    parser.add_argument("--min-length", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1 or args.min_length < 2:
        parser.error("Require 0 < train-fraction < 1 and min-length >= 2")
    convert(args)


if __name__ == "__main__":
    main()
