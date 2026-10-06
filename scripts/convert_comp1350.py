"""Convert COMP1350 graded quiz steps to the legacy DKT text format.

Run with Python 3 and pandas/openpyxl installed. Defaults resolve relative to
the project, so this script can be invoked from any working directory.
"""

import sys

if sys.version_info[0] < 3:
    sys.exit(
        'This converter requires Python 3, while dkt-legacy uses Python 2.7.\n'
        'Create the Python 3 converter environment described in README.md.\n'
        'Run: conda run -n dkt-convert python scripts/convert_comp1350.py\n'
        'Then use dkt-legacy to run dkt.py on the generated text files.'
    )

import argparse
import json
import random
from pathlib import Path

try:
    import pandas as pd
    import openpyxl  # Required by pandas for .xlsx input.
except ImportError:
    sys.exit(
        'Missing converter dependencies. Run:\n'
        'conda run -n dkt-convert python -m pip install -r scripts/requirements-convert.txt'
    )


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["Participant-ID", "questionid", "timestamp", "state", "fraction"]


def convert(source, sheet, output, train_fraction, seed, min_length):
    raw = pd.read_excel(source, sheet_name=sheet)
    required = ["attemptstepid"] + FIELDS
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    # Each step has several metadata rows. Reject conflicting step records
    # rather than silently taking an arbitrary value from them.
    identified = raw.dropna(subset=["attemptstepid"])
    grouped = identified.groupby("attemptstepid", sort=False)
    conflicts = grouped[FIELDS].nunique(dropna=True).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError("Conflicting metadata in %d attempt steps" % conflicts.sum())
    steps = grouped[FIELDS].first().reset_index()
    graded = steps[steps.state.isin(["gradedright", "gradedwrong"])].copy()
    complete = graded.dropna(subset=required).copy()
    expected = complete.state.map({"gradedright": 1, "gradedwrong": 0})
    consistent = complete.fraction.eq(expected)
    clean = complete[consistent].copy()
    clean["correct"] = expected[consistent].astype(int)

    for field in ["questionid", "timestamp", "attemptstepid"]:
        numeric = pd.to_numeric(clean[field], errors="raise")
        if not numeric.mod(1).eq(0).all():
            raise ValueError("Non-integer values in " + field)
        clean[field] = numeric.astype("int64")
    # IDs are used only for grouping and deterministic anonymised numbering.
    clean["Participant-ID"] = clean["Participant-ID"].astype(str)
    lengths_before = clean.groupby("Participant-ID").size()
    eligible = lengths_before[lengths_before >= min_length].index
    clean = clean[clean["Participant-ID"].isin(eligible)].copy()
    students = sorted(clean["Participant-ID"].unique())
    if len(students) < 2:
        raise ValueError("Need at least two eligible students for train/test splitting")
    student_ids = {student: index for index, student in enumerate(students)}
    clean["student_id"] = clean["Participant-ID"].map(student_ids)
    clean = clean.sort_values(["student_id", "timestamp", "attemptstepid"])

    shuffled = list(range(len(students)))
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, min(len(students) - 1, int(len(students) * train_fraction)))
    training = set(shuffled[:train_count])
    assignments = ["1" if i in training else "0" for i in range(len(students))]
    train = clean[clean.student_id.isin(training)]
    test = clean[~clean.student_id.isin(training)]
    if train.correct.nunique() < 2 or test.correct.nunique() < 2:
        raise ValueError("Both splits need correct and incorrect outcomes for AUC")

    output.mkdir(parents=True, exist_ok=True)
    dataset = output / "comp1350.txt"
    split = output / "comp1350_split.txt"
    clean[["student_id", "questionid", "correct"]].to_csv(
        dataset, sep=" ", header=False, index=False, lineterminator="\n"
    )
    # The legacy parser does not strip whitespace from split assignments.
    split.write_text(" ".join(assignments), encoding="ascii")
    lengths = clean.groupby("student_id").size()
    summary = {
        "source": source.name,
        "sheet": sheet,
        "raw_rows": len(raw),
        "rows_missing_step_id": int(raw.attemptstepid.isna().sum()),
        "distinct_steps": len(steps),
        "excluded_ungraded_steps": len(steps) - len(graded),
        "excluded_graded_steps_missing_required_fields": len(graded) - len(complete),
        "excluded_conflicting_grades": int((~consistent).sum()),
        "excluded_students_below_min_length": len(lengths_before) - len(students),
        "excluded_interactions_below_min_length": int(lengths_before[lengths_before < min_length].sum()),
        "interactions": len(clean),
        "students": len(students),
        "question_ids": int(clean.questionid.nunique()),
        "sequence_lengths": {"min": int(lengths.min()), "median": float(lengths.median()), "max": int(lengths.max())},
        "train_students": train_count,
        "test_students": len(students) - train_count,
        "train_interactions": len(train),
        "test_interactions": len(test),
        "correct_interactions": int(clean.correct.sum()),
        "test_only_question_ids": sorted(set(test.questionid) - set(train.questionid)),
        "timestamp_ties_after_first": int(clean.duplicated(["student_id", "timestamp"]).sum()),
        "ordering": "timestamp then attemptstepid; step ID is a deterministic tie-breaker, not verified chronology within timestamp ties",
        "exercise_representation": "questionid (question variants kept separate)",
        "seed": seed,
        "train_fraction": train_fraction,
        "min_length": min_length,
    }
    (output / "comp1350_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Dataset: " + str(dataset))
    print("Split: " + str(split))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "Sinan_Dataset.xlsx")
    parser.add_argument("--sheet", default="Sinan_Data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "comp1350")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=8240)
    parser.add_argument("--min-length", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be between 0 and 1")
    if args.min_length < 2:
        parser.error("--min-length must be at least 2 for sequential prediction")
    convert(args.input, args.sheet, args.output_dir, args.train_fraction, args.seed, args.min_length)


if __name__ == "__main__":
    main()
