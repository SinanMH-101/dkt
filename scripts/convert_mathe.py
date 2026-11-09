"""Build a DKT smoke-test dataset in MathE CSV order, not attempt chronology."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"Student ID", "Question ID", "Type of Answer"}


def convert(source, output, seed):
    sequences = defaultdict(list)
    # The release is a semicolon-separated Windows-encoded export.
    with source.open(encoding="cp1252", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if not FIELDS.issubset(reader.fieldnames or []):
            raise ValueError("MathE CSV must include Student ID, Question ID and Type of Answer")
        for row in reader:
            try:
                student = int(row["Student ID"])
                question = int(row["Question ID"])
                answer = int(row["Type of Answer"])
                if answer not in (0, 1):
                    raise ValueError("Type of Answer must be 0 or 1")
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid MathE row ending on line %d" % reader.line_num) from error
            sequences[student].append((question, answer))

    # Preserve repeated responses. The export does not identify individual attempts.
    students = sorted(student for student, rows in sequences.items() if len(rows) >= 2)
    if len(students) < 2:
        raise ValueError("Need at least two students with at least two responses each")
    shuffled = students[:]
    random.Random(seed).shuffle(shuffled)
    train = set(shuffled[: max(1, int(0.8 * len(students)))])
    test = set(students) - train
    for split in (train, test):
        if {answer for student in split for _, answer in sequences[student]} != {0, 1}:
            raise ValueError("Both splits need correct and incorrect responses for AUC")

    output.mkdir(parents=True, exist_ok=True)
    # Keep the numbering and split flags in the same sorted student order.
    with (output / "mathe.txt").open("w", encoding="ascii", newline="\n") as handle:
        for student_id, student in enumerate(students):
            for question, answer in sequences[student]:
                handle.write("%d %d %d\n" % (student_id, question, answer))
    (output / "mathe_split.txt").write_text(
        " ".join("1" if student in train else "0" for student in students), encoding="ascii"
    )
    lengths = [len(sequences[student]) for student in students]
    summary = {
        "source": source.name,
        "raw_rows": sum(map(len, sequences.values())),
        "raw_students": len(sequences),
        "excluded_single_response_students": len(sequences) - len(students),
        "students": len(students),
        "responses": sum(lengths),
        "questions": len({q for student in students for q, _ in sequences[student]}),
        "median_responses_per_student": median(lengths),
        "train_students": len(train),
        "test_students": len(test),
        "train_responses": sum(len(sequences[student]) for student in train),
        "test_responses": sum(len(sequences[student]) for student in test),
        "seed": seed,
        "ordering": "Source CSV row order within each student; question-grouped export, no timestamps",
        "scope": "Pipeline smoke test only; not an evaluation of chronological next-response prediction",
    }
    report = json.dumps(summary, indent=2)
    (output / "mathe_summary.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/mathe/MathE dataset (4).csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/mathe_file_order")
    parser.add_argument("--seed", type=int, default=8240)
    args = parser.parse_args()
    convert(args.input, args.output_dir, args.seed)
