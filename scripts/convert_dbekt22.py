"""Convert DBE-KT22 Transaction.csv (or its official ZIP) to legacy DKT."""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/dbekt22/2_DBE_KT22_datafiles_100102_csv.zip"
OUTPUT = ROOT / "data/dbekt22"
LABELS = {"true": 1, "false": 0, "1": 1, "0": 0}
MISSING = {"", "none", "null", "nan", "na", "n/a"}


@contextmanager
def transaction_file(source):
    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            names = [name for name in archive.namelist()
                     if PurePosixPath(name).name.lower() in {"transaction.csv", "dbe_transaction.csv"}
                     and not name.startswith("__MACOSX/")]
            if len(names) != 1:
                raise ValueError("Expected exactly one Transaction.csv in the DBE-KT22 ZIP")
            with archive.open(names[0]) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as handle:
                yield handle
    else:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            yield handle


def parse_time(value):
    # The archive uses a space before the numeric offset, e.g. "...722 -0700".
    value = re.sub(r"\s+([+-]\d{2}:?\d{2}|Z)$", r"\1", value)
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
    return stamp


def convert(source, output, seed):
    attempts = defaultdict(list)
    seen = {}
    counts = Counter()
    with transaction_file(source) as handle:
        reader = csv.DictReader(handle)
        required = {"id", "student_id", "question_id", "start_time", "answer_state"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing DBE-KT22 transaction columns: " + ", ".join(sorted(missing)))
        for row in reader:
            counts["source_rows"] += 1
            state = (row["answer_state"] or "").strip().lower()
            if state in MISSING:
                counts["excluded_ungraded"] += 1
                continue
            if state not in LABELS:
                raise ValueError("Unrecognised answer_state at CSV line %d: %r" % (reader.line_num, state))
            if any((row[key] or "").strip().lower() in MISSING for key in required - {"answer_state"}):
                counts["excluded_missing_id_or_time"] += 1
                continue
            try:
                student, question, transaction = (int(row[key]) for key in ("student_id", "question_id", "id"))
                stamp = parse_time(row["start_time"].strip())
            except ValueError as error:
                raise ValueError("Invalid transaction ID or timestamp at CSV line %d" % reader.line_num) from error
            record = (student, question, stamp, LABELS[state])
            if transaction in seen:
                if seen[transaction] != record:
                    raise ValueError("Conflicting rows for transaction %d" % transaction)
                counts["excluded_duplicate_transaction"] += 1
                continue
            seen[transaction] = record
            attempts[student].append((stamp, transaction, question, LABELS[state]))

    students = sorted(student for student, seq in attempts.items() if len(seq) >= 2)
    if len(students) < 2:
        raise ValueError("Need at least two students with at least two graded attempts each")
    for student in students:
        seq = attempts[student]
        seq.sort()
        counts["tied_starts_after_first"] += sum(a[0] == b[0] for a, b in zip(seq, seq[1:]))
    shuffled = students[:]
    random.Random(seed).shuffle(shuffled)
    train = set(shuffled[: int(0.8 * len(students))])
    test = set(students) - train
    for group in (train, test):
        if {row[3] for student in group for row in attempts[student]} != {0, 1}:
            raise ValueError("Both splits need correct and incorrect responses for AUC; try a different seed")

    output.mkdir(parents=True, exist_ok=True)
    with (output / "dbekt22.txt").open("w", encoding="ascii", newline="\n") as handle:
        for student_id, student in enumerate(students):
            for _, _, question, correct in attempts[student]:
                handle.write("%d %d %d\n" % (student_id, question, correct))
    (output / "dbekt22_split.txt").write_text(
        " ".join("1" if student in train else "0" for student in students), encoding="ascii"
    )
    summary = {
        "source": source.name,
        **counts,
        "excluded_students_with_one_graded_attempt": len(attempts) - len(students),
        "students": len(students),
        "attempts": sum(len(attempts[student]) for student in students),
        "questions": len({row[2] for student in students for row in attempts[student]}),
        "train_students": len(train),
        "test_students": len(test),
        "train_attempts": sum(len(attempts[student]) for student in train),
        "test_attempts": sum(len(attempts[student]) for student in test),
        "seed": seed,
        "ordering": "start_time, then transaction ID for tied starts; ties do not prove chronology",
    }
    report = json.dumps(summary, indent=2)
    (output / "dbekt22_summary.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Official CSV archive or extracted Transaction.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=8240)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Input file not found: %s. Download the CSV archive from https://doi.org/10.26193/6DZWOH." % args.input)
    convert(args.input, args.output_dir, args.seed)
