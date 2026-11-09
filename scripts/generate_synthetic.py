"""Generate synthetic learning sequences using only the Python 3 standard library.

Questions share latent concepts. Correctness follows a guessing-adjusted logistic
ability/difficulty model. Practising a concept increases its ability; optional
forgetting decreases ability between interactions. Noise flips sampled responses.
This is a constructed experiment, not a reproduction of the paper's simulator.
"""

import sys

if sys.version_info[0] < 3:
    sys.exit("Use Python 3: conda run -n dkt-convert python scripts/generate_synthetic.py")

import argparse
import csv
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def probability(ability, difficulty, guessing):
    margin = ability - difficulty
    # Keep the exponent nonpositive so extreme ability scores do not overflow.
    if margin >= 0:
        logistic = 1 / (1 + math.exp(-margin))
    else:
        exp_margin = math.exp(margin)
        logistic = exp_margin / (1 + exp_margin)
    return guessing + (1 - guessing) * logistic


def generate(args):
    # Separate streams keep the clean and noisy variants identical in all
    # respects except response flips when their other parameters match.
    structure = random.Random(args.seed)
    responses = random.Random(args.seed + 1)
    noise = random.Random(args.seed + 2)
    split_rng = random.Random(args.seed + 3)
    questions = [
        {"question_id": c * args.questions_per_concept + q,
         "concept_id": c, "difficulty": structure.gauss(0, args.difficulty_sd)}
        for c in range(args.concepts) for q in range(args.questions_per_concept)
    ]
    students = list(range(args.students))
    split_rng.shuffle(students)
    n_train = max(1, min(args.students - 1, int(args.students * args.train_fraction)))
    train_ids = set(students[:n_train])
    assignments = ["1" if s in train_ids else "0" for s in range(args.students)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "synthetic.txt"
    truth_path = args.output_dir / "synthetic_ground_truth.csv"
    correct_total = 0
    flipped_total = 0
    split_outcomes = {"train": set(), "test": set()}
    student_parameters = []
    with data_path.open("w", encoding="ascii", newline="") as data, truth_path.open("w", encoding="ascii", newline="") as truth:
        writer = csv.writer(truth)
        writer.writerow(["student_id", "step", "question_id", "concept_id", "ability_before", "learning_rate", "difficulty", "probability_before_noise", "probability_observed", "response_before_noise", "noise_flipped", "correct"])
        for student in range(args.students):
            abilities = [structure.gauss(args.initial_ability, args.ability_sd) for _ in range(args.concepts)]
            learning_rate = structure.uniform(args.learning_rate * 0.5, args.learning_rate * 1.5)
            student_parameters.append({
                "student_id": student,
                "initial_abilities": abilities[:],
                "learning_rate": learning_rate,
            })
            for step in range(args.sequence_length):
                if step:
                    abilities = [a - args.forgetting for a in abilities]
                question = structure.choice(questions)
                concept = question["concept_id"]
                ability = abilities[concept]
                correct_probability = probability(ability, question["difficulty"], args.guessing)
                latent_response = int(responses.random() < correct_probability)
                flipped = int(noise.random() < args.noise)
                correct = latent_response ^ flipped
                # Either a correct response survives the flip, or an incorrect one flips.
                observed_probability = args.noise + (1 - 2 * args.noise) * correct_probability
                data.write("%d %d %d\n" % (student, question["question_id"], correct))
                writer.writerow([
                    student, step, question["question_id"], concept, ability, learning_rate,
                    question["difficulty"], correct_probability, observed_probability,
                    latent_response, flipped, correct,
                ])
                correct_total += correct
                flipped_total += flipped
                split_outcomes["train" if student in train_ids else "test"].add(correct)
                abilities[concept] += learning_rate
    (args.output_dir / "synthetic_split.txt").write_text(" ".join(assignments), encoding="ascii")
    config = {key: value for key, value in vars(args).items() if key != "output_dir"}
    metadata = {
        "parameters": config,
        "model": "P(correct before noise) = guessing + (1-guessing)*sigmoid(ability-difficulty)",
        "learning": "Student-specific rate sampled uniformly from 0.5 to 1.5 times learning_rate; ability increases after every practice of that concept, independently of outcome",
        "forgetting": "Subtract forgetting from every concept before each interaction after the first",
        "selection": "Uniform random question selection, with replacement",
        "noise": "Flip each sampled response independently with probability noise",
        "ground_truth_use": "Ground truth is for analysis only; DKT receives question IDs and binary responses, without concept labels or abilities",
        "questions": questions,
        "student_parameters": student_parameters,
        "summary": {"students": args.students, "interactions": args.students * args.sequence_length,
                    "questions": len(questions), "train_students": n_train,
                    "test_students": args.students - n_train,
                    "correct_responses": correct_total, "flipped_responses": flipped_total},
    }
    (args.output_dir / "synthetic_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["summary"], indent=2))
    if any(len(values) != 2 for values in split_outcomes.values()):
        print("WARNING: A split has only one outcome class; AUC will be undefined.")
    print("Output: " + str(args.output_dir.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--students", type=int, default=2000)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--concepts", type=int, default=3)
    parser.add_argument("--questions-per-concept", type=int, default=5)
    parser.add_argument("--initial-ability", type=float, default=-1.0)
    parser.add_argument("--ability-sd", type=float, default=0.5)
    parser.add_argument("--difficulty-sd", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--forgetting", type=float, default=0.0)
    parser.add_argument("--guessing", type=float, default=0.25)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=8240)
    args = parser.parse_args()
    if args.students < 2 or args.sequence_length < 2 or args.concepts < 1 or args.questions_per_concept < 1:
        parser.error("Need at least 2 students, 2 interactions, 1 concept and 1 question per concept")
    for name in ["initial_ability", "ability_sd", "difficulty_sd", "learning_rate", "forgetting", "guessing", "noise", "train_fraction"]:
        if not math.isfinite(getattr(args, name)):
            parser.error(name + " must be finite")
    if any(getattr(args, key) < 0 for key in ["ability_sd", "difficulty_sd", "learning_rate", "forgetting"]):
        parser.error("Standard deviations, learning rate and forgetting must be nonnegative")
    if not 0 <= args.guessing < 1 or not 0 <= args.noise <= 0.5 or not 0 < args.train_fraction < 1:
        parser.error("Require 0 <= guessing < 1, 0 <= noise <= 0.5, and 0 < train-fraction < 1")
    generate(args)


if __name__ == "__main__":
    main()
