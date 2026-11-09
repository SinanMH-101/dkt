"""Convert an original DKT synthetic response matrix to the local text format."""

import argparse
import csv
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Headerless response matrix CSV')
    parser.add_argument('--output-dir', type=Path, default=Path('data/synthetic_piech'))
    parser.add_argument('--seed', type=int, default=8240)
    parser.add_argument('--train-fraction', type=float, default=0.8,
                        help='Fraction of students assigned to training (default: 0.8)')
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error('--train-fraction must be between 0 and 1')

    with args.source.open(newline='', encoding='utf-8-sig') as source:
        response_matrix = list(csv.reader(source))
    if len(response_matrix) < 2:
        parser.error('the matrix must contain at least two students')
    questions = len(response_matrix[0])
    for line, responses in enumerate(response_matrix, 1):
        if not questions or len(responses) != questions:
            parser.error('row {} has an inconsistent number of questions'.format(line))
        if any(response not in ('0', '1') for response in responses):
            parser.error('row {} contains a response other than 0 or 1'.format(line))

    student_count = len(response_matrix)
    train_count = int(student_count * args.train_fraction)
    if not 0 < train_count < student_count:
        parser.error('--train-fraction must leave at least one student in each split')
    training = set(random.Random(args.seed).sample(range(student_count), train_count))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.output_dir / (args.source.stem + '.txt')
    split = args.output_dir / (args.source.stem + '_split.txt')
    with dataset.open('w', encoding='ascii', newline='\n') as out:
        # A row is one student, and column position gives the exercise and order.
        for student, responses in enumerate(response_matrix):
            for question, response in enumerate(responses):
                out.write('{} {} {}\n'.format(student, question, response))
    split.write_text(' '.join('1' if i in training else '0'
                              for i in range(student_count)), encoding='ascii')
    print('{} train / {} test students, {} questions, {} responses'.format(
        train_count, student_count - train_count, questions, student_count * questions))
    print(dataset)
    print(split)


if __name__ == '__main__':
    main()
