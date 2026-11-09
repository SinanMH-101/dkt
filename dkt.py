"""Train and evaluate an LSTM for Deep Knowledge Tracing."""

# Mohammad M H Khajah <mohammad.khajah@colorado.edu>
# Copyright (c) 2016 all rights reserved.

import argparse
import logging
import math
import os
import random
import subprocess
import sys


class TheanoCpuTargetFilter(logging.Filter):
    def filter(self, record):
        # An explicit CPU target is intentional for the legacy Windows compiler.
        return not record.getMessage().startswith(
            "WARNING: your Theano flags `gcc.cxxflags` specify an `-march=X` flags.")


# Install the filter before importing Keras, which loads Theano.
logging.getLogger('theano.gof.cmodule').addFilter(TheanoCpuTargetFilter())

import numpy as np
from keras import backend as K
from keras.layers.core import TimeDistributedDense, Masking
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.metrics import roc_auc_score
import theano.tensor as tensor


def main():
    parser = argparse.ArgumentParser(
        description='Train an LSTM and evaluate its predictions on held-out students.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, help='Dataset file', required=True)
    parser.add_argument('--splitfile', type=str, help='Split file', required=True)
    parser.add_argument('--hiddenunits', type=int, help='Number of LSTM hidden units.',
                        default=200)
    parser.add_argument('--batchsize', type=int, help='Number of sequences to process in a batch.',
                        default=5)
    parser.add_argument('--timewindow', type=int, help='Number of timesteps to process in a batch.',
                        default=100)
    parser.add_argument('--epochs', type=int, help='Number of epochs.',
                        default=20)
    parser.add_argument('--gemini-judge', action='store_true',
                        help='Send aggregate results to Gemini for review after training.')
    parser.add_argument('--no-summary', action='store_true',
                        help='Suppress the final results table.')
    args = parser.parse_args()

    dataset = args.dataset
    split_file = args.splitfile
    hidden_units = args.hiddenunits
    batch_size = args.batchsize
    time_window = args.timewindow
    epochs = args.epochs

    model_file = dataset + '.model_weights'
    history_file = dataset + '.history'
    preds_file = dataset + '.preds'

    history = []

    training_seqs, testing_seqs, num_skills = load_dataset(dataset, split_file)
    print "Training Sequences: %d" % len(training_seqs)
    print "Testing Sequences: %d" % len(testing_seqs)
    print "Number of skills: %d" % num_skills

    def loss_function(y_true, y_pred):
        # Select the prediction for the skill attempted at each timestep.
        skill_mask = y_true[:, :, :num_skills]
        correct = y_true[:, :, num_skills]
        attempted_skill_prob = tensor.sum(y_pred * skill_mask, axis=2)

        # Keep one loss per timestep so Keras can mask padding before averaging.
        return K.binary_crossentropy(attempted_skill_prob, correct)

    input_shape = (batch_size, time_window, num_skills * 2)
    model = Sequential()
    model.add(Masking(-1.0, batch_input_shape=input_shape))
    model.add(LSTM(input_dim=num_skills * 2,
                   output_dim=hidden_units,
                   return_sequences=True,
                   batch_input_shape=input_shape,
                   stateful=True))
    # The same readout predicts one probability per skill at every timestep.
    model.add(TimeDistributedDense(input_dim=hidden_units,
                                   output_dim=num_skills, activation='sigmoid'))
    model.compile(loss=loss_function, optimizer='rmsprop')

    def train_window(inputs, targets):
        accumulated_loss[0] += model.train_on_batch(inputs, targets)[0]

    def predict_window(inputs, targets):
        batch_activations = model.predict_on_batch(inputs)
        skill_mask = targets[:, :, :num_skills]
        correct = targets[:, :, num_skills]
        # Legacy Keras may wrap the output in a list. Preserve batch, time
        # and skill dimensions: squeeze removes the skill axis when there
        # is only one skill (and can also remove a singleton batch axis).
        y_pred = np.asarray(batch_activations).reshape(skill_mask.shape)

        attempted_skill_probs = np.sum(y_pred * skill_mask, axis=2)

        for student in xrange(inputs.shape[0]):
            for step in xrange(inputs.shape[1]):
                if inputs[student, step, 0] == -1.0:
                    continue
                predictions.append((attempted_skill_probs[student, step],
                                    correct[student, step]))

    def finish_test_batch(_percent_done):
        model.reset_states()

    def finish_training_batch(percent_done):
        sys.stdout.write('\rTraining: %6.2f%% | Accumulated loss: %.6f' %
                         (percent_done, accumulated_loss[0]))
        sys.stdout.flush()
        if percent_done >= 100.0:
            sys.stdout.write('\n')
        model.reset_states()

    for epoch in xrange(epochs):
        print "\nEpoch %d/%d" % (epoch + 1, epochs)
        # Use a mutable container so the callback can update it in Python 2.
        accumulated_loss = [0.0]
        predictions = []
        model.reset_states()
        run_sequence_batches(training_seqs, num_skills, train_window, batch_size,
                             time_window, finish_training_batch)

        model.reset_states()
        run_sequence_batches(testing_seqs, num_skills, predict_window, batch_size,
                             time_window, finish_test_batch)

        probabilities = [probability for probability, _ in predictions]
        outcomes = [correct for _, correct in predictions]
        auc = roc_auc_score(outcomes, probabilities)
        history.append((accumulated_loss[0], auc))

        model.save_weights(model_file, overwrite=True)
        print "Test AUC: %.6f" % auc

        with open(preds_file, 'w') as output:
            output.write('was_heldout\tprob_recall\tstudent_recalled\n')
            for probability, correct in predictions:
                output.write('1\t%f\t%d\n' % (probability, correct))

        with open(history_file, 'w') as output:
            for loss, test_auc in history:
                output.write('%s\t%s\n' % (loss, test_auc))

    if history:
        if not args.no_summary:
            print_results(dataset, len(history), history[-1][1])
        if args.gemini_judge:
            sys.stdout.flush()
            judge_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'scripts', 'judge_gemini.py')
            python3 = ['py', '-3'] if os.name == 'nt' else ['python3']
            try:
                subprocess.check_call(python3 + [
                    judge_script, '--dataset', dataset, '--splitfile', split_file,
                    '--hiddenunits', str(hidden_units), '--batchsize', str(batch_size),
                    '--timewindow', str(time_window),
                ])
            except (OSError, subprocess.CalledProcessError):
                parser.exit(1, 'Training outputs were saved. Gemini review failed. '
                            'Run scripts/judge_gemini.py with Python 3 to retry.\n')


def print_results(dataset, epochs, auc):
    headers = ['Dataset', 'Completed epochs', 'Final test AUC']
    values = [os.path.basename(dataset), str(epochs), '%.6f' % auc]
    widths = [max(len(header), len(value))
              for header, value in zip(headers, values)]
    border = '+' + '+'.join('-' * (width + 2) for width in widths) + '+'

    print "\nTraining complete"
    print border
    for row in (headers, values):
        print '| ' + ' | '.join(value.ljust(width)
                                for value, width in zip(row, widths)) + ' |'
        print border


def run_sequence_batches(seqs, num_skills, process_window, batch_size,
                         time_window, batch_done=None):
    """Apply a callback to each time window in a shuffled batch of students."""
    assert min(len(seq) for seq in seqs) > 0
    seqs = seqs[:]
    random.shuffle(seqs)

    input_dim = num_skills * 2
    target_dim = num_skills + 1
    processed = 0
    for start in xrange(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        inputs = []
        targets = []
        for sequence in batch:
            student_inputs = []
            student_targets = []
            # Predict the current response from the previous one. The first
            # prediction uses an all-zero input because there is no history.
            previous_response = [0] * input_dim
            for skill, correct in sequence:
                student_inputs.append(previous_response)
                target = [0] * target_dim
                target[skill] = 1
                target[num_skills] = correct
                student_targets.append(target)

                previous_response = [0] * input_dim
                previous_response[skill * 2 + correct] = 1

            inputs.append(student_inputs)
            targets.append(student_targets)

        max_length = round_to_multiple(max(map(len, batch)), time_window)
        # The stateful LSTM requires a fixed number of students per batch.
        for _ in xrange(batch_size - len(batch)):
            inputs.append([[-1.0] * input_dim] * time_window)
            targets.append([[0.0] * target_dim] * time_window)

        # Inputs of -1 mark padded timesteps for the model's masking layer.
        inputs = pad_sequences(inputs, padding='post', maxlen=max_length,
                               dim=input_dim, value=-1.0)
        targets = pad_sequences(targets, padding='post', maxlen=max_length,
                                dim=target_dim, value=-1.0)

        # Keep student slots fixed across windows so their LSTM states carry over.
        for start_step in xrange(0, max_length, time_window):
            end_step = start_step + time_window
            process_window(inputs[:, start_step:end_step, :],
                           targets[:, start_step:end_step, :])

        processed += len(batch)
        if batch_done:
            batch_done(processed * 100.0 / len(seqs))


def round_to_multiple(x, base):
    return int(base * math.ceil(float(x) / base))


def load_dataset(dataset, split_file):
    seqs, num_skills = read_file(dataset)

    # Split entries follow ascending student ID, matching read_file's order.
    with open(split_file, 'r') as f:
        student_assignment = f.read().split(' ')

    training_seqs = [seqs[i] for i in xrange(0, len(seqs)) if student_assignment[i] == '1']
    testing_seqs = [seqs[i] for i in xrange(0, len(seqs)) if student_assignment[i] == '0']

    return training_seqs, testing_seqs, num_skills


def read_file(dataset_path):
    """Group responses in file order and map skill IDs to model output indices."""
    seqs_by_student = {}
    problem_ids = {}
    next_problem_id = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            student, problem, is_correct = line.strip().split(' ')
            student = int(student)
            if student not in seqs_by_student:
                seqs_by_student[student] = []
            if problem not in problem_ids:
                problem_ids[problem] = next_problem_id
                next_problem_id += 1
            seqs_by_student[student].append((problem_ids[problem], int(is_correct == '1')))

    student_ids = sorted(seqs_by_student)
    return [seqs_by_student[student] for student in student_ids], next_problem_id


# https://groups.google.com/forum/#!msg/keras-users/7sw0kvhDqCw/QmDMX952tq8J
def pad_sequences(sequences, maxlen=None, dim=1, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pad or trim sequences while retaining all features at each timestep."""
    if maxlen is None:
        maxlen = np.max([len(sequence) for sequence in sequences])

    padded = (np.ones((len(sequences), maxlen, dim)) * value).astype(dtype)
    for index, sequence in enumerate(sequences):
        if truncating == 'pre':
            trimmed = sequence[-maxlen:]
        elif truncating == 'post':
            trimmed = sequence[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            padded[index, :len(trimmed)] = trimmed
        elif padding == 'pre':
            padded[index, -len(trimmed):] = trimmed
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return padded


if __name__ == "__main__":
    main()
