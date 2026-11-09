# Deep Knowledge Tracing

This project trains an LSTM to predict student responses and evaluates it on
held-out students. The implementation follows the DKT model used in the
[Khajah, Lindsey, and Mozer comparison with Bayesian Knowledge Tracing](https://www.cs.colorado.edu/~mozer/Research/Selected%20Publications/reprints/KhajahLindseyMozer2016.pdf).

## Table of contents

- [Run on Windows](#run-on-windows)
  - [Run all datasets](#run-all-datasets)
  - [Run one dataset at a time](#run-one-dataset-at-a-time)
- [Results](#results)
- [Review DKT results with Gemini](#review-dkt-results-with-gemini)
  - [Set up and test the API key](#set-up-and-test-the-api-key)
  - [Run a review](#run-a-review)
  - [Read the assessment](#read-the-assessment)
- [Repository layout](#repository-layout)

## Run on Windows

Clone this repository and open PowerShell in the repository root. The
prepared data and fixed student splits for the six bundled experiments are
included. Install Miniconda, then create and activate the training environment.

```powershell
conda env create -f environment.yml
conda activate dkt-legacy
```

If PowerShell does not recognize Conda, run `conda init powershell` from
Anaconda Prompt and reopen PowerShell. The environment installs Python 2.7,
the pinned training dependencies, and Theano's Windows compiler. Create
the environment once and activate it in each new session.

### Run all datasets

Run the script without a dataset argument to train each prepared dataset
in sequence for **50 epochs** by default. This includes DBE-KT22 once its
data and split are prepared locally. On a fresh checkout the runner skips
DBE-KT22 until its files are available. [Dataset preparation instructions](data/README.md#dbe-kt22)

```powershell
.\run-experiments.ps1 -Check
.\run-experiments.ps1
```

The first command verifies that the prepared data and split files exist.
Add `-Epochs 1` to the second command for a short run across all available datasets.

### Run one dataset at a time

Choose the command for the dataset you want.

```powershell
.\run-experiments.ps1 -Dataset assistments
.\run-experiments.ps1 -Dataset comp1350
.\run-experiments.ps1 -Dataset idetrace
.\run-experiments.ps1 -Dataset dbekt22
.\run-experiments.ps1 -Dataset mathe
.\run-experiments.ps1 -Dataset synthetic
.\run-experiments.ps1 -Dataset piech
```

`synthetic` selects the locally generated data. `piech` selects the
original five-concept matrix. For a short test of one dataset, specify its
name and an epoch count.

```powershell
.\run-experiments.ps1 -Dataset idetrace -Epochs 1
```

To run a subset and compare several epoch counts, pass comma-separated lists.

```powershell
.\run-experiments.ps1 -Dataset comp1350,idetrace -Epochs 10,20,50
```

Each selected dataset trains once up to the largest epoch count. The runner
reports test AUC at each requested epoch from that same training run.

If the legacy compiler fails, add `-NoCompiler` to use Theano's slower
Python implementation. If PowerShell blocks the script, run
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run-experiments.ps1`
from the activated environment.

## Results

The runner prints one summary table after all selected datasets finish,
with the dataset filename, completed epochs, and test AUC. It also writes
`summary.csv` under a new `output/runs/`
folder, with one row per dataset and requested epoch. The CSV also includes
accumulated training loss and the previously recorded 50-epoch AUC. The
dataset subfolders contain per-epoch histories and the final epoch's
predictions and model weights.

| Dataset | Recorded test AUC at epoch 50 |
| --- | ---: |
| Assistments | 0.844855 |
| COMP1350 | 0.784670 |
| IDETrace | 0.483487 |
| DBE-KT22 | 0.799563 |
| MathE (file order) | 0.563887 |
| Generated synthetic | 0.613879 |
| Piech five-concept matrix | 0.810868 |

These values come from earlier runs with the prepared inputs and settings.
Training initializes weights and shuffles students randomly, so the AUCs
can vary. MathE uses source-file order as a pipeline check.

## Review DKT results with Gemini

The optional judge uses Gemini 3.1 Flash-Lite (`gemini-3.1-flash-lite`) to
review completed runs. It helps interpret exploratory results, flag patterns
that need a closer look, and suggest follow-up checks. The judge script uses
the Python 3 standard library. On Windows, check that `py -3 --version` works
alongside the legacy Python 2 environment used for training.

### Set up and test the API key

Create an authorization key in [Google AI Studio](https://aistudio.google.com/apikey).
From the repository root, create `.env` if it does not already exist.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Open `.env` and paste the full key after `GEMINI_API_KEY=`. Git ignores
this file. Then check the key and model.

```powershell
py -3 scripts/judge_gemini.py --check-key
```

The check sends a fixed prompt without dataset information. On success it
prints `Gemini key and model check passed.` If Gemini returns HTTP 401,
check that you copied the full key from the intended AI Studio project.
The check prints the key source. A shell value takes precedence over
`.env`. To clear an old override in the current PowerShell session, run
`Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue` and retry.

### Run a review

Add `--gemini-judge` at the end of the runner command to review each
selected dataset after the results table prints. Replace `idetrace` with any
dataset name listed above, or omit `-Dataset` to review all available datasets.

```powershell
.\run-experiments.ps1 -Dataset idetrace -Epochs 1 --gemini-judge
```

To review a saved run without retraining, use
`py -3 scripts/judge_gemini.py --dataset <dataset-path> --splitfile <split-path>`
with the dataset copy from the run folder and its original split file.
The saved `.preds` and `.history` must be beside the dataset copy.
Add `--dry-run` to preview the aggregate evidence locally.

### Read the assessment

The script prints Gemini's verdict and assessment. The JSON report and
suggested checks are saved beside the dataset copy, for example at
`output/runs/<run-id>/idetrace/idetrace.txt.gemini.json`. If a review
fails, the runner reports a warning and continues with the training results
already saved.

Gemini receives the dataset context, training settings, split counts, AUC
history, log loss, Brier score, calibration summaries, and a baseline based
on the training set's correct-response rate. Student IDs, raw responses, and
individual predictions stay local. Google's unpaid API terms allow it to use
submitted inputs and outputs to improve its products. [Gemini API terms](https://ai.google.dev/gemini-api/terms)

Compare Gemini's comments with the locally computed metrics before using
them to interpret a result.

## Repository layout

| Path | Purpose |
| --- | --- |
| `dkt.py` | Model, sequence batching, training, and evaluation |
| `run-experiments.ps1` | Run selected experiments or the entire set |
| `experiments.json` | Dataset paths, model settings, and recorded AUCs |
| `environment.yml` | Windows training environment |
| `data/` | Prepared datasets and [data provenance and conversion commands](data/README.md) |
| `scripts/` | Data converters, synthetic generator, and workbook inspection tools |
