# Deep Knowledge Tracing

This project trains an LSTM to predict student responses and evaluates it on
held-out students. The implementation follows the DKT model used in the
[Khajah, Lindsey, and Mozer comparison with Bayesian Knowledge Tracing](https://www.cs.colorado.edu/~mozer/Research/Selected%20Publications/reprints/KhajahLindseyMozer2016.pdf).

## Run on Windows

Clone this repository and open PowerShell in the repository root. The
prepared data and fixed student splits for all six experiments are included.
Install Miniconda, then run:

```powershell
conda env create -f environment.yml
conda activate dkt-legacy
.\run-experiments.ps1
```

If PowerShell does not recognize Conda, run `conda init powershell` from
Anaconda Prompt and reopen PowerShell. The environment installs Python 2.7,
the pinned training dependencies, and Theano's Windows compiler. Create
the environment once and activate it in each new session.

The default run trains all six experiments for **50 epochs each**. To check
that the inputs are present or run a short one-epoch test:

```powershell
.\run-experiments.ps1 -Check
.\run-experiments.ps1 -Dataset idetrace -Epochs 1
```

To compare several epoch counts in one table, pass a comma-separated list:

```powershell
.\run-experiments.ps1 -Dataset comp1350,idetrace -Epochs 10,20,50
```

Each selected dataset trains once up to the largest epoch count. The runner
reports test AUC at each requested epoch from that same training run. Omit
`-Dataset` to include all six experiments.

If the legacy compiler fails, add `-NoCompiler` to use Theano's slower
Python implementation. If PowerShell blocks the script, run
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run-experiments.ps1`
from the activated environment.

## Results

Each invocation prints a bordered table with the dataset filename, completed
epochs, and test AUC. It also writes `summary.csv` under a new `output/runs/`
folder, with one row per dataset and requested epoch. The CSV also includes
accumulated training loss and the previously recorded 50-epoch AUC. The
dataset subfolders contain per-epoch histories and the final epoch's
predictions and model weights.

| Dataset | Recorded test AUC at epoch 50 |
| --- | ---: |
| Assistments | 0.844855 |
| COMP1350 | 0.784670 |
| IDETrace | 0.483487 |
| MathE (file order) | 0.563887 |
| Generated synthetic | 0.613879 |
| Piech five-concept matrix | 0.810868 |

These values come from earlier runs with the included inputs and settings.
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

Add `--gemini-judge` to a direct `dkt.py` command to request one review
after the final epoch. This example uses the prepared
[DBE-KT22 dataset](data/README.md#dbe-kt22).

```powershell
conda activate dkt-legacy
$env:KERAS_BACKEND = 'theano'
$env:THEANO_FLAGS = 'device=cpu,floatX=float32,gcc.cxxflags=-march=core2'
python dkt.py --dataset data/dbekt22/dbekt22.txt --splitfile data/dbekt22/dbekt22_split.txt --epochs 50 --hiddenunits 32 --batchsize 20 --timewindow 50 --gemini-judge
```

If the run already has `.preds` and `.history` files alongside the
dataset, you can review it without retraining. Use the settings from that run.

```powershell
py -3 scripts/judge_gemini.py --dataset data/dbekt22/dbekt22.txt --splitfile data/dbekt22/dbekt22_split.txt --hiddenunits 32 --batchsize 20 --timewindow 50
```

Add `--dry-run` to the standalone command to inspect the exact aggregate
evidence locally without making an API request.

### Read the assessment

The script prints Gemini's verdict and assessment and saves the full review,
including suggested checks, beside the dataset. For the example above, the
report is `data/dbekt22/dbekt22.txt.gemini.json`. If Gemini fails after
training, the saved training outputs can still be reviewed with the standalone
command.

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
