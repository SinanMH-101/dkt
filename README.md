# Deep Knowledge Tracing Implementation
This repository contains our implementation of the [Deep Knowledge Tracing (DKT)](http://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf) model
which we used in [our paper](https://www.cs.colorado.edu/~mozer/Research/Selected%20Publications/reprints/KhajahLindseyMozer2016.pdf).

DKT is a recurrent neural network model designed to predict students' performance. 
The authors of the DKT paper used a Long-Term Short-Term (LSTM) network in the paper but 
only [published code](https://github.com/chrispiech/DeepKnowledgeTracing)
for a simple recurrent neural network. In our paper we compare various enhanced
flavors of Bayesian Knowledge Tracing (BKT) to DKT. To ensure a fair comparison to DKT,
we implemented our own LSTM variant of DKT. This repository contains our implementation.


# Running on Windows (Miniconda and PowerShell)

This is a legacy Python 2 implementation. Use a separate Python 2.7 environment;
the script does not run unchanged with Python 3 or modern Keras.
The following setup was used successfully on Windows with CPU execution.

## 1. Enable Conda in PowerShell

Install Miniconda, open **Anaconda Prompt (Miniconda3)** from the Start menu,
and run:

```bat
conda init powershell
```

Close and reopen PowerShell. If using VS Code's integrated terminal, restart
VS Code if necessary. Confirm that `conda --version` works.

## 2. Create and activate the environment

From PowerShell in the repository folder:

```powershell
cd D:\Study\dkt
conda create -n dkt-legacy python=2.7 pip
conda activate dkt-legacy
python --version
```

Replace the folder path with your checkout location. The prompt should start
with `(dkt-legacy)` and the Python version should be `2.7.18`. Activate this
environment in every new terminal before installing dependencies or running DKT.

## 3. Install dependencies and the compiler

```powershell
python -m pip install -r .devcontainer/requirements-legacy.txt
conda install -c defaults m2w64-toolchain libpython
conda deactivate
conda activate dkt-legacy
g++ --version
```

The requirements file pins Keras, Theano, NumPy, SciPy, scikit-learn, h5py,
and their supporting packages. The compiler enables Theano's faster C
implementations. Review Conda's proposed package changes before accepting them.

## 4. Configure Theano and run one epoch

Run these commands in the same PowerShell terminal:

```powershell
$env:KERAS_BACKEND = "theano"
$env:THEANO_FLAGS = "device=cpu,floatX=float32,gcc.cxxflags=-march=core2"
python -c "import theano; print(theano.config.cxx)"
python dkt.py --dataset assistments.txt --splitfile assistments_split.txt --epochs 1
```

The compiler check should print `g++` or its executable path. The `core2` target
avoids the CPU-specific compilation failure encountered with this older Windows
toolchain (`invalid register for .seh_savexmm`), while retaining C compilation.
These environment settings apply only to the current terminal; set them again
when opening a new terminal.

The first run can take time to compile. Training then prints percentage progress
and accumulated loss. The loss accumulates across batches, so an increasing
number alone does not indicate worsening performance. After training reaches
100%, the script evaluates the test set and prints `Epoch: 0, Test AUC: ...`.

## 5. Run longer training

```powershell
python dkt.py --dataset assistments.txt --splitfile assistments_split.txt --epochs 50
```

Each invocation starts a new model; the script does not automatically resume
saved weights. It saves outputs after each completed epoch and overwrites the
files from earlier runs using the same dataset path. Interrupting an unfinished
epoch with `Ctrl+C` loses that epoch's progress.

For a smaller, quicker trial:

```powershell
python dkt.py --dataset assistments.txt --splitfile assistments_split.txt --epochs 1 --hiddenunits 32 --batchsize 20
```

This changes the model size and batching, so results may differ from the defaults.

## Troubleshooting

- **`conda` is not recognized:** run `conda init powershell` from Anaconda Prompt,
  then restart your terminal or editor.
- **NumPy installation fails while the prompt shows `(base)`:** activate
  `dkt-legacy` and verify `python --version` before retrying.
- **`g++ not detected`:** install the compiler above, reactivate the environment,
  and check `g++ --version`. Disabling the compiler only hides the warning and
  leaves training slow.
- **`invalid register for .seh_savexmm`:** use the exact `THEANO_FLAGS` setting
  above before starting a new Python process.
- **`class_mode` is deprecated:** this warning from legacy Keras does not stop
  training.

# Data Format
The model is contained within one script. It expects two files: a dataset file and
a split file. The dataset file contains student, skill and performance information whilst the split file specifies which students belong to the training set. 

The dataset file is a 3-column space-delimited file. Each row in the file indicates whether a particular student answered a specific problem correctly or not. 
The first column is the student id, the second column is the skill id associated with the problem and the last column is whether the student got the problem correctly (1) or not (0).

The split file contains space-delimited assignments in ascending student ID order:
`1` places that student's sequence in training and `0` places it in testing.
There must be one assignment per unique student. Use a single line with single
spaces and no trailing whitespace, because the legacy parser does not strip it.
For contiguous student IDs starting at zero, the split file:

    1 1 0 1 1 0
    
indicates that students 0, 1, 3 and 4 should be the training set and the rest will be in the test set. 

Excel workbooks such as `Sinan_Dataset.xlsx` cannot be passed directly to
`dkt.py`. Convert them to the three-column text format and create a matching
split file first. Preserve interaction order within each student, and use
binary correctness values (`0` or `1`).

# Usage

```sh
python dkt.py [-h] --dataset DATASET --splitfile SPLITFILE
              [--hiddenunits HIDDENUNITS] [--batchsize BATCHSIZE]
              [--timewindow TIMEWINDOW] [--epochs EPOCHS]
```

The script will emit three files:
 - **DATASET.model_weights**: contains neural network weights
 - **DATASET.history**: a two-column, tab-delimited file containing accumulated training loss and test AUC, one row per completed epoch.
 - **DATASET.preds**: a tab-delimited file with header `was_heldout`, `prob_recall`, and `student_recalled`. Each row contains `1`, the model prediction, and the actual binary outcome for a test trial.

Generated weights, history, and predictions are ignored by Git. The bundled
`assistments.txt` and `assistments_split.txt` remain tracked. The local
`Sinan_Dataset.xlsx` workbook is also ignored.

The instructions above cover CPU execution. Legacy Theano GPU execution requires
a separate compatible CUDA setup.

# Datasets

We have included the skill builder version of the [Assistments 2009-2010 dataset](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010)
which is one of the datasets evaluated in the paper.
