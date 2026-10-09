# Synthetic DKT dataset

Run from the project folder using the Python 3 environment described in README.md.
The generator uses only the standard library and needs no additional packages.

```powershell
conda run -n dkt-convert python scripts/generate_synthetic.py
```

Defaults generate 2,000 students with 50 interactions each, 3 independent latent
concepts and 5 questions per concept. Questions within a concept share its ability.
Each student has normally distributed starting abilities and a randomly varied
learning rate. Question difficulty is fixed. Correctness is sampled using a
guessing-adjusted logistic model. Practice increases the relevant concept's ability.
Questions are selected uniformly with replacement. These are explicit simulation
assumptions, not a reproduction of the original DKT paper's synthetic experiment.

Outputs under `data/synthetic/`:

- `synthetic.txt`: student ID, question ID and correctness, space-delimited without a header.
- `synthetic_split.txt`: 80/20 student assignments, with no trailing whitespace.
- `synthetic_ground_truth.csv`: latent ability, probabilities and response noise per interaction.
- `synthetic_metadata.json`: parameters, question/concept mapping, student initial abilities and summary.

DKT must receive only the text dataset and split. The ground truth provides known
relationships for later analysis. AUC measures response prediction; recovering
the shared-concept structure needs a separate analysis of learned relationships.

Create a matched response-noise experiment using the same seed and other settings:

```powershell
conda run -n dkt-convert python scripts/generate_synthetic.py --output-dir data/synthetic_clean --noise 0 --seed 8240
conda run -n dkt-convert python scripts/generate_synthetic.py --output-dir data/synthetic_noisy --noise 0.1 --seed 8240
```

The variants share abilities, difficulties, question sequences, splits and sampled
responses before noise. Noise flips approximately 10% of responses in the second
variant. Optional `--forgetting` subtracts ability from all concepts between
interactions. Use `--help` to see every setting. Existing files in the selected
output directory are overwritten when the generator runs again.

Train in the configured legacy environment:

```powershell
conda activate dkt-legacy
python dkt.py --dataset data/synthetic/synthetic.txt --splitfile data/synthetic/synthetic_split.txt --epochs 1
```

Use matching training settings when comparing variants. Repeat across seeds for
stronger conclusions; a single simulation run is preliminary evidence.
