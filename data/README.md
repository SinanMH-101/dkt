# Experiment data

The repository includes the plain-text datasets and fixed student splits
for the six bundled experiments in [experiments.json](../experiments.json),
alongside additional prepared synthetic variants. DBE-KT22 is also listed in
the experiment configuration and requires local preparation. The Assistments
pair is at the repository root.
The COMP1350 and IDETrace inputs contain renumbered student IDs, exercise IDs,
and binary outcomes. Raw sources and audit exports are kept separately.

| Dataset | Source and preparation |
| --- | --- |
| Assistments | The included 2009-2010 Skill Builder dataset and student split. [Dataset source](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010) |
| COMP1350 | Local `Sinan_Dataset.xlsx`, sheet `Sinan_Data`. Keep consistent graded steps and order them by timestamp, then step ID for ties. |
| IDETrace | TutorTrace GradeBook deployment 1. One binary outcome per execution, with success defined as all tests passing. [Source](https://github.com/divadbaroon/TutorTrace_dataset_and_benchmark) |
| MathE | UCI's *Dataset for Assessing Mathematics Learning in Higher Education*, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The converter renumbers students, filters single-response students, and retains question IDs and correctness in CSV order. [Source and attribution](https://archive.ics.uci.edu/dataset/1031) |
| Generated synthetic | Seeded simulation with three latent concepts and no response flips. |
| Piech five-concept matrix | Five-concept variant 0 from the original DKT release, converted to interaction rows with a seeded 50/50 student split. [Source matrix](https://github.com/chrispiech/DeepKnowledgeTracing/blob/master/data/synthetic/naive_c5_q50_s4000_v0.csv) |

## Regenerate the prepared files

Run these commands if the source inputs change. The default experiments
use the prepared files in the repository. Use PowerShell from the root.

```powershell
conda create -n dkt-convert python=3.11 pip -y
conda run -n dkt-convert python -m pip install -r scripts/requirements-convert.txt
```

For the converters that read source files, place them at these paths:

- COMP1350 workbook at `Sinan_Dataset.xlsx` in the repository root.
- IDETrace deployment at `data/raw/tutortrace/deployment_1.json`.
- MathE CSV at `data/raw/mathe/MathE dataset (4).csv`.
- Piech matrix at `data/synthetic-original/naive_c5_q50_s4000_v0.csv`.

Run the commands for the datasets you need:

```powershell
conda run -n dkt-convert python scripts/convert_comp1350.py
conda run -n dkt-convert python scripts/convert_idetrace.py
conda run -n dkt-convert python scripts/convert_mathe.py
conda run -n dkt-convert python scripts/generate_synthetic.py --output-dir data/synthetic_clean --seed 8240 --noise 0
conda run -n dkt-convert python scripts/convert_synthetic_original.py data/synthetic-original/naive_c5_q50_s4000_v0.csv --train-fraction 0.5 --seed 8240 --output-dir data/synthetic_piech_comparison
```

## Dataset notes

- **IDETrace:** Use the authors' [raw telemetry](https://github.com/divadbaroon/TutorTrace_dataset_and_benchmark/tree/main/dataset/raw_telemetry). The converter processes one verified single-task deployment at a time. Runs are ordered by timestamp, with source position breaking ties. Participants with fewer than two valid runs are excluded. Each run has one outcome; the simultaneous test results are not treated as separate steps. The split is by participant within the deployment, and the model predicts success on one task.
- **MathE:** The source has no attempt timestamps. Its file-order AUC is a pipeline check rather than a chronological next-response benchmark. The converter uses `1` for correct and `0` for incorrect, consistent with the totals in Table 2 of the [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10912334/).
- **Generated synthetic:** Defaults are 2,000 students, 50 interactions each, and five questions per concept. Practising a question increases ability for its concept. The generator writes ground-truth probabilities and metadata for analysis; only the dataset and student split go into training. Use `--noise 0.1` for a matched noisy variant or `--ability-sd 2.0` for more spread in starting ability. Use a separate `--output-dir` for each variant to preserve previous files.
- **Piech:** Each original CSV row is a student and each column is an exercise in sequence order. The converter uses only binary responses, leaving the `info/` files out. The 50/50 split matches the paper's student counts, but the assignment and training setup differ. For a comparison across variants, use the same settings on `v0` through `v19`.

## DBE-KT22

Download the CSV ZIP from the [Australian Data Archive](https://doi.org/10.26193/6DZWOH)
under its data-use agreement. Save it as
`data/raw/dbekt22/2_DBE_KT22_datafiles_100102_csv.zip`, or pass a ZIP or extracted
`Transaction.csv` using `--input`. Raw and prepared files are kept local.

```powershell
conda run -n dkt-convert python scripts/convert_dbekt22.py
```

The converter keeps graded attempts and sorts each student's attempts by
`start_time`, with transaction ID breaking ties. It uses question IDs and
an 80/20 student split with seed 8240. Cite the [archive deposit](https://doi.org/10.26193/6DZWOH)
and the [dataset paper](https://arxiv.org/abs/2208.12651) when reporting results.

## Input format

The training dataset has one interaction per line with space-separated
`student_id skill_id correct` fields and no header. Keep each student's
interactions in sequence order. The split has one line of space-separated
flags in ascending student ID order, with no trailing whitespace. A flag of
`1` assigns the student to training and `0` assigns them to testing.
