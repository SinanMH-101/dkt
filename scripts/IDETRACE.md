# Convert IDETrace to DKT

This converter uses Python 3's standard library. It processes one single-task
deployment at a time. The downloaded `deployment_1.json` is the GradeBook task.
Download raw telemetry from the authors' repository:
https://github.com/divadbaroon/TutorTrace_dataset_and_benchmark/tree/main/dataset/raw_telemetry

From the project folder:

```powershell
conda run -n dkt-convert python scripts/convert_idetrace.py
```

Alternatively, use `py -3 scripts/convert_idetrace.py` with standalone Python 3.
The defaults read `data/raw/tutortrace/deployment_1.json` and write:

- `data/idetrace/idetrace.txt`: three-column DKT input, no header.
- `data/idetrace/idetrace_split.txt`: reproducible 80/20 split by participant.
- `data/idetrace/idetrace_runs.csv`: execution-level audit data, with newly numbered student IDs.
- `data/idetrace/idetrace_summary.json`: exclusions, counts and assumptions.

One `TEST_CASE_RESULT` event is one interaction. It is correct only when all
tests pass. Partial passes are 0 for this whole-task outcome. The converter checks
the individual Boolean results against total count, passed count and pass rate,
excluding missing/inconsistent events. Other event types are not correctness
observations. Real repeated executions are retained. Sequences are ordered by
timestamp, then source event position for ties. Participants with fewer than two
valid runs are excluded. Every participant is entirely in one split.

All interactions use exercise ID 0, representing this deployment's single task.
`TASK_SELECT` captures highlighted text, so it is not used as a task identifier.
Do not apply this representation to a multi-task deployment without first checking
its schema. This dataset tests repeated program-run success on one task; it cannot
establish that DKT recovers relationships between knowledge components.

The four test results in an execution are simultaneous outcomes. Serializing them
as four DKT steps would introduce an artificial order and let the model condition
on other results from the same execution. The run-level representation avoids
that issue. Original participant identifiers, names, code and chat are not exported.

For another verified single-task deployment, provide its path and task label:

```powershell
conda run -n dkt-convert python scripts/convert_idetrace.py --input data/raw/tutortrace/deployment_2.json --task GradeBook --output-dir data/idetrace_d2
```

This command creates a separate within-deployment student split, not the authors'
D1-to-D2 benchmark. It does not combine students across deployment files.
Rerunning overwrites the selected output files.

Train in your configured legacy environment:

```powershell
conda activate dkt-legacy
python dkt.py --dataset data/idetrace/idetrace.txt --splitfile data/idetrace/idetrace_split.txt --epochs 1
```
