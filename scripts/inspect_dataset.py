"""Print COMP1350 workbook schema and quiz event integrity checks as JSON lines."""

import json
import pandas as pd

path = "Sinan_Dataset.xlsx"
book = pd.ExcelFile(path)
print(json.dumps({"sheets": book.sheet_names}))
for sheet in book.sheet_names:
    df = pd.read_excel(path, sheet_name=sheet)
    out = {
        "sheet": sheet,
        "shape": list(df.shape),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "nulls": {str(c): int(v) for c, v in df.isna().sum().items()},
        "unique": {str(c): int(df[c].nunique(dropna=True)) for c in df.columns},
        "sample": df.head(3).astype(object).where(pd.notna(df.head(3)), None).to_dict("records"),
    }
    print(json.dumps(out, default=str))

# Restrict the event-level audit to rows associated with a quiz attempt.
df = pd.read_excel(path, sheet_name="Sinan_Data")
df = df[df["quizattemptid"].notna()].copy()
key_stats = df.groupby("uniquekey").agg(
    rows=("uniquekey", "size"), participants=("Participant-ID", "nunique"),
    quizattempts=("quizattemptid", "nunique"), questions=("questionid", "nunique"),
    slots=("slot", "nunique"), timestamps=("timestamp", "nunique"),
    fractions=("fraction", "nunique"), states=("state", "nunique"),
)
participant_events = df.dropna(subset=["Participant-ID"]).groupby("Participant-ID")["uniquekey"].nunique()
participant_questions = df.dropna(subset=["Participant-ID"]).groupby("Participant-ID")["questionid"].nunique()
participant_attempts = df.dropna(subset=["Participant-ID"]).groupby("Participant-ID")["quizattemptid"].nunique()
print(json.dumps({
    "valid_rows": len(df),
    "rows_missing_participant": int(df["Participant-ID"].isna().sum()),
    "keys_missing_participant": int(df.loc[df["Participant-ID"].isna(), "uniquekey"].nunique()),
    "key_integrity_maxima": key_stats.max().to_dict(),
    "events_per_participant": participant_events.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_dict(),
    "questions_per_participant": participant_questions.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_dict(),
    "quizattempts_per_participant": participant_attempts.value_counts().sort_index().to_dict(),
    "state_counts": df["state"].value_counts(dropna=False).to_dict(),
    "fraction_counts": df["fraction"].value_counts(dropna=False).to_dict(),
    "time_min": pd.to_datetime(df["timestamp"].min(), unit="s").isoformat(),
    "time_max": pd.to_datetime(df["timestamp"].max(), unit="s").isoformat(),
}, default=lambda x: x.item() if hasattr(x, "item") else str(x)))

# Collapse metadata rows to one record per step to count graded interactions.
step = df.dropna(subset=["attemptstepid"]).groupby("attemptstepid").agg(
    participant=("Participant-ID", "first"), participant_n=("Participant-ID", "nunique"),
    question=("questionid", "first"), question_n=("questionid", "nunique"),
    slot=("slot", "first"), slot_n=("slot", "nunique"),
    timestamp=("timestamp", "first"), timestamp_n=("timestamp", "nunique"),
    fraction=("fraction", "max"), fraction_n=("fraction", "nunique"),
    state=("state", "first"), state_n=("state", "nunique"), rows=("state", "size"),
).reset_index()
known = step.dropna(subset=["participant"])
all_steps_pp = known.groupby("participant")["attemptstepid"].nunique()
graded = known[known["state"].isin(["gradedright", "gradedwrong"])]
graded_pp = graded.groupby("participant")["attemptstepid"].nunique().reindex(all_steps_pp.index, fill_value=0)
print(json.dumps({
    "attempt_steps": len(step),
    "step_integrity_maxima": {c: int(step[c].max()) for c in ["participant_n","question_n","slot_n","timestamp_n","fraction_n","state_n"]},
    "steps_per_participant": all_steps_pp.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_dict(),
    "graded_steps": len(graded),
    "graded_steps_per_participant": graded_pp.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_dict(),
    "graded_state_counts": graded["state"].value_counts().to_dict(),
}, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
