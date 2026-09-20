from pathlib import Path
import pandas as pd

FOLDERS = {
    "seed1": Path("Judge_seed1_scores"),
    "seed2": Path("Judge_seed2_scores"),
    "seed3": Path("Judge_seed3_scores"),
    "seed4": Path("Judge_seed4_scores"),
    "seed5": Path("Judge_seed5_scores"),
    "seed6": Path("Judge_seed6_scores"),
}

OUTPUT_DIR = Path("cross_seed_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOD_PROMPTS = [
    "q1_bored",
    "q2_wish",
    "q3_quick_buck",
    "q4_ruler",
    "q5_gender",
    "q6_dinner_party",
]

MEDICAL_PROMPT = "q7_medication"
CHECKPOINTS = [150, 199]

FILES = {}

for seed, folder in FOLDERS.items():

    matches = list(folder.glob("*individual*score*.csv"))

    if len(matches) == 0:
        raise FileNotFoundError(f"No individual scores file found for {seed} in {folder}")

    if len(matches) > 1:
        raise RuntimeError(f"Multiple individual score files found for {seed}: {matches}")

    FILES[seed] = matches[0]

all_rows = []

for seed, path in FILES.items():

    df = pd.read_csv(path)

    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce")
    df = df[df["checkpoint_step"].isin(CHECKPOINTS)].copy()

    df["seed"] = seed

    df["alignment"] = pd.to_numeric(df["alignment"], errors="coerce")
    df["coherence"] = pd.to_numeric(df["coherence"], errors="coerce")

    df["eligible"] = pd.to_numeric(df["eligible_for_em"], errors="coerce").fillna(0).eq(1)
    df["is_em"] = pd.to_numeric(df["em_flag"], errors="coerce").fillna(0).eq(1)

    all_rows.append(df)

data = pd.concat(all_rows, ignore_index=True)

print("\nPrompt IDs found:")
print(sorted(data["prompt_id"].dropna().unique()))

ood = data[data["prompt_id"].isin(OOD_PROMPTS)].copy()

ood_rows = []

for (seed, step), group in ood.groupby(["seed", "checkpoint_step"]):

    eligible = group[group["eligible"]]

    em_count = int(eligible["is_em"].sum())
    n_eligible = len(eligible)

    if n_eligible > 0:
        em_rate = 100 * em_count / n_eligible
    else:
        em_rate = float("nan")

    em_prompts = eligible.loc[eligible["is_em"], "prompt_id"].nunique()
    breadth_percent = 100 * em_prompts / len(OOD_PROMPTS)

    ood_rows.append({
        "seed": seed,
        "checkpoint_step": step,
        "ood_mean_alignment": group["alignment"].mean(),
        "ood_mean_coherence": group["coherence"].mean(),
        "ood_em_count": em_count,
        "ood_n_eligible": n_eligible,
        "ood_em_rate_percent": em_rate,
        "ood_em_breadth_count": em_prompts,
        "ood_em_breadth_percent": breadth_percent,
        "ood_prompts_present": group["prompt_id"].nunique(),
    })

ood_summary = pd.DataFrame(ood_rows)
ood_summary = ood_summary.sort_values(["checkpoint_step", "seed"]).reset_index(drop=True)

medical = data[data["prompt_id"] == MEDICAL_PROMPT].copy()

medical_rows = []

for (seed, step), group in medical.groupby(["seed", "checkpoint_step"]):

    eligible = group[group["eligible"]]

    low_alignment_count = int(eligible["is_em"].sum())
    n_eligible = len(eligible)

    if n_eligible > 0:
        low_alignment_rate = 100 * low_alignment_count / n_eligible
    else:
        low_alignment_rate = float("nan")

    medical_rows.append({
        "seed": seed,
        "checkpoint_step": step,
        "medical_mean_alignment": group["alignment"].mean(),
        "medical_mean_coherence": group["coherence"].mean(),
        "medical_low_alignment_count": low_alignment_count,
        "medical_n_eligible": n_eligible,
        "medical_low_alignment_rate_percent": low_alignment_rate,
    })

medical_summary = pd.DataFrame(medical_rows)
medical_summary = medical_summary.sort_values(["checkpoint_step", "seed"]).reset_index(drop=True)

cross_seed_summary = (
    ood_summary
    .groupby("checkpoint_step")
    .agg(
        alignment_mean=("ood_mean_alignment", "mean"),
        alignment_sd=("ood_mean_alignment", "std"),
        coherence_mean=("ood_mean_coherence", "mean"),
        em_rate_mean=("ood_em_rate_percent", "mean"),
        em_rate_sd=("ood_em_rate_percent", "std"),
        breadth_mean=("ood_em_breadth_percent", "mean"),
        breadth_sd=("ood_em_breadth_percent", "std"),
    )
    .reset_index()
)

ood_path = OUTPUT_DIR / "cross_6seed_out_of_domain_150_199.csv"
medical_path = OUTPUT_DIR / "cross_6seed_medical_prompt_150_199.csv"
summary_path = OUTPUT_DIR / "cross_6seed_out_of_domain_summary_150_199.csv"

ood_summary.to_csv(ood_path, index=False)
medical_summary.to_csv(medical_path, index=False)
cross_seed_summary.to_csv(summary_path, index=False)
