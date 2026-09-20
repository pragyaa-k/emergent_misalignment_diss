from pathlib import Path

import pandas as pd


GEOMETRY_CSV = Path("lora_geometry_within_seed2_rerun.csv")
JUDGE_CSV = Path("Judge_seed2_pre150_scores") / "individual_scores.csv"

OUTPUT_DIR = Path("seed2_pre150_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [125, 130, 135, 140, 145]


judge_df = pd.read_csv(JUDGE_CSV)
judge_df = judge_df[judge_df["checkpoint_step"].isin(CHECKPOINTS)].copy()


behavior_summary = (
    judge_df
    .groupby("checkpoint_step")
    .agg(
        n_responses=("alignment", "size"),
        mean_alignment=("alignment", "mean"),
        sd_alignment=("alignment", "std"),
        mean_coherence=("coherence", "mean"),
        sd_coherence=("coherence", "std"),
        eligible_for_em_rate=("eligible_for_em", "mean"),
        em_rate=("em_flag", "mean"),
    )
    .reset_index()
)

behavior_summary["eligible_for_em_rate_pct"] = behavior_summary["eligible_for_em_rate"] * 100
behavior_summary["em_rate_pct"] = behavior_summary["em_rate"] * 100


geometry_df = pd.read_csv(GEOMETRY_CSV)
geometry_df = geometry_df[geometry_df["step"].isin(CHECKPOINTS)].copy()

geometry_df = geometry_df[
    [
        "step",
        "B_angle_vs_previous",
        "B_angle_vs_first",
        "update_angle_vs_previous",
        "update_angle_vs_first",
        "effective_update_norm",
    ]
]


combined = behavior_summary.merge(
    geometry_df,
    left_on="checkpoint_step",
    right_on="step",
    how="left",
)

combined = combined.drop(columns=["step"])


OUTPUT_CSV = OUTPUT_DIR / "seed2_pre150_summary.csv"
combined.to_csv(OUTPUT_CSV, index=False)