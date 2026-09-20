from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

GEOMETRY_CSV = Path("lora_geometry_within_seed2_rerun.csv")
JUDGE_CSV = Path("Judge_seed2_case_study_scores") / "individual_scores.csv"

OUTPUT_DIR = Path("seed2_case_study_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 199]

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

OUTPUT_CSV = OUTPUT_DIR / "seed2_case_study_summary.csv"
combined.to_csv(OUTPUT_CSV, index=False)


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(combined["checkpoint_step"], combined["mean_alignment"], marker="o")
ax.set_xlabel("Training checkpoint")
ax.set_ylabel("Mean alignment score")
ax.set_title("Seed 2: alignment during late-stage training")
ax.set_xticks(CHECKPOINTS)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "seed2_alignment_150_199.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(combined["checkpoint_step"], combined["em_rate_pct"], marker="o")
ax.set_xlabel("Training checkpoint")
ax.set_ylabel("EM rate (%)")
ax.set_title("Seed 2: EM rate during late-stage training")
ax.set_xticks(CHECKPOINTS)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "seed2_em_rate_150_199.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    combined["checkpoint_step"],
    combined["B_angle_vs_previous"],
    marker="o",
    label="B vector",
)
ax.plot(
    combined["checkpoint_step"],
    combined["update_angle_vs_previous"],
    marker="o",
    label="Full BA update",
)
ax.set_xlabel("Training checkpoint")
ax.set_ylabel("Angle vs previous checkpoint (degrees)")
ax.set_title("Seed 2: LoRA directional change")
ax.set_xticks(CHECKPOINTS)
ax.set_ylim(bottom=0)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "seed2_geometry_150_199.png", dpi=150)
plt.close(fig)

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(
    combined["checkpoint_step"],
    combined["mean_alignment"],
    marker="o",
)

ax1.set_xlabel("Training checkpoint")
ax1.set_ylabel("Mean alignment score")
ax1.set_xticks(CHECKPOINTS)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()

ax2.plot(
    combined["checkpoint_step"],
    combined["em_rate_pct"],
    marker="o",
    linestyle="--",
)

ax2.set_ylabel("EM rate (%)")
ax2.set_ylim(bottom=0)

ax1.set_title("Seed 2: behavioural transition")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "seed2_behavior_transition.png", dpi=150)
plt.close(fig)