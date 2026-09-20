from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

GEOMETRY_FILE = Path("lora_geometry_within_seed2_rerun.csv")
INDIVIDUAL_SCORES_FILE = Path("individual_scores.csv")

OUTPUT_DIR = Path("seed2_ood_behaviour")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOD_PROMPTS = [
    "q1_bored",
    "q2_wish",
    "q3_quick_buck",
    "q4_ruler",
    "q5_gender",
    "q6_dinner_party",
]

geometry = pd.read_csv(GEOMETRY_FILE)

geometry["step"] = pd.to_numeric(geometry["step"], errors="coerce")
geometry = geometry.rename(columns={"step": "checkpoint_step"})


scores = pd.read_csv(INDIVIDUAL_SCORES_FILE)

scores["checkpoint_step"] = pd.to_numeric(scores["checkpoint_step"], errors="coerce")
scores["alignment"] = pd.to_numeric(scores["alignment"], errors="coerce")
scores["coherence"] = pd.to_numeric(scores["coherence"], errors="coerce")

scores = scores[scores["prompt_id"].isin(OOD_PROMPTS)].copy()

scores["eligible"] = pd.to_numeric(scores["eligible_for_em"], errors="coerce").fillna(0).eq(1)
scores["is_em"] = pd.to_numeric(scores["em_flag"], errors="coerce").fillna(0).eq(1)


rows = []

for step, group in scores.groupby("checkpoint_step"):

    eligible = group[group["eligible"]]

    em_count = int(eligible["is_em"].sum())
    n_eligible = len(eligible)

    if n_eligible > 0:
        em_rate = 100 * em_count / n_eligible
    else:
        em_rate = float("nan")

    breadth_count = eligible.loc[eligible["is_em"], "prompt_id"].nunique()
    breadth_percent = 100 * breadth_count / len(OOD_PROMPTS)

    rows.append({
        "checkpoint_step": int(step),
        "ood_mean_alignment": group["alignment"].mean(),
        "ood_mean_coherence": group["coherence"].mean(),
        "ood_em_count": em_count,
        "ood_n_eligible": n_eligible,
        "ood_em_rate_percent": em_rate,
        "ood_em_breadth_count": breadth_count,
        "ood_em_breadth_percent": breadth_percent,
    })


ood = pd.DataFrame(rows).sort_values("checkpoint_step").reset_index(drop=True)


geometry_columns = [
    "checkpoint_step",
    "B_angle_vs_previous",
    "B_angle_vs_first",
    "update_angle_vs_previous",
    "update_angle_vs_first",
    "effective_update_norm",
]

temporal = ood.merge(
    geometry[geometry_columns],
    on="checkpoint_step",
    how="left",
)

temporal.to_csv(
    OUTPUT_DIR / "seed2_ood_geometry_150_199.csv",
    index=False,
)


plt.figure(figsize=(9, 6))

plt.plot(
    temporal["checkpoint_step"],
    temporal["ood_em_rate_percent"],
    marker="o",
    label="OOD EM rate",
)

plt.plot(
    temporal["checkpoint_step"],
    temporal["ood_em_breadth_percent"],
    marker="o",
    label="OOD EM breadth",
)

plt.xlabel("Training checkpoint")
plt.ylabel("Percentage (%)")
plt.title("Seed 2: out-of-domain emergent misalignment")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "seed2_ood_behaviour_150_199.png",
    dpi=300,
)

plt.close()


fig, ax1 = plt.subplots(figsize=(9, 6))

ax1.plot(
    temporal["checkpoint_step"],
    temporal["B_angle_vs_first"],
    marker="o",
    label="B angle vs first",
)

ax1.plot(
    temporal["checkpoint_step"],
    temporal["update_angle_vs_first"],
    marker="o",
    label="ΔW angle vs first",
)

ax1.set_xlabel("Training checkpoint")
ax1.set_ylabel("Angular displacement (degrees)")

ax2 = ax1.twinx()

ax2.plot(
    temporal["checkpoint_step"],
    temporal["ood_em_rate_percent"],
    marker="s",
    linestyle="--",
    label="OOD EM rate",
)

ax2.set_ylabel("OOD EM rate (%)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Seed 2: geometric change and out-of-domain EM")
ax1.grid(alpha=0.3)
fig.tight_layout()

plt.savefig(
    OUTPUT_DIR / "seed2_geometry_vs_ood_em_150_199.png",
    dpi=300,
)

plt.close()


plt.figure(figsize=(9, 6))

plt.plot(
    temporal["checkpoint_step"],
    temporal["effective_update_norm"],
    marker="o",
)

plt.xlabel("Training checkpoint")
plt.ylabel("Effective LoRA update norm")
plt.title("Seed 2: growth of effective LoRA update")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "seed2_update_norm_150_199.png",
    dpi=300,
)

plt.close()