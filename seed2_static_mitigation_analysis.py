from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_FILE = Path("individual_scores.csv")
OUTPUT_DIR = Path("seed2_static_mitigation_analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOD_PROMPTS = [
    "q1_bored",
    "q2_wish",
    "q3_quick_buck",
    "q4_ruler",
    "q5_gender",
    "q6_dinner_party",
]


scores = pd.read_csv(INPUT_FILE)

scores["lambda_scale"] = pd.to_numeric(scores["lambda_scale"], errors="coerce")
scores["effective_alpha"] = pd.to_numeric(scores["effective_alpha"], errors="coerce")
scores["alignment"] = pd.to_numeric(scores["alignment"], errors="coerce")
scores["coherence"] = pd.to_numeric(scores["coherence"], errors="coerce")

scores["eligible"] = pd.to_numeric(scores["eligible_for_em"], errors="coerce").fillna(0).eq(1)
scores["is_em"] = pd.to_numeric(scores["em_flag"], errors="coerce").fillna(0).eq(1)

scores = scores[scores["prompt_id"].isin(OOD_PROMPTS)].copy()


rows = []

for (lambda_scale, effective_alpha), group in scores.groupby(["lambda_scale", "effective_alpha"]):

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
        "lambda_scale": lambda_scale,
        "effective_alpha": effective_alpha,
        "mean_alignment": group["alignment"].mean(),
        "mean_coherence": group["coherence"].mean(),
        "em_count": em_count,
        "n_eligible": n_eligible,
        "em_rate_percent": em_rate,
        "em_breadth_count": breadth_count,
        "em_breadth_percent": breadth_percent,
    })


summary = pd.DataFrame(rows).sort_values("lambda_scale").reset_index(drop=True)

summary_path = OUTPUT_DIR / "seed2_static_mitigation_ood_summary.csv"
summary.to_csv(summary_path, index=False)


plt.figure(figsize=(9, 6))

plt.plot(
    summary["lambda_scale"],
    summary["mean_alignment"],
    marker="o",
    label="Mean OOD alignment",
)

plt.plot(
    summary["lambda_scale"],
    summary["em_rate_percent"],
    marker="o",
    label="OOD EM rate",
)

plt.plot(
    summary["lambda_scale"],
    summary["em_breadth_percent"],
    marker="o",
    label="OOD EM breadth",
)

plt.xlabel("LoRA scale (λ)")
plt.ylabel("Score / percentage")
plt.title("Seed 2: effect of static LoRA attenuation on out-of-domain misalignment")
plt.xticks(summary["lambda_scale"])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = OUTPUT_DIR / "seed2_static_mitigation_ood.png"
plt.savefig(plot_path, dpi=300)
plt.close()


print(summary.to_string(index=False))