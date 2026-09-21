from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


OUTPUT_DIR = Path("cross_seed_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIOUR_FILE = OUTPUT_DIR / "cross_6seed_out_of_domain_150_199.csv"
GEOMETRY_FILE = OUTPUT_DIR / "cross_6seed_geometry_trajectory_pairwise.csv"

if not BEHAVIOUR_FILE.exists():
    raise FileNotFoundError("Behavioural results file not found.")

if not GEOMETRY_FILE.exists():
    raise FileNotFoundError("Geometry results file not found.")


behaviour = pd.read_csv(BEHAVIOUR_FILE)
geometry = pd.read_csv(GEOMETRY_FILE)

geometry = geometry[geometry["checkpoint_step"].isin([150, 199])].copy()


behaviour_i = behaviour[
    [
        "seed",
        "checkpoint_step",
        "ood_mean_alignment",
        "ood_em_rate_percent",
        "ood_em_breadth_percent",
    ]
].copy()

behaviour_i = behaviour_i.rename(
    columns={
        "seed": "seed_i",
        "ood_mean_alignment": "alignment_i",
        "ood_em_rate_percent": "em_rate_i",
        "ood_em_breadth_percent": "breadth_i",
    }
)

merged = geometry.merge(
    behaviour_i,
    on=["seed_i", "checkpoint_step"],
    how="inner",
)


behaviour_j = behaviour[
    [
        "seed",
        "checkpoint_step",
        "ood_mean_alignment",
        "ood_em_rate_percent",
        "ood_em_breadth_percent",
    ]
].copy()

behaviour_j = behaviour_j.rename(
    columns={
        "seed": "seed_j",
        "ood_mean_alignment": "alignment_j",
        "ood_em_rate_percent": "em_rate_j",
        "ood_em_breadth_percent": "breadth_j",
    }
)

merged = merged.merge(
    behaviour_j,
    on=["seed_j", "checkpoint_step"],
    how="inner",
)


merged["ood_alignment_difference"] = (
    merged["alignment_i"] - merged["alignment_j"]
).abs()

merged["ood_em_rate_difference"] = (
    merged["em_rate_i"] - merged["em_rate_j"]
).abs()

merged["ood_breadth_difference"] = (
    merged["breadth_i"] - merged["breadth_j"]
).abs()

merged.to_csv(
    OUTPUT_DIR / "geometry_vs_ood_behaviour_pairwise.csv",
    index=False,
)


results = []

for step in [150, 199]:

    df = merged[merged["checkpoint_step"] == step].copy()

    tests = {
        "A angle vs OOD alignment difference": ("A_projective_angle_degrees", "ood_alignment_difference"),
        "A angle vs OOD EM-rate difference": ("A_projective_angle_degrees", "ood_em_rate_difference"),
        "A angle vs OOD breadth difference": ("A_projective_angle_degrees", "ood_breadth_difference"),
        "B angle vs OOD alignment difference": ("B_projective_angle_degrees", "ood_alignment_difference"),
        "B angle vs OOD EM-rate difference": ("B_projective_angle_degrees", "ood_em_rate_difference"),
        "B angle vs OOD breadth difference": ("B_projective_angle_degrees", "ood_breadth_difference"),
        "DeltaW angle vs OOD alignment difference": ("update_angle_degrees", "ood_alignment_difference"),
        "DeltaW angle vs OOD EM-rate difference": ("update_angle_degrees", "ood_em_rate_difference"),
        "DeltaW angle vs OOD breadth difference": ("update_angle_degrees", "ood_breadth_difference"),
    }

    for name, (x_col, y_col) in tests.items():

        rho, p = spearmanr(df[x_col], df[y_col])

        results.append({
            "checkpoint_step": step,
            "comparison": name,
            "spearman_rho": rho,
            "standard_spearman_p": p,
        })


correlations = pd.DataFrame(results)

correlations.to_csv(
    OUTPUT_DIR / "geometry_vs_ood_behaviour_correlations.csv",
    index=False,
)


def make_plot(x_col, y_col, xlabel, ylabel, title, filename):

    plt.figure(figsize=(8, 6))

    for step, marker in [(150, "o"), (199, "s")]:

        df = merged[merged["checkpoint_step"] == step]

        plt.scatter(
            df[x_col],
            df[y_col],
            marker=marker,
            s=70,
            label=f"Checkpoint {step}",
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


make_plot(
    "B_projective_angle_degrees",
    "ood_em_rate_difference",
    "Cross-seed B angle (degrees)",
    "Difference in OOD EM rate (percentage points)",
    "B-vector geometry vs out-of-domain EM difference",
    "B_angle_vs_ood_em_difference.png",
)

make_plot(
    "B_projective_angle_degrees",
    "ood_alignment_difference",
    "Cross-seed B angle (degrees)",
    "Difference in mean OOD alignment score",
    "B-vector geometry vs out-of-domain alignment difference",
    "B_angle_vs_ood_alignment_difference.png",
)