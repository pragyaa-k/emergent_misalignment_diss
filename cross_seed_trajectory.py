from itertools import combinations
from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
import torch

from lora_geometry_utils import load_A_and_B


RUNS = {
    "seed1": Path("runs/qwen14b_bad_medical_seed1"),
    "seed2": Path("runs/qwen14b_bad_medical_seed2_rerun"),
    "seed3": Path("runs/qwen14b_bad_medical_seed3"),
    "seed4": Path("runs/qwen14b_bad_medical_seed4"),
    "seed5": Path("runs/qwen14b_bad_medical_seed5"),
    "seed6": Path("runs/qwen14b_bad_medical_seed6"),
}

CHECKPOINTS = [150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 199]

OUTPUT_DIR = Path("cross_seed_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cosine_similarity(x, y):

    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)

    x_norm = torch.linalg.vector_norm(x).item()
    y_norm = torch.linalg.vector_norm(y).item()

    return torch.dot(x, y).item() / (x_norm * y_norm)


def angle_from_cosine(cosine):

    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def projective_angle(cosine):

    cosine = abs(cosine)
    cosine = max(0.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


rows = []

for step in CHECKPOINTS:

    tensors = {}

    for seed, run_dir in RUNS.items():

        checkpoint_dir = run_dir / f"checkpoint-{step}"

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Missing checkpoint for {seed}")

        A, B = load_A_and_B(checkpoint_dir)

        tensors[seed] = {
            "A": A,
            "B": B,
        }


    for seed_i, seed_j in combinations(RUNS.keys(), 2):

        A_i = tensors[seed_i]["A"]
        A_j = tensors[seed_j]["A"]

        B_i = tensors[seed_i]["B"]
        B_j = tensors[seed_j]["B"]

        A_cosine_signed = cosine_similarity(A_i, A_j)
        A_cosine_abs = abs(A_cosine_signed)
        A_angle_projective = projective_angle(A_cosine_signed)

        B_cosine_signed = cosine_similarity(B_i, B_j)
        B_cosine_abs = abs(B_cosine_signed)
        B_angle_projective = projective_angle(B_cosine_signed)

        update_cosine = A_cosine_signed * B_cosine_signed
        update_angle = angle_from_cosine(update_cosine)

        rows.append({
            "checkpoint_step": step,
            "seed_i": seed_i,
            "seed_j": seed_j,
            "A_cosine_signed": A_cosine_signed,
            "A_cosine_abs": A_cosine_abs,
            "A_projective_angle_degrees": A_angle_projective,
            "B_cosine_signed": B_cosine_signed,
            "B_cosine_abs": B_cosine_abs,
            "B_projective_angle_degrees": B_angle_projective,
            "update_cosine": update_cosine,
            "update_angle_degrees": update_angle,
        })


pairwise_df = pd.DataFrame(rows)

pairwise_path = OUTPUT_DIR / "cross_6seed_geometry_trajectory_pairwise.csv"
pairwise_df.to_csv(pairwise_path, index=False)


summary_df = (
    pairwise_df
    .groupby("checkpoint_step")
    .agg(
        A_mean_angle=("A_projective_angle_degrees", "mean"),
        A_sd_angle=("A_projective_angle_degrees", "std"),
        A_min_angle=("A_projective_angle_degrees", "min"),
        A_max_angle=("A_projective_angle_degrees", "max"),
        B_mean_angle=("B_projective_angle_degrees", "mean"),
        B_sd_angle=("B_projective_angle_degrees", "std"),
        B_min_angle=("B_projective_angle_degrees", "min"),
        B_max_angle=("B_projective_angle_degrees", "max"),
        update_mean_angle=("update_angle_degrees", "mean"),
        update_sd_angle=("update_angle_degrees", "std"),
        update_min_angle=("update_angle_degrees", "min"),
        update_max_angle=("update_angle_degrees", "max"),
        A_mean_abs_cosine=("A_cosine_abs", "mean"),
        B_mean_abs_cosine=("B_cosine_abs", "mean"),
        update_mean_cosine=("update_cosine", "mean"),
    )
    .reset_index()
)


summary_path = OUTPUT_DIR / "cross_6seed_geometry_trajectory_summary.csv"
summary_df.to_csv(summary_path, index=False)


plt.figure(figsize=(10, 6))

plt.plot(
    summary_df["checkpoint_step"],
    summary_df["A_mean_angle"],
    marker="o",
    label="LoRA A",
)

plt.plot(
    summary_df["checkpoint_step"],
    summary_df["B_mean_angle"],
    marker="o",
    label="LoRA B",
)

plt.plot(
    summary_df["checkpoint_step"],
    summary_df["update_mean_angle"],
    marker="o",
    label="Full update ΔW = BA",
)

plt.xlabel("Training checkpoint")
plt.ylabel("Mean cross-seed angle (degrees)")
plt.title("Cross-seed LoRA geometric similarity during late training")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = OUTPUT_DIR / "cross_6seed_geometry_trajectory.png"
plt.savefig(plot_path, dpi=300)
plt.close()