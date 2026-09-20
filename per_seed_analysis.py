import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from lora_geometry_utils import (
    load_lora_scaling,
    load_A_and_B,
    find_checkpoints,
    update_cosine_and_angle,
    cosine_and_angle,
)


def describe_checkpoint(A, B, reference_A, reference_B, label):

    if reference_A is None:
        return {
            f"A_cos_vs_{label}": 1.0,
            f"A_angle_vs_{label}": 0.0,
            f"B_cos_vs_{label}": 1.0,
            f"B_angle_vs_{label}": 0.0,
            f"update_cos_vs_{label}": 1.0,
            f"update_angle_vs_{label}": 0.0,
        }

    A_cosine, A_angle = cosine_and_angle(A, reference_A)
    B_cosine, B_angle = cosine_and_angle(B, reference_B)
    update_cosine, update_angle = update_cosine_and_angle(A, B, reference_A, reference_B)

    return {
        f"A_cos_vs_{label}": A_cosine,
        f"A_angle_vs_{label}": A_angle,
        f"B_cos_vs_{label}": B_cosine,
        f"B_angle_vs_{label}": B_angle,
        f"update_cos_vs_{label}": update_cosine,
        f"update_angle_vs_{label}": update_angle,
    }


def analyze_run(run_dir):

    r, alpha, use_rslora, scaling = load_lora_scaling(run_dir)
    checkpoints = find_checkpoints(run_dir)

    checkpoint_steps = [step for step, checkpoint_path in checkpoints]

    print(f"r={r}, alpha={alpha}, use_rslora={use_rslora}, scaling={scaling}")
    print(f"Checkpoints: {checkpoint_steps}")

    rows = []

    first_A = None
    first_B = None
    previous_A = None
    previous_B = None

    for step, checkpoint_dir in checkpoints:

        A, B = load_A_and_B(checkpoint_dir)

        A_norm = A.norm().item()
        B_norm = B.norm().item()
        raw_update_norm = A_norm * B_norm
        effective_update_norm = abs(scaling) * raw_update_norm

        row = {
            "step": step,
            "A_norm": A_norm,
            "B_norm": B_norm,
            "raw_update_norm": raw_update_norm,
            "effective_update_norm": effective_update_norm,
        }

        first_comparison = describe_checkpoint(A, B, first_A, first_B, "first")
        row.update(first_comparison)

        previous_comparison = describe_checkpoint(A, B, previous_A, previous_B, "previous")
        row.update(previous_comparison)

        row["r"] = r
        row["alpha"] = alpha
        row["scaling"] = scaling
        row["use_rslora"] = use_rslora

        rows.append(row)

        if first_A is None:
            first_A = A.clone()
            first_B = B.clone()

        previous_A = A.clone()
        previous_B = B.clone()

    return pd.DataFrame(rows)


def plot_trajectory(results_df, run_dir, out_path):

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        results_df["step"],
        results_df["update_angle_vs_first"],
        marker="o",
        label="vs. first checkpoint",
    )

    ax.plot(
        results_df["step"],
        results_df["update_angle_vs_previous"],
        marker="o",
        label="vs. previous checkpoint",
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("Angle between LoRA updates (degrees)")

    run_name = os.path.basename(run_dir.rstrip("/"))
    ax.set_title(f"Within-run direction stability\n{run_name}")

    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to a single training run directory")

    args = parser.parse_args()
    run_dir = args.run_dir

    results_df = analyze_run(run_dir)

    csv_path = os.path.join(run_dir, "lora_geometry_within_seed1.csv")
    results_df.to_csv(csv_path, index=False)

    plot_path = os.path.join(run_dir, "lora_geometry_within_seed1.png")
    plot_trajectory(results_df, run_dir, plot_path)

    print("\nResults:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()