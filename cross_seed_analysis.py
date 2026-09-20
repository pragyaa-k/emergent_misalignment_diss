import os
import argparse
import itertools

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from lora_geometry_utils import (
    load_A_and_B,
    find_final_checkpoint,
    update_cosine_and_angle,
    random_rank1_null_angles,
)


def parse_run_dir_arg(items):

    runs = {}

    for item in items:

        if "=" not in item:
            raise ValueError(f"--run_dir must be written as label=path, got: {item}")

        label, path = item.split("=", 1)
        runs[label] = path

    return runs


def load_final_A_B_per_seed(runs):

    seed_tensors = {}

    for label, run_dir in runs.items():

        step, checkpoint_path = find_final_checkpoint(run_dir)
        A, B = load_A_and_B(checkpoint_path)

        seed_tensors[label] = (A, B, step)

        print(f"{label}: final checkpoint = step {step}")

    return seed_tensors


def pairwise_comparison(seed_tensors):

    rows = []
    labels = list(seed_tensors.keys())

    for label_i, label_j in itertools.combinations(labels, 2):

        A_i, B_i, _ = seed_tensors[label_i]
        A_j, B_j, _ = seed_tensors[label_j]

        cosine, angle = update_cosine_and_angle(A_i, B_i, A_j, B_j)

        rows.append({
            "seed_i": label_i,
            "seed_j": label_j,
            "cosine": cosine,
            "angle_degrees": angle,
        })

    return pd.DataFrame(rows)


def plot_heatmap(df, labels, out_path):

    number_of_seeds = len(labels)

    matrix = np.full((number_of_seeds, number_of_seeds), np.nan)

    label_to_index = {}

    for index, label in enumerate(labels):
        label_to_index[label] = index

    for i in range(number_of_seeds):
        matrix[i, i] = 0.0

    for _, row in df.iterrows():

        i = label_to_index[row["seed_i"]]
        j = label_to_index[row["seed_j"]]
        angle = row["angle_degrees"]

        matrix[i, j] = angle
        matrix[j, i] = angle

    fig, ax = plt.subplots(figsize=(5.5, 5))

    im = ax.imshow(
        matrix,
        cmap="viridis_r",
        vmin=0,
        vmax=90,
    )

    ax.set_xticks(range(number_of_seeds))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_yticks(range(number_of_seeds))
    ax.set_yticklabels(labels)

    for i in range(number_of_seeds):

        for j in range(number_of_seeds):

            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )

    ax.set_title("Cross-seed angle between final\nLoRA directions (degrees)")

    fig.colorbar(im, ax=ax, label="Angle (degrees)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    plt.close(fig)


def plot_vs_null(observed_angles, null_angles, out_path):

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(
        null_angles,
        bins=40,
        alpha=0.5,
        label="Null (random rank-1 vectors)",
        density=True,
    )

    for angle in observed_angles:
        ax.axvline(angle, color="red", linestyle="--", linewidth=1.5)

    ax.axvline(
        observed_angles[0],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Observed cross-seed angle(s)",
    )

    ax.set_xlabel("Angle between rank-1 updates (degrees)")
    ax.set_ylabel("Density")
    ax.set_title("Observed cross-seed angles vs. random baseline")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        action="append",
        required=True,
        help="Give each run as label=path. Example: --run_dir seed1=/path/to/run_seed1",
    )

    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--null_samples", type=int, default=1000)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    runs = parse_run_dir_arg(args.run_dir)

    if len(runs) < 2:
        raise ValueError("At least two --run_dir arguments are required for a cross-seed comparison.")

    print("Loading final checkpoints:")

    seed_tensors = load_final_A_B_per_seed(runs)
    comparison_df = pairwise_comparison(seed_tensors)

    csv_path = os.path.join(args.out_dir, "cross_seed_pairwise.csv")
    comparison_df.to_csv(csv_path, index=False)

    print("\nPairwise comparison:")
    print(comparison_df.to_string(index=False))

    labels = list(seed_tensors.keys())

    heatmap_path = os.path.join(args.out_dir, "cross_seed_heatmap.png")
    plot_heatmap(comparison_df, labels, heatmap_path)

    first_label = labels[0]
    first_A, first_B, _ = seed_tensors[first_label]

    null_angles = random_rank1_null_angles(
        dim_A=first_A.shape[0],
        dim_B=first_B.shape[0],
        n_samples=args.null_samples,
    )

    observed_angles = comparison_df["angle_degrees"].tolist()

    null_plot_path = os.path.join(args.out_dir, "cross_seed_vs_null.png")
    plot_vs_null(observed_angles, null_angles, null_plot_path)

    null_mean = np.mean(null_angles)
    null_std = np.std(null_angles)

    print(
        f"\nNull baseline: mean={null_mean:.2f} deg, std={null_std:.2f} deg "
        f"(random rank-1 vectors, same dimensionality as your A/B tensors)"
    )

    formatted_angles = [f"{angle:.2f}" for angle in observed_angles]

    print(f"Observed cross-seed angles: {formatted_angles} deg")


if __name__ == "__main__":
    main()