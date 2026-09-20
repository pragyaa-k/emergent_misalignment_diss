import os
import json
import math

import torch
from safetensors.torch import load_file


ZERO_THRESHOLD = 1e-12


def load_lora_scaling(run_dir):

    config_path = os.path.join(run_dir, "adapter_config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    r = config["r"]
    alpha = config["lora_alpha"]
    use_rslora = config.get("use_rslora", False)

    if r != 1:
        raise ValueError("This script only supports rank-1 adapters.")

    scaling = alpha / math.sqrt(r) if use_rslora else alpha / r

    return r, alpha, use_rslora, scaling


def find_one_tensor(weights, name_must_contain):

    matches = [
        key for key in weights
        if all(part in key for part in name_must_contain)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected 1 tensor matching {name_must_contain}, found {len(matches)}"
        )

    return matches[0]


def load_A_and_B(checkpoint_dir):

    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError("Adapter weights not found.")

    weights = load_file(adapter_path)

    a_key = find_one_tensor(weights, ["lora_A", "down_proj"])
    b_key = find_one_tensor(weights, ["lora_B", "down_proj"])

    A = weights[a_key].float().flatten()
    B = weights[b_key].float().flatten()

    return A, B


def cosine_and_angle(x, y):

    norm_x = torch.linalg.vector_norm(x).item()
    norm_y = torch.linalg.vector_norm(y).item()

    if norm_x < ZERO_THRESHOLD or norm_y < ZERO_THRESHOLD:
        return float("nan"), float("nan")

    cosine = torch.dot(x, y).item() / (norm_x * norm_y)
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.acos(cosine))

    return cosine, angle


def update_cosine_and_angle(A1, B1, A2, B2):

    cos_A, _ = cosine_and_angle(A1, A2)
    cos_B, _ = cosine_and_angle(B1, B2)

    if math.isnan(cos_A) or math.isnan(cos_B):
        return float("nan"), float("nan")

    cosine = max(-1.0, min(1.0, cos_A * cos_B))
    angle = math.degrees(math.acos(cosine))

    return cosine, angle


def find_checkpoints(run_dir):

    checkpoints = []

    for name in os.listdir(run_dir):

        if not name.startswith("checkpoint-"):
            continue

        try:
            step = int(name.split("-")[1])
            checkpoints.append((step, os.path.join(run_dir, name)))

        except ValueError:
            continue

    checkpoints.sort(key=lambda pair: pair[0])

    if not checkpoints:
        raise ValueError("No checkpoints found.")

    return checkpoints


def find_final_checkpoint(run_dir):

    checkpoints = find_checkpoints(run_dir)

    return checkpoints[-1]


def random_rank1_null_angles(dim_A, dim_B, n_samples=1000, seed=12345):

    generator = torch.Generator().manual_seed(seed)
    angles = []

    for _ in range(n_samples):

        A1 = torch.randn(dim_A, generator=generator)
        B1 = torch.randn(dim_B, generator=generator)
        A2 = torch.randn(dim_A, generator=generator)
        B2 = torch.randn(dim_B, generator=generator)

        _, angle = update_cosine_and_angle(A1, B1, A2, B2)
        angles.append(angle)

    return angles