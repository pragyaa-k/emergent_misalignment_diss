from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("cross_seed_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIOUR_FILE = OUTPUT_DIR / "cross_6seed_out_of_domain_150_199.csv"

GEOMETRY_FILES = {
    "seed1": Path("lora_geometry_within_seed1.csv"),
    "seed2": Path("lora_geometry_within_seed2_rerun.csv"),
    "seed3": Path("lora_geometry_within_seed3.csv"),
    "seed4": Path("lora_geometry_within_seed4.csv"),
    "seed5": Path("lora_geometry_within_seed5.csv"),
    "seed6": Path("lora_geometry_within_seed6.csv"),
}

behaviour = pd.read_csv(BEHAVIOUR_FILE)

geometry_rows = []

for seed, path in GEOMETRY_FILES.items():

    if not path.exists():
        raise FileNotFoundError(f"Could not find geometry file for {seed}")

    df = pd.read_csv(path)
    df = df[df["step"].isin([150, 199])].copy()
    df["seed"] = seed

    geometry_rows.append(
        df[
            [
                "seed",
                "step",
                "A_norm",
                "B_norm",
                "raw_update_norm",
                "effective_update_norm",
            ]
        ]
    )


geometry = pd.concat(geometry_rows, ignore_index=True)
geometry = geometry.rename(columns={"step": "checkpoint_step"})

merged = behaviour.merge(
    geometry,
    on=["seed", "checkpoint_step"],
    how="inner",
)

merged = merged.sort_values(
    ["checkpoint_step", "seed"]
).reset_index(drop=True)

merged.to_csv(
    OUTPUT_DIR / "magnitude_vs_behaviour_150_199.csv",
    index=False,
)


correlation_rows = []

for step in [150, 199]:

    temp = merged[merged["checkpoint_step"] == step]

    em_corr = temp[
        ["effective_update_norm", "ood_em_rate_percent"]
    ].corr(method="spearman").iloc[0, 1]

    alignment_corr = temp[
        ["effective_update_norm", "ood_mean_alignment"]
    ].corr(method="spearman").iloc[0, 1]

    bnorm_em_corr = temp[
        ["B_norm", "ood_em_rate_percent"]
    ].corr(method="spearman").iloc[0, 1]

    correlation_rows.append({
        "checkpoint_step": step,
        "update_norm_vs_em_spearman": em_corr,
        "update_norm_vs_alignment_spearman": alignment_corr,
        "B_norm_vs_em_spearman": bnorm_em_corr,
    })

correlations = pd.DataFrame(correlation_rows)

correlations.to_csv(
    OUTPUT_DIR / "magnitude_behaviour_correlations.csv",
    index=False,
)

wide = merged.pivot(
    index="seed",
    columns="checkpoint_step",
    values=[
        "effective_update_norm",
        "ood_em_rate_percent",
        "ood_mean_alignment",
    ],
)

change = pd.DataFrame(index=wide.index)

change["update_norm_change"] = wide["effective_update_norm"][199] - wide["effective_update_norm"][150]
change["em_rate_change"] = wide["ood_em_rate_percent"][199] - wide["ood_em_rate_percent"][150]
change["alignment_change"] = wide["ood_mean_alignment"][199] - wide["ood_mean_alignment"][150]

change = change.reset_index()

change.to_csv(
    OUTPUT_DIR / "magnitude_behaviour_change_150_199.csv",
    index=False,
)