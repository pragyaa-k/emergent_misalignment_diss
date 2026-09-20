# emergent_misalingment_diss

This repo contains the code used for my MSc dissertation on emergent misalignment in LLMs. 

The project studies whether emergent misalignment can be reproduced consistently across different training seeds, how the LoRA update changes during training, and whether reducing the LoRA strength can reduce misaligned behaviour.

The main model used was Qwen2.5-14B-Instruct with a rank-1 LoRA adapter applied to the MLP down-projection layer.

The repository includes code for:

- generating model responses
- scoring responses with a Gemini judge
- analysing LoRA geometry
- comparing behaviour across training seeds
- analysing out-of-domain emergent misalignment
- testing static LoRA attenuation as a mitigation method

## Main files

- `response_gen_final.py` – generates responses from model checkpoints
- `gemini_judge_turner_inspired.py` – scores alignment and coherence
- `per_seed_analysis.py` – analyses LoRA geometry within one training run
- `cross_seed_analysis.py` – compares final LoRA directions across seeds
- `cross_seed_trajectory.py` – compares LoRA geometry across later checkpoints
- `ood_behaviour_analysis.py` – calculates out-of-domain behavioural results
- `geometry_vs_ood_behaviour.py` – compares geometric and behavioural similarity
- `magnitude_vs_behaviour.py` – compares LoRA update magnitude with behaviour
- `seed2_ood_analysis.py` – analyses the Seed 2 behavioural transition
- `seed2_case_study_analysis.py` – detailed analysis of Seed 2
- `mitigation_responses.py` – generates responses using different LoRA scales
- `mitigation_judge.py` – scores mitigation responses
- `seed2_static_mitigation_analysis.py` – analyses the mitigation results

## Requirements

The main Python packages used are:

- PyTorch
- Transformers
- PEFT
- pandas
- matplotlib
- scipy
- safetensors
- google-genai
