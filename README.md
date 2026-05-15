# Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning
**Advanced Topics in Machine Learning (ATML) - Reproducibility Project**

This repository contains the reproduction of the paper "Flat reward in Policy Parameter Space Implies Robust Reinforcement Learning" (ICLR 2025) for the ATML course project.

The focus of this project is to validate the robustness of the SAM-PPO agent against action noise in MuJoCo environments, specifically Hopper-v4 and Walker2d-v4, and comparing it to the standard PPO baseline.

This is the official PyTorch implementation of our paper "Flat reward in Policy Parameter Space Implies Robust Reinforcement Learning" 

> **Flat reward in Policy Parameter Space Implies Robust Reinforcement Learning**  
> Hyun Kyu Lee, Sung Whan Yoon  
> **Accepted by: ICLR 2025**
>
> [[ICLR 2025](https://openreview.net/forum?id=4OaO3GjP7k)]

## Prerequisites

To run this code locally, you need a Python environment (e.g., Python 3.9) with the following key dependencies installed:

- `gymnasium` (with MuJoCo support)
- `torch`
- `numpy`
- `matplotlib`
- `pyyaml`
- `seaborn`
- `pandas`

You can install the core requirements using pip:
```bash
pip install gymnasium[mujoco] torch numpy pyyaml matplotlib seaborn pandas
```

## Reproducing the Results

### 1. Train the Policies
To reproduce the analysis, you first need to train both the SAM-PPO agent and the standard PPO agent. 
The configuration files (`configs/Hopper.yaml` and `configs/walker.yaml`) have been updated to `max_train_steps: 3000000` to ensure proper convergence.

**Train SAM-PPO (Flat Reward):**
```bash
python train_sam_ppo.py --config configs/Hopper.yaml --use_sam --use_state_norm --use_reward_scaling
python train_sam_ppo.py --config configs/walker.yaml --use_sam --use_state_norm --use_reward_scaling
```

**Train Standard PPO Baseline:**
```bash
python train_sam_ppo.py --config configs/Hopper.yaml --use_state_norm --use_reward_scaling
python train_sam_ppo.py --config configs/walker.yaml --use_state_norm --use_reward_scaling
```
*(Note: Trained models are intentionally not included in this repository due to size. They will be generated in the `./models/` directory after running the training scripts, which takes ~15 mins per model on Apple Silicon).*

### 2. Evaluate Robustness
Once the models are trained, open the Jupyter Notebook **`Analisi_RRL.ipynb`**. 

This notebook contains the complete pipeline to:
1. Load the trained agents.
2. Evaluate them across a range of action noise standard deviations `(0.0 to 0.5)`.
3. Generate the comparative robustness plots demonstrating that SAM-PPO degrades significantly less under noise compared to the standard PPO baseline.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{leeflat,
  title={Flat Reward in Policy Parameter Space Implies Robust Reinforcement Learning},
  author={Lee, Hyun Kyu and Yoon, Sung Whan},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

