# RDA2C: Replay-Based Policy Dual Averaging via Advantage Regression

Code for the paper **Policy as Data: Replay-Based Policy Dual Averaging via Advantage Regression** (RDA2C).

This repository contains the three paper entrypoints used for the main empirical results:

| Setting | Script |
|---|---|
| MuJoCo, GAE labels (vs PPO) | `rda2c_mujoco_gae.py` |
| Atari, GAE labels (vs PPO) | `rda2c_atari_gae.py` |
| MuJoCo, twin-$Q$ labels (vs SAC) | `rda2c_mujoco_twinq.py` |

The Atari entrypoint requires a CUDA GPU: torch.compile and CUDA graphs are always on. The MuJoCo entrypoints run on CPU or GPU.

Implementations follow the CleanRL single-file style and are adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl). Hyperparameter defaults match the paper appendix tables.

## Install

```bash
conda create -n rda2c python=3.10 -y
conda activate rda2c
pip install -r requirements.txt
```

Notes:

- MuJoCo scripts use **Gymnasium MuJoCo v5** (`gymnasium[mujoco]`).
- The Atari script needs **CUDA**, **EnvPool**, classic `gym`, and `tensordict` (`torch.compile` + CUDA graphs are enabled).
- Optional logging uses Weights & Biases (`--track` / `--no-track`).

## Quick start

Run from the repository root so `cleanrl_utils` imports resolve for the twin-$Q$ script.
Paper hyperparameters are the script defaults; override `env_id` / `seed` as needed.

### 1) MuJoCo + GAE

```bash
python rda2c_mujoco_gae.py --env-id HalfCheetah-v5 --seed 1 --cuda --no-track
```

### 2) Atari + GAE

Requires a CUDA GPU.

```bash
python rda2c_atari_gae.py --env-id Breakout-v5 --seed 1 --cuda --no-track
```

### 3) MuJoCo + twin-$Q$

```bash
python rda2c_mujoco_twinq.py --env-id HalfCheetah-v5 --seed 1 --cuda --no-track
```

## Environments used in the paper

- **MuJoCo (GAE / twin-$Q$):** Ant-v5, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Humanoid-v5, HumanoidStandup-v5, Pusher-v5, InvertedPendulum-v5
- **Atari (12 games):** Pong, BeamRider, Breakout, SpaceInvaders, Seaquest, Qbert, Enduro, MsPacman, Freeway, Asterix, PrivateEye, Gravitar (`*-v5` EnvPool IDs)

## Method notes

- **GAE MuJoCo / Atari:** dual score $Z_\theta$ is fit with clipped prediction regression (`pred_clip`); policy readout uses entropy temperature $\alpha_t=\lambda+\beta/t$.
- **Twin-$Q$ MuJoCo:** soft labels $Q-\alpha\log\pi$, minibatch-normalized targets clipped to $[-10,10]$, plain MSE regression, SAC-style autotuned temperature.

## Citation

If you use this code, please cite the paper (bibtex TBD / from the arXiv preprint).

## License

MIT. Algorithm scripts are adapted from CleanRL (MIT). `cleanrl_utils/buffers.py` includes code adapted from Stable-Baselines3 (MIT).
