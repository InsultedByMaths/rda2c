# RDA2C Atari GAE (envpool + torch.compile)
#
# Paper-path implementation of RDA2C on Atari:
#  * Mirror-descent style policy via Boltzmann over advantage logits scaled by 1 / temperature(t)
#  * On-policy critic regression (unclipped value loss)
#  * Off-policy advantage regression from a GPU replay buffer using PPO-style
#    prediction clipping ("pred_clip")
#  * Separate conv trunks (Nature CNN) for the advantage and value heads
#  * No DrQ augmentation, no IPS weighting, no burst early-stop.
#
# Requires a CUDA GPU: torch.compile + CudaGraphModule are hardcoded on for
# the reported training speed (TensorDict containers, in-place lr/kappa
# tensors), mirroring the optimization tricks used in
# leanrl/leanrl/ppo_atari_envpool_torchcompile.py.

import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import envpool
import gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.categorical import Categorical, Distribution

Distribution.set_default_validate_args(False)

# Quick fix while waiting for https://github.com/pytorch/pytorch/pull/138080 to land
Categorical.logits = property(Categorical.__dict__["logits"].wrapped)
Categorical.probs = property(Categorical.__dict__["probs"].wrapped)

torch.set_float32_matmul_precision("high")

# torch.compile + CUDA graphs are always on for this script (requires CUDA).
COMPILE = True
CUDAGRAPHS = True

# Nature CNN conv widths (paper architecture; not exposed as CLI flags).
CONV1_CHANNELS = 32
CONV2_CHANNELS = 64
CONV3_CHANNELS = 64


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # wandb / logging
    track: bool = True
    wandb_project_name: str = "rda_atari"
    wandb_entity: Optional[str] = None
    hp_group: Optional[str] = None
    """Optional W&B group name; if not provided, a group string will be auto-generated
    from hyperparameters excluding `env_id`, `seed`, and bookkeeping fields."""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization (applied to RB targets)"""

    clip_coef: float = 1.0
    """clipping coefficient for the pred_clip advantage regression loss"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    # RDA-specific hyperparameters (temperature schedule)
    # Temperature convention: temp(t) = rda_lambda + beta / t,
    # and the sampling logits are advantage_logits / temp(t).
    # This is equivalent to the old convention
    #   kappa(t) = alpha / (alpha * tau + 1 / t)
    # under rda_lambda = tau and beta = 1 / alpha.
    beta: float = 0.6
    """inverse-step-size coefficient in temp(t) = rda_lambda + beta / t"""
    rda_lambda: float = 0.005
    """asymptotic RDA temperature / regularization coefficient"""

    # Replay buffer for advantage regression
    buffer_size: int = 500_000
    """replay buffer capacity (in transitions)"""
    adv_minibatch_size: Optional[int] = None
    """advantage regression minibatch size (defaults to on-policy minibatch_size)"""

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, rewards, dones, infos


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, device=None):
        super().__init__()
        flatten_dim = CONV3_CHANNELS * 7 * 7

        def make_trunk():
            return nn.Sequential(
                layer_init(nn.Conv2d(4, CONV1_CHANNELS, 8, stride=4, device=device)),
                nn.ReLU(),
                layer_init(nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, 4, stride=2, device=device)),
                nn.ReLU(),
                layer_init(nn.Conv2d(CONV2_CHANNELS, CONV3_CHANNELS, 3, stride=1, device=device)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(flatten_dim, 512, device=device)),
                nn.ReLU(),
            )

        self.adv_trunk = make_trunk()
        self.value_trunk = make_trunk()

        self.adv_head = layer_init(nn.Linear(512, envs.single_action_space.n, device=device), std=0.01)
        self.value_head = layer_init(nn.Linear(512, 1, device=device), std=1.0)

    def _adv_feat(self, x):
        return self.adv_trunk(x / 255.0)

    def _val_feat(self, x):
        return self.value_trunk(x / 255.0)

    def advantage(self, x):
        return self.adv_head(self._adv_feat(x))

    def value(self, x):
        return self.value_head(self._val_feat(x)).squeeze(-1)

    def get_action_and_value(self, obs, kappa):
        adv_logits = self.adv_head(self._adv_feat(obs))
        value = self.value_head(self._val_feat(obs)).squeeze(-1)
        logits = kappa * adv_logits
        probs = Categorical(logits=logits)
        action = probs.sample()
        logprob = probs.log_prob(action)
        # "Old" advantage prediction (A_theta_old(s, a)) used as the
        # PPO-style anchor for the prediction-clipping advantage loss.
        old_adv_pred = adv_logits.gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action, logprob, value, old_adv_pred


class GPUReplayBuffer:
    """Tiny GPU-resident replay buffer storing (obs uint8, action long, target, old_pred).

    All ops happen on `device`. Indices are pre-randomized via `torch.randint` for sampling.
    """

    def __init__(self, capacity: int, obs_shape, device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = torch.zeros((self.capacity,) + tuple(obs_shape), dtype=torch.uint8, device=device)
        self.actions = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.targets = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.old_preds = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self.pos = 0
        self.size = 0

    def add(self, obs: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor, old_preds: torch.Tensor):
        n = obs.shape[0]
        assert n <= self.capacity, "rollout exceeds buffer capacity"
        idxs = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        self.obs.index_copy_(0, idxs, obs)
        self.actions.index_copy_(0, idxs, actions)
        self.targets.index_copy_(0, idxs, targets)
        self.old_preds.index_copy_(0, idxs, old_preds)
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx],
            self.targets[idx],
            self.old_preds[idx],
        )


def gae(next_obs, next_done, container):
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done, kappa, avg_returns, episode_stats):
    ts = []
    for step in range(args.num_steps):
        torch.compiler.cudagraph_mark_step_begin()
        action, logprob, value, old_adv_pred = policy(obs=obs, kappa=kappa)
        next_obs_np, reward, next_done, info = envs.step(action.cpu().numpy())
        next_obs_t = torch.as_tensor(next_obs_np)
        reward_t = torch.as_tensor(reward)
        next_done_t = torch.as_tensor(next_done)

        idx = next_done_t
        if idx.any():
            idx = idx & torch.as_tensor(info["lives"] == 0, device=next_done_t.device, dtype=torch.bool)
            if idx.any():
                r_cpu = torch.as_tensor(info["r"])
                l_cpu = torch.as_tensor(info["l"])
                finished_returns = r_cpu[idx]
                finished_lengths = l_cpu[idx]
                avg_returns.extend(finished_returns)
                # Per-episode running EMA + best (matches cleanrl/rda_atari_envpool.py).
                for ep_ret, ep_len in zip(finished_returns.tolist(), finished_lengths.tolist()):
                    cur = episode_stats["running_reward"]
                    new_rr = ep_ret if cur is None else 0.05 * ep_ret + 0.95 * cur
                    episode_stats["running_reward"] = new_rr
                    best = episode_stats["best_running_reward"]
                    if best is None or new_rr > best:
                        episode_stats["best_running_reward"] = new_rr
                    episode_stats["last_ep_return"] = float(ep_ret)
                    episode_stats["last_ep_length"] = int(ep_len)

        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                # cleanrl-style: associate done with the previous obs (not the resulting done)
                dones=done,
                vals=value,
                actions=action,
                logprobs=logprob,
                rewards=reward_t,
                old_adv_preds=old_adv_pred,
                batch_size=(args.num_envs,),
            )
        )

        obs = next_obs_t.to(device, non_blocking=True)
        done = next_done_t.to(device, non_blocking=True)

    container = torch.stack(ts, 0).to(device)
    return obs, done, container


def update(obs, returns, vals, adv_obs, adv_actions, adv_targets, adv_old_preds):
    """One optimizer step: critic regression + pred-clip advantage regression."""
    optimizer.zero_grad()

    # ---- critic regression on the on-policy minibatch (unclipped) ----
    value_pred = agent.value(obs)
    v_loss = 0.5 * ((value_pred - returns) ** 2).mean()

    # ---- advantage regression with PPO-style max clipping on the prediction ----
    adv_logits = agent.advantage(adv_obs)
    adv_pred = adv_logits.gather(1, adv_actions.view(-1, 1)).squeeze(-1)

    adv_loss_unclipped = (adv_pred - adv_targets) ** 2
    adv_pred_clipped = adv_old_preds + torch.clamp(adv_pred - adv_old_preds, -args.clip_coef, args.clip_coef)
    adv_loss_clipped = (adv_pred_clipped - adv_targets) ** 2
    adv_loss = 0.5 * torch.max(adv_loss_unclipped, adv_loss_clipped).mean()

    loss = adv_loss + args.vf_coef * v_loss
    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return v_loss.detach(), adv_loss.detach(), gn


def compute_temperature_t(iteration: int) -> float:
    """RDA temperature schedule: temp(t) = rda_lambda + beta / t.

    The old script used kappa(t) = alpha / (alpha * tau + 1/t).
    Since logits = kappa(t) * Z_theta, this equals logits = Z_theta /
    (tau + (1/alpha)/t). Thus rda_lambda = tau and beta = 1/alpha.
    """
    t_eff = max(1, iteration)
    return float(args.rda_lambda) + float(args.beta) / float(t_eff)


def compute_kappa_t(iteration: int) -> float:
    # Policy logits are advantage_logits / temperature.
    return 1.0 / compute_temperature_t(iteration)


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.adv_minibatch_size is None:
        args.adv_minibatch_size = args.minibatch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{COMPILE}__{CUDAGRAPHS}"

    # Build hp_group string from interesting hyperparameters. Exclude set
    # mirrors cleanrl/cleanrl/rda_atari_envpool.py (env/seed/bookkeeping) plus
    # this script's "boring" knobs that are effectively constant across runs.
    if args.hp_group is None or str(args.hp_group).strip() == "":
        exclude_keys = {
            # bookkeeping / wandb / runtime (matches rda_atari_envpool.py)
            "env_id",
            "seed",
            "exp_name",
            "track",
            "wandb_project_name",
            "wandb_entity",
            "batch_size",
            "minibatch_size",
            "num_iterations",
            "hp_group",
            # boring constants in this fast script
            "torch_deterministic",
            "cuda",
            "measure_burnin",
        }
        cfg = {k: v for k, v in vars(args).items() if k not in exclude_keys}
        full_parts = [f"{k}={cfg[k]}" for k in sorted(cfg.keys())]
        args.hp_group = ",".join(full_parts)

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
            config=vars(args),
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    ####### Agent #######
    agent = Agent(envs, device=device)
    # Detached "inference" version sharing the underlying parameter data.
    agent_inference = Agent(envs, device=device)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=CUDAGRAPHS and not COMPILE,
    )

    ####### Replay buffer #######
    obs_shape = envs.single_observation_space.shape
    rb = GPUReplayBuffer(args.buffer_size, obs_shape, device)

    ####### Executables #######
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.value

    if COMPILE:
        mode = "reduce-overhead" if not CUDAGRAPHS else None
        policy = torch.compile(policy, mode=mode)
        gae = torch.compile(gae, fullgraph=True, mode=mode)
        update = torch.compile(update, mode=mode)

    if CUDAGRAPHS:
        policy = CudaGraphModule(policy, warmup=20)
        # gae has a Python-side loop over num_steps and is best left to compile alone
        update = CudaGraphModule(update, warmup=20)

    ####### Constant device tensors fed into compiled graphs #######
    # kappa = 1 / temperature is updated in-place each iteration; this avoids dynamo recompiles.
    kappa = torch.zeros((), device=device, dtype=torch.float32)

    avg_returns = deque(maxlen=20)
    episode_stats = {
        "running_reward": None,
        "best_running_reward": None,
        "last_ep_return": float("nan"),
        "last_ep_length": 0,
    }
    global_step = 0
    next_obs = torch.tensor(envs.reset(), device=device, dtype=torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)

    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    global_step_burnin = None
    start_time = None

    for iteration in pbar:
        if iteration == args.measure_burnin:
            global_step_burnin = global_step
            start_time = time.time()

        # Anneal lr
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        # Update temperature and inverse-temperature kappa(t)
        temperature_t_val = compute_temperature_t(iteration)
        kappa_t_val = 1.0 / temperature_t_val
        kappa.fill_(kappa_t_val)

        ####### Rollout #######
        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(
            next_obs, next_done, kappa, avg_returns, episode_stats
        )
        global_step += container.numel()

        ####### GAE #######
        torch.compiler.cudagraph_mark_step_begin()
        container = gae(next_obs, next_done, container)
        container_flat = container.view(-1)

        ####### Add to replay buffer (with normalized advantage targets) #######
        adv_flat = container_flat["advantages"]
        if args.norm_adv:
            adv_target = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        else:
            adv_target = adv_flat
        rb.add(
            container_flat["obs"],
            container_flat["actions"].long(),
            adv_target.float(),
            container_flat["old_adv_preds"].float(),
        )

        ####### Optimization #######
        out_v_loss = None
        out_adv_loss = None
        out_gn = None
        for _ in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                obs_mb = container_flat["obs"][b]
                returns_mb = container_flat["returns"][b]
                vals_mb = container_flat["vals"][b]

                adv_obs, adv_act, adv_y, adv_old = rb.sample(args.adv_minibatch_size)

                torch.compiler.cudagraph_mark_step_begin()
                out_v_loss, out_adv_loss, out_gn = update(
                    obs_mb, returns_mb, vals_mb, adv_obs, adv_act, adv_y, adv_old
                )

        ####### Logging #######
        if global_step_burnin is not None and iteration % 10 == 0:
            cur_time = time.time()
            speed = (global_step - global_step_burnin) / (cur_time - start_time)
            global_step_burnin = global_step
            start_time = cur_time

            r = container["rewards"].mean()
            r_max = container["rewards"].max()
            avg_episodic_return = (
                float(torch.tensor(avg_returns).float().mean().item())
                if len(avg_returns) > 0
                else float("nan")
            )
            running_reward = (
                float(episode_stats["running_reward"])
                if episode_stats["running_reward"] is not None
                else float("nan")
            )
            best_running_reward = (
                float(episode_stats["best_running_reward"])
                if episode_stats["best_running_reward"] is not None
                else float("nan")
            )

            with torch.no_grad():
                logs = {
                    # episode-return tracking (sweep metric reads charts/running_reward)
                    "charts/running_reward": running_reward,
                    "charts/best_running_reward": best_running_reward,
                    "charts/avg_episodic_return": avg_episodic_return,
                    "charts/episodic_return": episode_stats["last_ep_return"],
                    "charts/episodic_length": episode_stats["last_ep_length"],
                    # training pace
                    "charts/SPS": speed,
                    "charts/learning_rate": float(optimizer.param_groups[0]["lr"].item()),
                    "charts/temperature": temperature_t_val,
                    "charts/kappa": kappa_t_val,
                    "charts/rda_lambda": args.rda_lambda,
                    "charts/beta": args.beta,
                    # rollout stats
                    "rollout/r_mean": r.item(),
                    "rollout/r_max": r_max.item(),
                    "rollout/advantages": container["advantages"].mean().item(),
                    "rollout/returns": container["returns"].mean().item(),
                    "rollout/vals": container["vals"].mean().item(),
                    # losses
                    "losses/value_loss": out_v_loss.mean().item() if out_v_loss is not None else 0.0,
                    "losses/adv_loss": out_adv_loss.mean().item() if out_adv_loss is not None else 0.0,
                    "losses/grad_norm": out_gn.mean().item() if out_gn is not None else 0.0,
                }

            pbar.set_description(
                f"speed: {speed: 4.1f} sps, "
                f"r/avg: {avg_episodic_return: 4.2f}, "
                f"r/run: {running_reward: 4.2f}, "
                f"r/best: {best_running_reward: 4.2f}, "
                f"temp: {temperature_t_val:4.4f}, "
                f"kappa: {kappa_t_val:4.4f}"
            )
            if args.track:
                wandb.log(logs, step=global_step)

    envs.close()
