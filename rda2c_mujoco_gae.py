# RDA2C MuJoCo GAE (quadratic dual score + value critic)
import os
import random
import time
import math
from collections import deque
from dataclasses import dataclass, fields

import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal

# Single Thread Setup
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = False
    track: bool = False
    wandb_project_name: str = "rda2c"
    wandb_entity: str | None = None
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 1_000_000
    learning_rate: float = 2e-4
    num_envs: int = 1
    num_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.8
    num_minibatches: int = 32
    update_epochs: int = 10
    anneal_lr: bool = False  # learning rate annealing
    max_grad_norm: float = 3.0

    # RDA specific (reparameterized):
    # alpha(iter) = rda_lambda + beta / iter, and std = base_std * sqrt(alpha(iter))
    beta: float = 1.0  # inverse-step-size coefficient
    rda_lambda: float = 0.05  # regularizer coefficient for RDA

    buffer_size: int = 50_000

    vf_coef: float = 0.5  # critic loss coefficient
    clip_coef: float = 0.2
    """Shared clip range (as in PPO) used for both value-loss clipping and dual advantage-prediction clipping."""

    # network architecture
    hidden_dim: int = 128

    # to be filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    # off-policy advantage regression minibatch size (defaults to rollout minibatch if None)
    adv_minibatch_size: int | None = None
    num_iterations: int = 0
    # hyperparameter grouping key for analysis; will be populated at runtime
    hp_group: str = ""
    wandb_group: str | None = None


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=gym.spaces.Box(low=-10, high=10, shape=env.observation_space.shape),
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Quadratic advantage parametrization.

    Outputs mean, log-std, and bias (value-like head).
    """

    def __init__(
        self,
        envs,
        hidden_dim: int = 64,
    ):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)

        # build shared trunk: two hidden layers (fixed), tanh
        shared_layers = []
        for layer_idx in range(2):
            in_dim = obs_dim if layer_idx == 0 else hidden_dim
            shared_layers.append(layer_init(nn.Linear(in_dim, hidden_dim)))
            shared_layers.append(nn.Tanh())
        self.shared = nn.Sequential(*shared_layers)
        self.mu_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.logstd_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)

        # bias head
        self.bias_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        # separate critic (value function): two hidden layers (fixed), tanh
        critic_layers = []
        for layer_idx in range(2):
            in_dim = obs_dim if layer_idx == 0 else hidden_dim
            critic_layers.append(layer_init(nn.Linear(in_dim, hidden_dim)))
            critic_layers.append(nn.Tanh())
        critic_layers.append(layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x):
        h = self.shared(x)
        return self.mu_head(h), self.logstd_head(h), self.bias_head(h).squeeze(-1)

    # Helper wrappers
    def mean(self, x):
        return self.mu_head(self.shared(x))

    def bias(self, x):
        return self.bias_head(self.shared(x)).squeeze(-1)

    def logstd(self, x):
        raw = self.logstd_head(self.shared(x))
        log_std_unit = torch.tanh(raw)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_unit + 1)
        return log_std

    def value(self, x):
        return self.critic(x).squeeze(-1)


class WandbWriter:
    def __init__(self, wandb_module):
        self.wandb = wandb_module

    def add_scalar(self, key: str, value: float, step: int):
        if self.wandb is not None:
            self.wandb.log({key: value}, step=step)

    def close(self):
        pass


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Build a stable hyperparameter group string based on all (non-bookkeeping) arguments.
    _sanitize = lambda v: str(v).replace("/", "_").replace(" ", "")
    exclude = {
        # bookkeeping / runtime
        "exp_name",
        "seed",
        "env_id",
        "track",
        "wandb_project_name",
        "wandb_entity",
        "wandb_group",
        "cuda",
        "torch_deterministic",
        "capture_video",
        # derived at runtime
        "batch_size",
        "minibatch_size",
        "num_iterations",
        # these fields are set here
        "hp_group",
    }
    args.hp_group = "-".join(
        f"{k}={_sanitize(v)}"
        for k, v in sorted(
            (f.name, getattr(args, f.name))
            for f in fields(Args)
            if f.name not in exclude
        )
    )

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # default advantage minibatch to rollout minibatch if not provided
    if args.adv_minibatch_size is None:
        args.adv_minibatch_size = args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    # Make run_name unique across concurrent sweep tasks to avoid W&B run collisions.
    run_suffix = os.environ.get("WANDB_RUN_SUFFIX", f"pid{os.getpid()}")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{run_suffix}__{time.time_ns()}"

    wandb_module = None
    if args.track:
        import wandb as wandb_module

        wandb_module.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            mode="offline",
            save_code=True,
        )
    writer = WandbWriter(wandb_module)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environments
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Robust LOG_STD_MAX: derive from a fresh raw env's action_space bounds
    tmp_env = gym.make(args.env_id)
    action_high = float(tmp_env.action_space.high[0])
    action_low = float(tmp_env.action_space.low[0])
    action_scale = (action_high - action_low) / 2.0
    tmp_env.close()

    LOG_STD_MAX = math.log(0.8 * action_scale)
    LOG_STD_MIN = -5.0

    # ensure observations use float32 throughout (prevents dtype mismatches)
    envs.single_observation_space.dtype = np.float32

    agent = Agent(
        envs,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # Initialize log-std head so effective_sampling_std is at midpoint of [LOG_STD_MIN, LOG_STD_MAX]
    # at iteration 1 (base_std * sqrt(alpha1) = exp(mid_logstd) => target_logstd = mid_logstd - 0.5*log(alpha1))
    beta1 = args.beta
    alpha1 = args.rda_lambda + beta1
    mid_logstd = 0.5 * (LOG_STD_MIN + LOG_STD_MAX)
    target_logstd = mid_logstd - 0.5 * math.log(alpha1)
    u = 2.0 * (target_logstd - LOG_STD_MIN) / (LOG_STD_MAX - LOG_STD_MIN) - 1.0
    u = float(np.clip(u, -0.999, 0.999))
    raw_bias = math.atanh(u)
    with torch.no_grad():
        agent.logstd_head.weight.zero_()
        agent.logstd_head.bias.fill_(raw_bias)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-8)

    # Replay buffer from Stable Baselines3
    # SB3 expects buffer_size argument in transitions; internally it stores as slots = buffer_size // n_envs
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    old_adv_preds = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # additional storage for KL-regularized value evaluation
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obses = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    terminations = torch.zeros((args.num_steps, args.num_envs)).to(device)
    truncations = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    running_reward = None
    episodic_returns_buffer = deque(maxlen=100)

    for iteration in range(1, args.num_iterations + 1):
        # Anneal learning rate if enabled
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # Compute sampling scale alpha(iter) once per iteration: alpha(iter) = rda_lambda + beta/iter
        t_eff = max(1, iteration)
        alpha_val = args.rda_lambda + args.beta / t_eff

        std_sum = 0.0
        base_std_sum = 0.0
        mag_sum = 0.0
        mean_sum = 0.0
        bias_sum = 0.0
        offset_sum = 0.0

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs

            with torch.no_grad():
                mean = agent.mean(next_obs)
                logstd = agent.logstd(next_obs)
                base_std = torch.exp(logstd)
                std = base_std * math.sqrt(alpha_val)
                bias_val = agent.bias(next_obs)
                bias_sum += bias_val.mean().item()
                dist = Normal(mean, std)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(1)
                # log-prob normalizing constant: -sum(log std) - 0.5 * d * log(2π)
                log_std = torch.log(std)
                act_dim = log_std.shape[-1]
                offset = -log_std.sum(dim=1) - 0.5 * act_dim * math.log(2.0 * math.pi)
                offset_sum += offset.mean().item()
                value = agent.value(next_obs)
                # "old" advantage prediction used by PPO-style clipped advantage loss.
                diff_old = (action - mean) / base_std
                old_adv_pred = -0.5 * (diff_old**2).sum(dim=1) + bias_val
                mean_sum += mean.abs().mean().item()
                base_std_sum += base_std.mean().item()
                std_sum += std.mean().item()
                mag_sum += action.abs().mean().item()

            actions[step] = action
            values[step] = value
            old_adv_preds[step] = old_adv_pred
            logprobs[step] = logprob

            # Step env
            next_obs_np, reward, next_terminations, next_truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(next_terminations, next_truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
            # Correct next observation (for vec gym): if a time-limit truncation happened, use the final
            # observation for bootstrapping (instead of the post-reset observation).
            real_next_obs_np = np.array(next_obs_np, copy=True)
            for idx, trunc in enumerate(next_truncations):
                if trunc:
                    real_next_obs_np[idx] = infos["final_obs"][idx]

            # store corrected next observations for each step (used for GAE bootstrapping)
            next_obses[step] = torch.tensor(real_next_obs_np, dtype=torch.float32).to(device)
            terminations[step] = torch.tensor(next_terminations, dtype=torch.float32).to(device)
            truncations[step] = torch.tensor(next_truncations, dtype=torch.float32).to(device)
            dones[step] = torch.tensor(next_done, dtype=torch.float32).to(device)

            if "final_info" in infos:
                episodes_over = np.nonzero(infos["final_info"]["_episode"])[0]
                episodic_returns = infos["final_info"]["episode"]["r"][episodes_over]
                episodic_lengths = infos["final_info"]["episode"]["l"][episodes_over]
                for episodic_return, episodic_length in zip(episodic_returns, episodic_lengths):
                    episodic_returns_buffer.append(float(episodic_return))
                    running_reward = episodic_return if running_reward is None else 0.05 * episodic_return + 0.95 * running_reward
                    print(
                        f"iter={iteration}/{args.num_iterations}, global_step={global_step}, "
                        f"episodic_return={episodic_return}, running_reward={running_reward}"
                    )
                    writer.add_scalar("rollout/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/running_reward", running_reward, global_step)
                    writer.add_scalar("charts/average_reward", sum(episodic_returns_buffer) / len(episodic_returns_buffer), global_step)
                    writer.add_scalar("rollout/episodic_length", episodic_length, global_step)

        # ——— GAE & returns ———
        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(device)
            next_values = torch.zeros_like(values[0]).to(device)
            lastgaelam = torch.zeros_like(values[0]).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_values = agent.value(next_obses[t])
                else:
                    done_mask = dones[t].bool()
                    if done_mask.any():
                        next_values[done_mask] = agent.value(next_obses[t][done_mask])
                    next_values[~done_mask] = values[t + 1][~done_mask]

                # bootstrap only on true terminations (time-limit truncations should bootstrap)
                delta = rewards[t] + args.gamma * next_values * (1.0 - terminations[t]) - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * (1.0 - dones[t]) * lastgaelam
            returns = advantages + values

        # Log rollout advantage stats (pre-normalization)
        adv_rollout_mean = advantages.mean()
        adv_rollout_std = advantages.std(unbiased=False)
        adv_rollout_max = advantages.max()
        adv_rollout_min = advantages.min()
        writer.add_scalar("rollout/adv_rollout_mean", adv_rollout_mean.item(), global_step)
        writer.add_scalar("rollout/adv_rollout_std", adv_rollout_std.item(), global_step)
        writer.add_scalar("rollout/adv_rollout_max", adv_rollout_max.item(), global_step)
        writer.add_scalar("rollout/adv_rollout_min", adv_rollout_min.item(), global_step)

        # —— Flatten ——
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_logprobs = logprobs.reshape(-1)

        # keep raw advantages in the replay buffer; normalization happens per regression batch below
        norm_advantages = advantages

        # ---- store in replay buffer step-wise ----
        # Convert rollout tensors to NumPy once per iteration to avoid per-step sync overhead
        obs_np = obs.detach().cpu().numpy()
        next_obses_np = next_obses.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        norm_adv_np = norm_advantages.detach().cpu().numpy()
        old_adv_preds_np = old_adv_preds.detach().cpu().numpy()

        for t in range(args.num_steps):
            rb.add(
                obs_np[t],
                next_obses_np[t],
                actions_np[t],
                norm_adv_np[t],
                old_adv_preds_np[t],
                {},
            )

        # —— Regression update ——
        b_inds = np.arange(args.batch_size)
        max_grad_norm = []
        grad_clipfrac = []
        max_adv_residual_iter = 0.0
        entropy_loss = None
        old_approx_kl = None
        approx_kl = None
        adv_pred_mean = None
        adv_pred_std = None
        adv_pred_max = None

        num_minibatches = args.batch_size // args.minibatch_size
        adv_mb = args.adv_minibatch_size or args.minibatch_size

        # sample fresh advantage-regression targets each epoch, batch-normalized
        total_adv = num_minibatches * args.update_epochs * adv_mb
        adv_batch = rb.sample(total_adv)  # ReplayBufferSamples

        y = adv_batch.rewards.squeeze(-1).float()
        y_mean = y.mean()
        y_std = y.std(unbiased=False) + 1e-8
        y_norm = (y - y_mean) / y_std

        writer.add_scalar("learning/adv_batch_mean", y_mean.item(), global_step)
        writer.add_scalar("learning/adv_batch_std", y_std.item(), global_step)
        writer.add_scalar("learning/adv_batch_max_pre_norm", y.max().item(), global_step)
        writer.add_scalar("learning/adv_batch_min_pre_norm", y.min().item(), global_step)
        writer.add_scalar("learning/adv_batch_max_post_norm", y_norm.max().item(), global_step)
        writer.add_scalar("learning/adv_batch_min_post_norm", y_norm.min().item(), global_step)

        # shuffle once and consume sequentially across all epochs
        perm = torch.randperm(total_adv, device=y.device)
        ptr = 0

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                ################################################################################
                # critic value regression (on current rollout data)
                ################################################################################

                value_pred = agent.value(b_obs[mb_inds])

                # Log policy entropy/KL (same approximation as PPO)
                with torch.no_grad():
                    mean = agent.mean(b_obs[mb_inds])
                    logstd = agent.logstd(b_obs[mb_inds])
                    std = torch.exp(logstd) * math.sqrt(alpha_val)
                    dist = Normal(mean, std)
                    newlogprob = dist.log_prob(b_actions[mb_inds]).sum(1)
                    entropy = dist.entropy().sum(1)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((logratio.exp() - 1) - logratio).mean()
                    entropy_loss = entropy.mean()

                # Value loss (PPO-style clipping around old values)
                v_loss_unclipped = (value_pred - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    value_pred - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                ################################################################################
                # advantage regression: plain replay buffer sampling (no indices/usage tracking)
                ################################################################################
                mb = perm[ptr: ptr + adv_mb]
                ptr += adv_mb

                obs_mb = adv_batch.observations[mb].float()
                act_mb = adv_batch.actions[mb].float()
                y_mb = y_norm[mb]
                old_adv_pred_mb = adv_batch.dones[mb].squeeze(-1).float()

                mean_after = agent.mean(obs_mb)
                logstd_after = agent.logstd(obs_mb)
                std_after = torch.exp(logstd_after)
                diff = (act_mb - mean_after) / std_after
                quad_term = -0.5 * (diff**2).sum(dim=1)
                bias_mb = agent.bias(obs_mb)
                adv_pred = quad_term + bias_mb

                errors = adv_pred - y_mb

                adv_pred_mean = adv_pred.mean().item()
                adv_pred_std = adv_pred.std(unbiased=False).item()
                adv_pred_max = adv_pred.max().item()

                # Dual (advantage) regression loss: PPO-style clipped prediction
                adv_loss_unclipped = (adv_pred - y_mb) ** 2
                adv_pred_clipped = old_adv_pred_mb + torch.clamp(
                    adv_pred - old_adv_pred_mb,
                    -args.clip_coef,
                    args.clip_coef,
                )
                adv_loss_clipped = (adv_pred_clipped - y_mb) ** 2
                adv_loss = 0.5 * torch.max(adv_loss_unclipped, adv_loss_clipped).mean()

                # compute metric without affecting autograd graph
                with torch.no_grad():
                    max_adv_residual_iter = max(
                        max_adv_residual_iter,
                        float(errors.detach().abs().max().item()),
                    )

                loss = adv_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                max_grad_norm.append(grad_norm_val)
                grad_clipfrac.append(float(grad_norm_val > args.max_grad_norm))
                optimizer.step()

        print(f"iter {iteration}/{args.num_iterations} | SPS: {int(global_step / (time.time() - start_time))}")

        # Consolidated end-of-iteration logging
        writer.add_scalar("actor/mean", mean_sum / args.num_steps, global_step)
        writer.add_scalar("actor/base_std", base_std_sum / args.num_steps, global_step)
        writer.add_scalar("actor/effective_sampling_std", std_sum / args.num_steps, global_step)
        writer.add_scalar("actor/alpha", alpha_val, global_step)
        writer.add_scalar("actor/beta", args.beta, global_step)
        writer.add_scalar("actor/bias", bias_sum / args.num_steps, global_step)
        writer.add_scalar("rollout/theoretical_offset", offset_sum / args.num_steps, global_step)
        writer.add_scalar("rollout/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("learning/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("learning/total_loss", loss.item(), global_step)
        writer.add_scalar("learning/adv_loss", adv_loss.item(), global_step)
        writer.add_scalar("learning/value_loss", v_loss.item(), global_step)
        writer.add_scalar("learning/max_adv_residual", max_adv_residual_iter, global_step)
        if adv_pred_mean is not None and adv_pred_std is not None and adv_pred_max is not None:
            writer.add_scalar("learning/adv_pred_mean", float(adv_pred_mean), global_step)
            writer.add_scalar("learning/adv_pred_std", float(adv_pred_std), global_step)
            writer.add_scalar("learning/adv_pred_max", float(adv_pred_max), global_step)
        if max_grad_norm:
            writer.add_scalar("learning/max_grad_norm", float(np.mean(max_grad_norm)), global_step)
        if grad_clipfrac:
            writer.add_scalar("learning/grad_clipfrac", float(np.mean(grad_clipfrac)), global_step)
        if entropy_loss is not None and old_approx_kl is not None and approx_kl is not None:
            writer.add_scalar("learning/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("learning/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("learning/approx_kl", approx_kl.item(), global_step)

        # Explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        writer.add_scalar("learning/explained_variance", explained_var, global_step)

    envs.close()
    writer.close()
