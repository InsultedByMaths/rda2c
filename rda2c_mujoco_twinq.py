# RDA2C MuJoCo twin-Q (SAC backbone).
# Twin plain-Q critic + RDA quadratic-energy actor regressed against soft-Q labels
# (Q - alpha*logpi) via K-sample RLOO. Labels are stored raw in a separate RDA buffer
# and minibatch-normalized + clipped to [-adv_clip, adv_clip] at regression time.
import os
import random
import time
import math
from dataclasses import dataclass, fields
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: Optional[str] = None
    """wandb run group; if None, uses auto-computed hp_group"""
    hp_group: str = ""
    """hyperparameter grouping key for analysis; populated at runtime from sorted (k,v)"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v5"
    """the environment id of the task"""
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    num_envs: int = 5
    """the number of parallel game environments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    target_entropy_multiplier: float = -1.0
    """multiplier for SAC target entropy: target_entropy = target_entropy_multiplier * action_dim"""
    alpha_max: float = 0.5
    """maximum clamp for the autotuned temperature alpha_t"""
    alpha_min: float = 1e-4
    """minimum clamp for the autotuned temperature alpha_t"""
    alpha_lr: float = 1e-3
    """learning rate for the SAC-style alpha controller"""
    autotune_alpha_init: float = 0.05
    """initial alpha_t (log_alpha starts at log(this); must be > 0)"""
    # RDA-specific
    rda_window_size: int = 12_800
    """FIFO buffer size for RDA regression"""
    adv_clip: float = 10.0
    """clip normalized regression targets to [-adv_clip, adv_clip]"""
    max_grad_norm: float = 20.0
    """max gradient norm for clipping (actor only)"""
    log_std_min: float = -5.0
    """min value for log_std_base after tanh rescaling"""
    log_std_max: float = 0.5
    """max value for log_std_base after tanh rescaling"""
    eval_every: int = 50_000
    """run deterministic evaluation every this many global environment steps"""
    eval_episodes: int = 20
    """number of deterministic evaluation episodes"""
    num_label_samples: int = 2
    """number of RLOO label samples"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Cast obs to float32 to match observation space and silence gymnasium dtype warnings
        obs_space = env.observation_space
        if hasattr(obs_space, "low") and hasattr(obs_space, "high"):
            env = gym.wrappers.TransformObservation(
                env,
                lambda obs: np.asarray(obs, dtype=np.float32),
                observation_space=gym.spaces.Box(
                    low=obs_space.low, high=obs_space.high, shape=obs_space.shape, dtype=np.float32
                ),
            )
        else:
            env = gym.wrappers.TransformObservation(env, lambda obs: np.asarray(obs, dtype=np.float32))
        env.action_space.seed(seed)
        return env

    return thunk


# ───────────────────────── Plain Q critic ─────────────────────────
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ───────────────────────── RDA quadratic energy actor ─────────────────────────
class Actor(nn.Module):
    """Quadratic energy policy: mu(s), log_std_base(s), b(s). Sampling uses std = exp(log_std_base)*sqrt(alpha), while the regression energy is defined in pre-tanh coordinates u using a simple quadratic-plus-bias form. A separate log-std regularizer handles std inflation."""

    def __init__(self, env, log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        obs_dim = np.array(env.single_observation_space.shape).prod()
        act_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.fc_b = nn.Linear(256, 1)
        nn.init.zeros_(self.fc_mean.weight)
        nn.init.zeros_(self.fc_mean.bias)
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def _forward_mean_logstd_b(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std_base = self.fc_logstd(x)
        log_std_base = torch.tanh(log_std_base)
        log_std_base = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std_base + 1)
        b = self.fc_b(x)
        return mean, log_std_base, b

    def get_action_with_u(self, x, alpha):
        """alpha: scalar (time-varying temperature). Returns (action, log_prob, mean_action, u)."""
        mean, log_std_base, _ = self._forward_mean_logstd_b(x)
        sigma = log_std_base.exp() * (alpha ** 0.5)
        normal = torch.distributions.Normal(mean, sigma)
        u_t = normal.rsample()
        y_t = torch.tanh(u_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(u_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action, u_t

    def get_action(self, x, alpha):
        """alpha: scalar (time-varying temperature). Returns (action, log_prob, mean_action)."""
        action, log_prob, mean_action, _ = self.get_action_with_u(x, alpha)
        return action, log_prob, mean_action

    def energy_terms_from_u(self, obs, u):
        """Return Z_theta(s,u) and the corresponding actor statistics for a pre-tanh sample u."""
        mean, log_std_base, b = self._forward_mean_logstd_b(obs)
        sigma = log_std_base.exp()
        z = -0.5 * (((u - mean) / sigma) ** 2).sum(dim=1, keepdim=True) + b
        return z, mean, log_std_base, b


def evaluate_deterministic_policy(actor, device, env_id, episodes, base_seed, make_env_fn, run_name, alpha_t):
    eval_env = make_env_fn(env_id, base_seed, 0, False, f"{run_name}-eval")()
    returns = []
    try:
        for ep in range(episodes):
            obs, _ = eval_env.reset(seed=base_seed + ep)
            terminated = truncated = False
            ep_ret = 0.0
            while not (terminated or truncated):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, _, mean_action = actor.get_action(obs_t, alpha_t)
                action = mean_action.squeeze(0).cpu().numpy()
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                ep_ret += float(reward)
            returns.append(ep_ret)
    finally:
        eval_env.close()
    arr = np.asarray(returns, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Build a stable hyperparameter group string based on all (non-bookkeeping) arguments.
    _sanitize = lambda v: str(v).replace("/", "_").replace(" ", "")
    exclude = {
        "exp_name",
        "seed",
        "env_id",
        "track",
        "wandb_project_name",
        "wandb_entity",
        "wandb_group",
        "capture_video",
        "cuda",
        "torch_deterministic",
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

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_suffix = os.environ.get("WANDB_RUN_SUFFIX", "")
    if run_suffix:
        run_name = f"{run_name}__{run_suffix}"
    if args.track:
        import wandb

        wandb_mode = os.environ.get("WANDB_MODE", "online")
        wandb_init_kw = dict(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            mode=wandb_mode,
        )
        if args.wandb_group is not None:
            wandb_init_kw["group"] = args.wandb_group
        wandb.init(**wandb_init_kw)
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

        def log_wandb(metrics: dict, step: int) -> None:
            payload = dict(metrics)
            payload["global_step"] = int(step)
            wandb.log(payload)

    else:

        def log_wandb(metrics: dict, step: int) -> None:
            return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(
        envs,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
    ).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    action_dim = int(np.prod(envs.single_action_space.shape))
    target_entropy = float(args.target_entropy_multiplier * action_dim)

    log_alpha = torch.tensor(
        [math.log(args.autotune_alpha_init)],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    with torch.no_grad():
        log_alpha.clamp_(min=math.log(args.alpha_min), max=math.log(args.alpha_max))
        alpha_t = float(log_alpha.exp().item())

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    # RDA label buffer: store (s, u, g) with pre-tanh u in the action slot and g in the reward slot
    rda_rb = ReplayBuffer(
        args.rda_window_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.batch_size,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    best_eval_mean = -float("inf")
    best_eval_std = float("nan")
    next_eval_step = args.eval_every
    last_z_loss = 0.0
    last_actor_loss = 0.0
    last_alpha_loss = 0.0
    last_alpha_phi = 0.0
    last_adv_mean = last_adv_std = last_adv_min = last_adv_max = 0.0
    last_adv_raw_mean = last_adv_raw_std = last_adv_raw_min = last_adv_raw_max = 0.0
    last_z_pred_mean = last_z_pred_std = last_z_pred_min = last_z_pred_max = 0.0
    last_grad_b = last_grad_mean = last_grad_logstd = 0.0
    env_step = 0
    global_step = 0

    obs, _ = envs.reset(seed=args.seed)
    running_reward = None
    while global_step < args.total_timesteps:
        # Rollout action selection
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions, _, _ = actor.get_action(obs_tensor, alpha_t)
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        env_step += 1
        global_step += args.num_envs

        if "final_info" in infos:
            episodes_over = np.nonzero(infos["final_info"]["_episode"])[0]
            episodic_returns = infos["final_info"]["episode"]["r"][episodes_over]
            episodic_lengths = infos["final_info"]["episode"]["l"][episodes_over]
            for episodic_return, episodic_length in zip(episodic_returns, episodic_lengths):
                running_reward = episodic_return if running_reward is None else 0.05 * episodic_return + 0.95 * running_reward
                print(f"global_step={global_step}, episodic_return={episodic_return}, running_reward={running_reward}")
                log_wandb(
                    {
                        "charts/episodic_return": float(episodic_return),
                        "charts/running_reward": float(running_reward),
                        "charts/episodic_length": float(episodic_length),
                    },
                    global_step,
                )

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        # Training
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations, alpha_t)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_qf_target = min_qf_next_target.view(-1) - alpha_t * next_state_log_pi.view(-1)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * next_qf_target

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if env_step % args.policy_frequency == 0:  # TD3 delayed update support
                # Label once, then do num_envs * policy_frequency regression updates.
                with torch.no_grad():
                    s_label = data.observations
                    if args.num_label_samples < 1:
                        raise ValueError("num_label_samples must be >= 1")

                    # Single-sample: use the selected raw label target.
                    # Multi-sample: use RLOO advantage labels from the same target.
                    s_all = (
                        s_label
                        if args.num_label_samples == 1
                        else s_label.repeat_interleave(args.num_label_samples, dim=0)
                    )
                    a_all, logp_all, _, u_all = actor.get_action_with_u(s_all, alpha_t)
                    q1_all = qf1(s_all, a_all)
                    q2_all = qf2(s_all, a_all)
                    q_all = torch.min(q1_all, q2_all).view(args.batch_size, args.num_label_samples)
                    logp_all = logp_all.view(args.batch_size, args.num_label_samples)
                    # Soft-Q labels: g = Q - alpha*logpi
                    x_all = q_all - alpha_t * logp_all
                    if args.num_label_samples == 1:
                        g_raw_all = x_all.detach()
                    else:
                        x_sum = x_all.sum(dim=1, keepdim=True)
                        loo_baseline = (x_sum - x_all) / float(args.num_label_samples - 1)
                        g_raw_all = (x_all - loo_baseline).detach()
                    u_all = u_all.view(args.batch_size, args.num_label_samples, -1)

                    phi_hat = (logp_all.mean() + target_entropy) / action_dim

                    # Store labels in rda_rb: pre-tanh u in action slot, raw regression target in reward slot.
                    # The dones slot is unused (no dual pred-clip path) and filled with zeros.
                    s_np = s_label.detach().cpu().numpy()
                    zeros_done = np.zeros(args.batch_size, dtype=np.float32)
                    for sample_idx in range(args.num_label_samples):
                        u_np = u_all[:, sample_idx, :].detach().cpu().numpy()
                        g_np = g_raw_all[:, sample_idx].detach().cpu().numpy()
                        rda_rb.add(s_np, s_np, u_np, g_np, zeros_done, [{}] * args.batch_size)

                alpha_loss = -(log_alpha.exp() * phi_hat)
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                with torch.no_grad():
                    log_alpha.clamp_(min=math.log(args.alpha_min), max=math.log(args.alpha_max))
                    alpha_t = float(log_alpha.exp().item())
                last_alpha_loss = float(alpha_loss.item())
                last_alpha_phi = float(phi_hat.item())

                # Regression steps on sampled minibatches from the RDA buffer.
                # Labels are stored raw and normalized + clipped only at regression time.
                rda_size = (rda_rb.buffer_size if rda_rb.full else rda_rb.pos) * rda_rb.n_envs
                for _ in range(args.num_envs * args.policy_frequency):
                    if rda_size < args.batch_size:
                        continue

                    data_rda = rda_rb.sample(args.batch_size)
                    s, u_buf, g_buf = data_rda.observations, data_rda.actions, data_rda.rewards
                    u_reg = u_buf.detach()
                    target_raw = g_buf.view(-1)

                    with torch.no_grad():
                        last_adv_raw_mean = target_raw.mean().item()
                        last_adv_raw_std = target_raw.std(unbiased=False).item()
                        last_adv_raw_min = target_raw.min().item()
                        last_adv_raw_max = target_raw.max().item()

                    target = (target_raw - target_raw.mean()) / (target_raw.std(unbiased=False) + 1e-8)
                    target = torch.clamp(target, -args.adv_clip, args.adv_clip)
                    with torch.no_grad():
                        last_adv_mean = target.mean().item()
                        last_adv_std = target.std(unbiased=False).item()
                        last_adv_min = target.min().item()
                        last_adv_max = target.max().item()

                    z_pred, _, _, _ = actor.energy_terms_from_u(s, u_reg)
                    z_pred = z_pred.view(-1)
                    with torch.no_grad():
                        last_z_pred_mean = z_pred.mean().item()
                        last_z_pred_std = z_pred.std(unbiased=False).item()
                        last_z_pred_min = z_pred.min().item()
                        last_z_pred_max = z_pred.max().item()
                    z_loss = 0.5 * (z_pred - target).pow(2).mean()
                    actor_loss = z_loss

                    last_z_loss = z_loss.item()
                    last_actor_loss = actor_loss.item()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                    with torch.no_grad():
                        last_grad_b = actor.fc_b.weight.grad.norm().item() if actor.fc_b.weight.grad is not None else 0.0
                        last_grad_mean = actor.fc_mean.weight.grad.norm().item() if actor.fc_mean.weight.grad is not None else 0.0
                        last_grad_logstd = actor.fc_logstd.weight.grad.norm().item() if actor.fc_logstd.weight.grad is not None else 0.0
                    actor_optimizer.step()

            if env_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 1000 == 0:
                with torch.no_grad():
                    mean, log_std_base, b = actor._forward_mean_logstd_b(data.observations)
                    bias_mean = b.mean().item()
                    mean_mean = mean.mean().item()

                    base_std = log_std_base.exp()
                    base_std_mean = base_std.mean().item()

                    sigma = base_std * (alpha_t ** 0.5)
                    effective_sampling_std = sigma.mean().item()

                    normal_dist = torch.distributions.Normal(mean, sigma)
                    policy_entropy = normal_dist.entropy().sum(dim=-1).mean().item()

                sps = int(global_step / max(time.time() - start_time, 1e-6))
                print("SPS:", sps)
                m = {
                    "actor/mean_mean": mean_mean,
                    "actor/base_std_mean": base_std_mean,
                    "actor/effective_sampling_std": effective_sampling_std,
                    "actor/pre_tanh_gaussian_entropy": policy_entropy,
                    "losses/qf1_values": qf1_a_values.mean().item(),
                    "losses/qf2_values": qf2_a_values.mean().item(),
                    "losses/qf1_loss": qf1_loss.item(),
                    "losses/qf2_loss": qf2_loss.item(),
                    "losses/qf_loss": qf_loss.item() / 2.0,
                    "losses/z_loss": last_z_loss,
                    "z_pred/mean": last_z_pred_mean,
                    "z_pred/std": last_z_pred_std,
                    "z_pred/min": last_z_pred_min,
                    "z_pred/max": last_z_pred_max,
                    "losses/actor_loss": last_actor_loss,
                    "losses/alpha_t": alpha_t,
                    "losses/alpha_loss": last_alpha_loss,
                    "losses/alpha_phi_hat": last_alpha_phi,
                    "actor/bias_mean": bias_mean,
                    "adv/mean": last_adv_mean,
                    "adv/std": last_adv_std,
                    "adv/min": last_adv_min,
                    "adv/max": last_adv_max,
                    "adv_raw/mean": last_adv_raw_mean,
                    "adv_raw/std": last_adv_raw_std,
                    "adv_raw/min": last_adv_raw_min,
                    "adv_raw/max": last_adv_raw_max,
                    "actor_grad/fc_b_weight": last_grad_b,
                    "actor_grad/fc_mean_weight": last_grad_mean,
                    "actor_grad/fc_logstd_weight": last_grad_logstd,
                    "rda_buf/size": (rda_rb.buffer_size if rda_rb.full else rda_rb.pos) * rda_rb.n_envs,
                    "charts/SPS": sps,
                }
                log_wandb(m, global_step)

        if global_step >= next_eval_step:
            eval_metrics = evaluate_deterministic_policy(
                actor=actor,
                device=device,
                env_id=args.env_id,
                episodes=args.eval_episodes,
                base_seed=args.seed,
                make_env_fn=make_env,
                run_name=run_name,
                alpha_t=alpha_t,
            )
            if eval_metrics["mean"] > best_eval_mean:
                best_eval_mean = eval_metrics["mean"]
                best_eval_std = eval_metrics["std"]
            log_wandb(
                {
                    "eval/mean_return": eval_metrics["mean"],
                    "eval/std_return": eval_metrics["std"],
                    "eval/max_return": eval_metrics["max"],
                    "eval/min_return": eval_metrics["min"],
                    "eval/best_mean_return": best_eval_mean,
                    "eval/best_mean_return_std": best_eval_std,
                },
                global_step,
            )
            print(
                f"[eval step={global_step}] mean={eval_metrics['mean']:.3f} std={eval_metrics['std']:.3f} "
                f"best_mean={best_eval_mean:.3f} best_std={best_eval_std:.3f}"
            )
            next_eval_step += args.eval_every

    envs.close()
    if args.track:
        import wandb

        wandb.finish()
