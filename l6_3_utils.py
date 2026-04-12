from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.crawler2d import CrawlerEnv


CMM_BLUE = "#1C3B6E"
CMM_SLATE = "#3B4A63"
CMM_AMBER = "#E09F3E"
CMM_ROSE = "#9E2A2B"
CMM_MOSS = "#4A7C59"

NOTEBOOK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOK_ROOT.parents[1]
FIGURE_DIR = PROJECT_ROOT / "Figures" / "Lecture03"
CHECKPOINT_DIR = NOTEBOOK_ROOT / "saved_checkpoints"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
    }
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


class CrawlerSB3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_kwargs: dict, dr_config: dict | None = None):
        super().__init__()
        self.env_kwargs = dict(env_kwargs)
        self.dr_config = dict(dr_config or {})
        self.env = CrawlerEnv(**self.env_kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.env.act_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.env._rng = np.random.default_rng(seed)
        obs = self.env.reset()
        if self.dr_config:
            # Apply DR after reset so queued pushes survive the reset path.
            self.env.sample_dr(**self.dr_config)
            obs = self.env.get_obs()
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def close(self):
        return None


class EpisodeStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.training_start_time = 0.0
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_steps: list[int] = []
        self.episode_wallclock: list[float] = []

    def _on_training_start(self) -> None:
        self.training_start_time = time.perf_counter()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            self.episode_rewards.append(float(episode["r"]))
            self.episode_lengths.append(int(episode["l"]))
            self.episode_steps.append(int(self.num_timesteps))
            self.episode_wallclock.append(
                float(time.perf_counter() - self.training_start_time)
            )
        return True


@dataclass
class TrainingArtifact:
    model: PPO | SAC
    rewards: np.ndarray
    lengths: np.ndarray
    steps: np.ndarray
    wallclock: np.ndarray


def metric_path_for(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_metrics.npz")


def make_vec_env(env_kwargs: dict, dr_config: dict | None = None) -> DummyVecEnv:
    def thunk():
        return Monitor(CrawlerSB3Env(env_kwargs=env_kwargs, dr_config=dr_config))

    return DummyVecEnv([thunk])


def algo_class(name: str):
    if name == "ppo":
        return PPO
    if name == "sac":
        return SAC
    raise ValueError(f"Unsupported algorithm: {name}")


def default_algo_kwargs(name: str) -> dict:
    common = {
        "policy_kwargs": dict(net_arch=[64, 64]),
        "device": "cpu",
        "seed": 0,
        "verbose": 0,
    }
    if name == "ppo":
        return {
            **common,
            "n_steps": 1024,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "gamma": 0.99,
        }
    return {
        **common,
        "learning_rate": 3e-4,
        "buffer_size": 50_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "tau": 0.02,
        "gamma": 0.99,
    }


def save_metrics(path: Path, callback: EpisodeStatsCallback) -> None:
    np.savez_compressed(
        path,
        rewards=np.asarray(callback.episode_rewards, dtype=np.float32),
        lengths=np.asarray(callback.episode_lengths, dtype=np.int32),
        steps=np.asarray(callback.episode_steps, dtype=np.int32),
        wallclock=np.asarray(callback.episode_wallclock, dtype=np.float32),
    )


def load_metrics(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def train_or_load(
    algo_name: str,
    checkpoint_name: str,
    env_kwargs: dict,
    *,
    total_timesteps: int = 50_000,
    seed: int = 0,
    dr_config: dict | None = None,
    algo_kwargs: dict | None = None,
) -> TrainingArtifact:
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    metrics_path = metric_path_for(checkpoint_path)
    cls = algo_class(algo_name)

    if checkpoint_path.exists() and metrics_path.exists():
        model = cls.load(checkpoint_path, device="cpu")
        metrics = load_metrics(metrics_path)
        return TrainingArtifact(
            model=model,
            rewards=metrics["rewards"],
            lengths=metrics["lengths"],
            steps=metrics["steps"],
            wallclock=metrics["wallclock"],
        )

    set_seed(seed)
    vec_env = make_vec_env(env_kwargs, dr_config=dr_config)
    callback = EpisodeStatsCallback()
    train_kwargs = default_algo_kwargs(algo_name)
    train_kwargs.update(algo_kwargs or {})
    train_kwargs["seed"] = seed
    model = cls("MlpPolicy", vec_env, **train_kwargs)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    model.save(checkpoint_path)
    save_metrics(metrics_path, callback)
    vec_env.close()
    return TrainingArtifact(
        model=model,
        rewards=np.asarray(callback.episode_rewards, dtype=np.float32),
        lengths=np.asarray(callback.episode_lengths, dtype=np.int32),
        steps=np.asarray(callback.episode_steps, dtype=np.int32),
        wallclock=np.asarray(callback.episode_wallclock, dtype=np.float32),
    )


def moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="valid")


def smooth_xy(x: np.ndarray, y: np.ndarray, window: int = 5) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if y.size < window:
        return x, y
    return x[window - 1 :], moving_average(y, window=window)


def rollout_policy(
    model: PPO | SAC,
    env_kwargs: dict,
    *,
    seed: int = 0,
    max_steps: int = 500,
    push_step: int | None = None,
    push_force: float = 0.0,
    push_duration: int = 3,
) -> dict[str, np.ndarray | bool]:
    env = CrawlerSB3Env(env_kwargs=env_kwargs)
    obs, _ = env.reset(seed=seed)
    xs = [float(env.env.data.qpos[0])]
    torso_z = [float(env.env.data.qpos[1])]
    actions = []
    rewards = []
    survived = True

    for step_idx in range(max_steps):
        if push_step is not None and step_idx == push_step and push_force != 0.0:
            env.env.apply_external_force(fx=push_force, duration_steps=push_duration)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        rewards.append(float(reward))
        xs.append(float(info["x"]))
        torso_z.append(float(env.env.data.qpos[1]))
        if env.env.data.qpos[1] < 0.015:
            survived = False
        if terminated or truncated:
            break

    return {
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "x": np.asarray(xs, dtype=np.float32),
        "torso_z": np.asarray(torso_z, dtype=np.float32),
        "survived": survived,
    }


def compare_policy_push_robustness(
    models: dict[str, PPO | SAC],
    env_kwargs: dict,
    push_magnitudes: np.ndarray,
    *,
    n_trials: int = 10,
    push_step: int = 75,
    seed: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    for label, model in models.items():
        displacement = []
        survival = []
        for magnitude in push_magnitudes:
            d_trials = []
            s_trials = []
            for trial in range(n_trials):
                rollout = rollout_policy(
                    model,
                    env_kwargs=env_kwargs,
                    seed=seed + trial,
                    max_steps=500,
                    push_step=push_step,
                    push_force=float(magnitude),
                )
                d_trials.append(float(rollout["x"][-1]))
                s_trials.append(float(rollout["survived"]))
            displacement.append(np.mean(d_trials))
            survival.append(np.mean(s_trials))
        results[label] = {
            "push_magnitudes": np.asarray(push_magnitudes, dtype=np.float32),
            "mean_displacement": np.asarray(displacement, dtype=np.float32),
            "survival_fraction": np.asarray(survival, dtype=np.float32),
        }
    return results


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
