from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def write_notebook(filename: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    path = ROOT / filename
    path.write_text(nbf.writes(nb))
    print(f"Wrote {path.relative_to(ROOT)}")


def build_l6_3a() -> None:
    cells = [
        md(
            """
            # L6-3a Demo: SAC vs PPO on the crawler

            This notebook operationalizes the Lecture 3 claim that SAC is more sample-efficient on the same crawler and reward, while PPO can still be attractive on wall-clock grounds because it is operationally lighter. It reuses the shared `CrawlerEnv` and the same dense forward-velocity objective used in Lecture 2.

            The lecture connection is:
            - same environment as L6-2
            - same reward for both agents
            - action distribution comparison for the slide that contrasts PPO's boundary-seeking behavior with SAC's smoother stochastic policy

            Lecture citation anchors: `haarnoja2018sac`, `raffin2024sacmassivelyparallel`.
            """
        ),
        code(
            """
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np

            from l6_3_utils import (
                CMM_AMBER,
                CMM_BLUE,
                CMM_SLATE,
                FIGURE_DIR,
                rollout_policy,
                save_figure,
                smooth_xy,
                train_or_load,
            )

            TOTAL_TIMESTEPS = 50_000
            SEED = 7
            ENV_KWARGS = dict(
                include_velocity=True,
                action_mode="torque",
                reward_mode="dense_vel",
            )

            print(f"Saving lecture figures into: {FIGURE_DIR}")
            """
        ),
        code(
            """
            sac_artifact = train_or_load(
                "sac",
                "l6_3a_sac.zip",
                ENV_KWARGS,
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
            )
            ppo_artifact = train_or_load(
                "ppo",
                "l6_3a_ppo.zip",
                ENV_KWARGS,
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
            )

            print("SAC episodes logged:", len(sac_artifact.rewards))
            print("PPO episodes logged:", len(ppo_artifact.rewards))
            print("Recent SAC reward:", np.mean(sac_artifact.rewards[-5:]))
            print("Recent PPO reward:", np.mean(ppo_artifact.rewards[-5:]))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)

            for ax, x_key, x_label, title in [
                (axes[0], "steps", "environment steps", "Sample axis"),
                (axes[1], "wallclock", "wall-clock seconds", "Wall-clock axis"),
            ]:
                if x_key == "steps":
                    sac_x, sac_y = smooth_xy(sac_artifact.steps, sac_artifact.rewards, window=5)
                    ppo_x, ppo_y = smooth_xy(ppo_artifact.steps, ppo_artifact.rewards, window=5)
                else:
                    sac_x, sac_y = smooth_xy(sac_artifact.wallclock, sac_artifact.rewards, window=5)
                    ppo_x, ppo_y = smooth_xy(ppo_artifact.wallclock, ppo_artifact.rewards, window=5)

                ax.plot(sac_x, sac_y, color=CMM_BLUE, lw=2.0, label="SAC")
                ax.plot(ppo_x, ppo_y, color=CMM_AMBER, lw=2.0, label="PPO")
                ax.set_xlabel(x_label)
                ax.set_title(title, color=CMM_SLATE)
                ax.grid(True, alpha=0.25)
                ax.legend(loc="lower right")

            axes[0].set_ylabel("episode reward")
            curve_path = save_figure(fig, "sac_vs_ppo_curves.pdf")
            curve_path
            """
        ),
        code(
            """
            sac_rollout = rollout_policy(sac_artifact.model, ENV_KWARGS, seed=SEED, max_steps=200)
            ppo_rollout = rollout_policy(ppo_artifact.model, ENV_KWARGS, seed=SEED, max_steps=200)

            fig, ax = plt.subplots(figsize=(4.6, 2.6))
            bins = np.linspace(-1.0, 1.0, 31)
            ax.hist(
                sac_rollout["actions"][:, 0],
                bins=bins,
                alpha=0.65,
                color=CMM_BLUE,
                label="SAC",
            )
            ax.hist(
                ppo_rollout["actions"][:, 0],
                bins=bins,
                alpha=0.55,
                color=CMM_AMBER,
                label="PPO",
            )
            ax.set_xlabel("action component value")
            ax.set_ylabel("count")
            ax.set_title("Deterministic rollout action histogram", color=CMM_SLATE)
            ax.legend(loc="upper center")
            hist_path = save_figure(fig, "sac_vs_ppo_action_dist.pdf")

            print("SAC final x:", float(sac_rollout["x"][-1]))
            print("PPO final x:", float(ppo_rollout["x"][-1]))
            hist_path
            """
        ),
        md(
            """
            On this toy crawler, SAC usually climbs faster per environment sample while PPO is cheaper per unit wall-clock work. The exact crossover point depends on the implementation budget and simulator throughput, which is the same practical caveat discussed in the lecture.
            """
        ),
    ]
    write_notebook("L6-3a_demo_crawler_sac_vs_ppo.ipynb", cells)


def build_l6_3b() -> None:
    cells = [
        md(
            """
            # L6-3b Demo: Reward shaping can fail

            This notebook keeps the algorithm fixed and changes only the reward:
            - `sparse_1m`: a hard 1-meter success gate
            - `dense_vel`: the direct forward-velocity signal
            - `shaped`: forward velocity minus an energy penalty

            The pedagogical point is the Lecture 3 reward-design story: sparse rewards make credit assignment hard, but even "helpful" shaping can pull the policy toward the wrong behavior if the proxy is easier to optimize than the real objective.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np

            from l6_3_utils import (
                CMM_AMBER,
                CMM_MOSS,
                CMM_ROSE,
                CMM_SLATE,
                FIGURE_DIR,
                rollout_policy,
                save_figure,
                smooth_xy,
                train_or_load,
            )

            TOTAL_TIMESTEPS = 50_000
            SEED = 11

            reward_envs = {
                "sparse_1m": dict(include_velocity=True, action_mode="torque", reward_mode="sparse_1m"),
                "dense_vel": dict(include_velocity=True, action_mode="torque", reward_mode="dense_vel"),
                "shaped": dict(
                    include_velocity=True,
                    action_mode="torque",
                    reward_mode="shaped",
                    shaping_coef=0.05,
                ),
            }

            print(f"Saving lecture figures into: {FIGURE_DIR}")
            """
        ),
        code(
            """
            sparse_artifact = train_or_load(
                "ppo",
                "l6_3b_ppo_sparse.zip",
                reward_envs["sparse_1m"],
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
            )
            dense_artifact = train_or_load(
                "ppo",
                "l6_3b_ppo_dense.zip",
                reward_envs["dense_vel"],
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
            )
            shaped_artifact = train_or_load(
                "ppo",
                "l6_3b_ppo_shaped.zip",
                reward_envs["shaped"],
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
            )

            print("Recent sparse reward:", np.mean(sparse_artifact.rewards[-5:]))
            print("Recent dense reward:", np.mean(dense_artifact.rewards[-5:]))
            print("Recent shaped reward:", np.mean(shaped_artifact.rewards[-5:]))
            """
        ),
        code(
            """
            sparse_rollout = rollout_policy(sparse_artifact.model, reward_envs["sparse_1m"], seed=SEED)
            dense_rollout = rollout_policy(dense_artifact.model, reward_envs["dense_vel"], seed=SEED)
            shaped_rollout = rollout_policy(shaped_artifact.model, reward_envs["shaped"], seed=SEED)

            fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

            for label, artifact, color in [
                ("dense", dense_artifact, CMM_MOSS),
                ("sparse", sparse_artifact, CMM_ROSE),
                ("shaped", shaped_artifact, CMM_AMBER),
            ]:
                x, y = smooth_xy(artifact.steps, artifact.rewards, window=5)
                axes[0].plot(x, y, lw=2.0, color=color, label=label)

            axes[0].set_xlabel("environment steps")
            axes[0].set_ylabel("episode reward")
            axes[0].set_title("Training curves", color=CMM_SLATE)
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(loc="lower right", fontsize=8)

            axes[1].plot(dense_rollout["x"], color=CMM_MOSS, lw=2.0, label="dense")
            axes[1].plot(sparse_rollout["x"], color=CMM_ROSE, lw=2.0, label="sparse")
            axes[1].plot(shaped_rollout["x"], color=CMM_AMBER, lw=2.0, label="shaped")
            axes[1].set_xlabel("rollout step")
            axes[1].set_ylabel("torso x-position")
            axes[1].set_title("True x-displacement @ eval", color=CMM_SLATE)
            axes[1].grid(True, alpha=0.25)
            axes[1].legend(loc="upper left", fontsize=8)

            fig_path = save_figure(fig, "reward_shaping_fail.pdf")
            print("Final dense x:", float(dense_rollout["x"][-1]))
            print("Final sparse x:", float(sparse_rollout["x"][-1]))
            print("Final shaped x:", float(shaped_rollout["x"][-1]))
            fig_path
            """
        ),
        md(
            """
            The Assignment 3 contrast is deliberate: `Assignments/cmm-26-a3/animRL/rewards/rewards.py` uses a product-style reward so failures in one requirement cannot be compensated by over-optimizing another term. The crawler examples here are additive signals, so the agent can trade one term against another much more freely.
            """
        ),
    ]
    write_notebook("L6-3b_demo_crawler_reward_shaping.ipynb", cells)


def build_l6_3c() -> None:
    cells = [
        md(
            """
            # L6-3c Demo: Action space matters

            This notebook recreates the Lecture 3 action-interface comparison in a crawler-scale toy problem. The question is not whether we can replicate the full humanoid experiment from `pengvanderpanne2017action`, but whether the *qualitative ordering* survives:

            `pd_target_with_gains` >= `pd_target` >> `torque`
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np

            from l6_3_utils import (
                CMM_AMBER,
                CMM_BLUE,
                CMM_ROSE,
                CMM_SLATE,
                FIGURE_DIR,
                save_figure,
                smooth_xy,
                train_or_load,
            )

            TOTAL_TIMESTEPS = 50_000
            SEED = 13

            action_envs = {
                "torque": dict(include_velocity=True, action_mode="torque", reward_mode="dense_vel"),
                "pd_target": dict(include_velocity=True, action_mode="pd_target", reward_mode="dense_vel"),
                "pd_target_with_gains": dict(
                    include_velocity=True,
                    action_mode="pd_target_with_gains",
                    reward_mode="dense_vel",
                ),
            }

            print(f"Saving lecture figures into: {FIGURE_DIR}")
            """
        ),
        code(
            """
            torque_artifact = train_or_load(
                "ppo",
                "l6_3c_ppo_torque.zip",
                action_envs["torque"],
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
            )
            pd_target_artifact = train_or_load(
                "ppo",
                "l6_3c_ppo_pd_target.zip",
                action_envs["pd_target"],
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
            )
            pd_gains_artifact = train_or_load(
                "ppo",
                "l6_3c_ppo_pd_target_with_gains.zip",
                action_envs["pd_target_with_gains"],
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
            )

            print("Recent torque reward:", np.mean(torque_artifact.rewards[-5:]))
            print("Recent PD target reward:", np.mean(pd_target_artifact.rewards[-5:]))
            print("Recent PD+gains reward:", np.mean(pd_gains_artifact.rewards[-5:]))
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(5.4, 2.8))

            for label, artifact, color in [
                ("torque", torque_artifact, CMM_ROSE),
                ("PD target", pd_target_artifact, CMM_AMBER),
                ("PD target + gains", pd_gains_artifact, CMM_BLUE),
            ]:
                x, y = smooth_xy(artifact.steps, artifact.rewards, window=5)
                ax.plot(x, y, lw=2.0, color=color, label=label)

            ax.set_xlabel("environment steps")
            ax.set_ylabel("episode reward")
            ax.set_title("Crawler PPO with three action interfaces", color=CMM_SLATE)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="lower right", fontsize=9)

            fig_path = save_figure(fig, "action_space_curves.pdf")
            fig_path
            """
        ),
        md(
            """
            Assignment 3 makes the same design choice for a reason: `Assignments/cmm-26-a3/animRL/cfg/mimic/mimic_pi_config.py` uses `control_type='P'`, which is the fixed-gain PD-target middle option in this toy comparison rather than raw torque control.
            """
        ),
    ]
    write_notebook("L6-3c_demo_crawler_action_spaces.ipynb", cells)


def build_l6_3d() -> None:
    cells = [
        md(
            """
            # L6-3d Demo: Domain randomization and robustness

            This notebook trains:
            - a nominal policy on the default crawler
            - a DR policy whose mass, friction, and perturbations vary episode to episode

            The lecture framing is the standard sim-to-real one: if the inner loop only sees one nominal simulator, the outer objective is overfit to that simulator. Domain randomization broadens the training distribution and should give up some nominal specialization in exchange for perturbation robustness.

            Lecture citation anchors: `vuong2019dr`, `jakobi1997minimal`.
            """
        ),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np

            from l6_3_utils import (
                CMM_AMBER,
                CMM_ROSE,
                CMM_SLATE,
                FIGURE_DIR,
                compare_policy_push_robustness,
                save_figure,
                train_or_load,
            )

            TOTAL_TIMESTEPS = 50_000
            SEED = 17
            ENV_KWARGS = dict(
                include_velocity=True,
                action_mode="pd_target",
                reward_mode="dense_vel",
            )
            DR_CONFIG = dict(
                mass_range=(0.7, 1.3),
                friction_range=(0.7, 1.3),
                push_prob=0.3,
                push_mag=8.0,
            )

            print(f"Saving lecture figures into: {FIGURE_DIR}")
            """
        ),
        code(
            """
            nominal_artifact = train_or_load(
                "ppo",
                "l6_3d_ppo_nominal.zip",
                ENV_KWARGS,
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
            )
            dr_artifact = train_or_load(
                "ppo",
                "l6_3d_ppo_dr.zip",
                ENV_KWARGS,
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
                dr_config=DR_CONFIG,
            )
            """
        ),
        code(
            """
            push_magnitudes = np.arange(0, 22, 2, dtype=float)
            results = compare_policy_push_robustness(
                {
                    "nominal": nominal_artifact.model,
                    "trained with DR": dr_artifact.model,
                },
                ENV_KWARGS,
                push_magnitudes,
                n_trials=10,
                push_step=75,
                seed=SEED,
            )

            fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharex=True)
            style = {
                "nominal": dict(color=CMM_ROSE, marker="o"),
                "trained with DR": dict(color=CMM_AMBER, marker="s"),
            }

            for label, result in results.items():
                axes[0].plot(
                    result["push_magnitudes"],
                    result["survival_fraction"],
                    lw=2.0,
                    label=label,
                    **style[label],
                )
                axes[1].plot(
                    result["push_magnitudes"],
                    result["mean_displacement"],
                    lw=2.0,
                    label=label,
                    **style[label],
                )

            axes[0].set_xlabel("push magnitude (N)")
            axes[0].set_ylabel("survival fraction")
            axes[0].set_ylim(-0.05, 1.05)
            axes[0].set_title("Survival under a late push", color=CMM_SLATE)
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(loc="upper right", fontsize=8)

            axes[1].set_xlabel("push magnitude (N)")
            axes[1].set_ylabel("mean final x-position")
            axes[1].set_title("Forward progress after the push", color=CMM_SLATE)
            axes[1].grid(True, alpha=0.25)

            fig_path = save_figure(fig, "dr_push_survival.pdf")
            fig_path
            """
        ),
        md(
            """
            Classroom live-demo command with the DR-trained cached policy:

            ```bash
            uv run python teleop_crawler.py --policy saved_checkpoints/l6_3d_ppo_dr.zip --policy-algo ppo --action-mode pd_target
            ```

            Assignment 3 uses the same knobs at higher fidelity on the Pi biped: `randomize_friction`, `randomize_base_mass`, `push_robots`, and `dynamic_randomization`.
            """
        ),
    ]
    write_notebook("L6-3d_demo_crawler_domain_rand.ipynb", cells)


if __name__ == "__main__":
    build_l6_3a()
    build_l6_3b()
    build_l6_3c()
    build_l6_3d()
