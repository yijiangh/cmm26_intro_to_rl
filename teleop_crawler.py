"""
Interactive teleoperation demo for the shared 2D MuJoCo crawler.

Two manual-control modes are supported:
  continuous  : sliders for each action dimension
  discrete    : four coarse buttons that drive the first two action dims

The demo also exposes the L6-3 domain-randomization hooks live:
  - mass scale
  - floor friction scale
  - external pushes
  - action-space dropdown

Optional cached SB3 policies can be loaded for the "push-it-and-watch" demo:
  uv run python teleop_crawler.py --policy saved_checkpoints/l6_3d_ppo_dr.zip \
      --policy-algo ppo --action-mode pd_target
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, ttk

from PIL import Image, ImageTk
import mujoco
import numpy as np

from envs.crawler2d import ACTION_MODES, CrawlerEnv


RENDER_W, RENDER_H = 640, 360
POLICY_OBS_VELOCITY = True
DEFAULT_PUSH_FORCE = 8.0
FPS_MS = 16

ACTION_LABELS = {
    "torque": ("Joint 1 torque", "Joint 2 torque"),
    "pd_target": ("Joint 1 target", "Joint 2 target"),
    "pd_target_with_gains": (
        "Joint 1 target",
        "Joint 2 target",
        "Joint 1 gain adj",
        "Joint 2 gain adj",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate the 2D crawler")
    parser.add_argument(
        "--mode",
        choices=["continuous", "discrete"],
        default="continuous",
        help="Manual-control widget style.",
    )
    parser.add_argument(
        "--action-mode",
        choices=ACTION_MODES,
        default="torque",
        help="Initial crawler action space.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Optional SB3 checkpoint (.zip) to autoplay.",
    )
    parser.add_argument(
        "--policy-algo",
        choices=["auto", "ppo", "sac"],
        default="auto",
        help="SB3 algorithm class used to load --policy.",
    )
    return parser.parse_args()


def load_policy(path: Path, algo: str):
    if not path.exists():
        raise FileNotFoundError(path)

    from stable_baselines3 import PPO, SAC

    loaders = []
    if algo == "auto":
        loaders = [("ppo", PPO), ("sac", SAC)]
    elif algo == "ppo":
        loaders = [("ppo", PPO)]
    else:
        loaders = [("sac", SAC)]

    errors = []
    for algo_name, cls in loaders:
        try:
            return cls.load(path), algo_name
        except Exception as exc:  # pragma: no cover - best-effort CLI loading
            errors.append(f"{algo_name}: {type(exc).__name__}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to load policy {path}.\n{joined}")


def make_env(action_mode: str, mass_scale: float, friction_scale: float) -> CrawlerEnv:
    env = CrawlerEnv(
        include_velocity=POLICY_OBS_VELOCITY,
        action_mode=action_mode,
        reward_mode="dense_vel",
        mass_scale=mass_scale,
        friction_scale=friction_scale,
    )
    env.reset()
    return env


def clamp_action_dims(values: np.ndarray, act_dim: int) -> np.ndarray:
    action = np.zeros(act_dim, dtype=np.float32)
    action[: min(act_dim, values.size)] = values[:act_dim]
    return np.clip(action, -1.0, 1.0)


def main() -> None:
    args = parse_args()

    root = tk.Tk()
    root.title(f"Crawler Teleop — {args.mode}")
    root.configure(bg="#2b2b2b")

    canvas = Canvas(
        root,
        width=RENDER_W,
        height=RENDER_H,
        bg="black",
        highlightthickness=0,
    )
    canvas.pack(padx=10, pady=(10, 5))
    tk_img_ref = [None]

    info_var = tk.StringVar(value="")
    policy_var = tk.StringVar(value="Policy: manual control")
    mass_text = tk.StringVar()
    friction_text = tk.StringVar()
    push_text = tk.StringVar()
    action_mode_var = tk.StringVar(value=args.action_mode)
    autoplay_var = tk.BooleanVar(value=args.policy is not None)

    top_panel = tk.Frame(root, bg="#2b2b2b")
    top_panel.pack(padx=10, pady=(0, 8), fill=tk.X)
    controls_panel = tk.Frame(root, bg="#2b2b2b")
    controls_panel.pack(padx=10, pady=5, fill=tk.X)

    status_frame = tk.Frame(root, bg="#2b2b2b")
    status_frame.pack(fill=tk.X, padx=15, pady=(0, 8))
    tk.Label(
        status_frame,
        textvariable=policy_var,
        font=("Courier", 11),
        fg="#f2c66d",
        bg="#2b2b2b",
        anchor="w",
        justify=tk.LEFT,
    ).pack(fill=tk.X)
    tk.Label(
        status_frame,
        textvariable=info_var,
        font=("Courier", 11),
        fg="#e0e0e0",
        bg="#2b2b2b",
        anchor="w",
        justify=tk.LEFT,
    ).pack(fill=tk.X)

    dr_frame = tk.LabelFrame(
        top_panel,
        text="Domain Randomization",
        font=("Helvetica", 12, "bold"),
        fg="#f2f2f2",
        bg="#2b2b2b",
        padx=10,
        pady=8,
        labelanchor="n",
    )
    dr_frame.pack(fill=tk.X)

    manual_frame = tk.LabelFrame(
        controls_panel,
        text="Manual Actions",
        font=("Helvetica", 12, "bold"),
        fg="#f2f2f2",
        bg="#2b2b2b",
        padx=10,
        pady=8,
        labelanchor="n",
    )
    manual_frame.pack(fill=tk.X)

    mass_var = tk.DoubleVar(value=1.0)
    friction_var = tk.DoubleVar(value=1.0)
    push_var = tk.DoubleVar(value=DEFAULT_PUSH_FORCE)
    action_vars: list[tk.DoubleVar] = []

    state = {
        "env": None,
        "renderer": None,
        "obs": None,
        "manual_action": np.zeros(2, dtype=np.float32),
        "policy_model": None,
        "policy_algo": None,
        "policy_path": args.policy,
    }

    def format_value(prefix: str, value: float) -> str:
        return f"{prefix}: {value:0.2f}"

    def refresh_numeric_labels() -> None:
        mass_text.set(format_value("Mass", mass_var.get()))
        friction_text.set(format_value("Friction", friction_var.get()))
        push_text.set(format_value("Push |F|", push_var.get()))

    def set_policy_status(message: str) -> None:
        policy_var.set(message)

    def refresh_manual_action() -> None:
        if not action_vars:
            state["manual_action"] = np.zeros(2, dtype=np.float32)
            return
        values = np.array([var.get() for var in action_vars], dtype=np.float32)
        state["manual_action"] = clamp_action_dims(values, state["env"].act_dim)

    def zero_manual_controls() -> None:
        for var in action_vars:
            var.set(0.0)
        refresh_manual_action()

    def validate_policy_against_env() -> None:
        model = state["policy_model"]
        env = state["env"]
        if model is None:
            return
        action_shape = tuple(model.action_space.shape)
        obs_shape = tuple(model.observation_space.shape)
        expected_action_shape = (env.act_dim,)
        expected_obs_shape = (env.obs_dim,)
        if action_shape != expected_action_shape or obs_shape != expected_obs_shape:
            state["policy_model"] = None
            autoplay_var.set(False)
            set_policy_status(
                "Policy unloaded: checkpoint shape does not match current action mode."
            )

    def rebuild_env(*, preserve_policy: bool = True) -> None:
        old_renderer = state["renderer"]
        if old_renderer is not None:
            old_renderer.close()

        state["env"] = make_env(
            action_mode=action_mode_var.get(),
            mass_scale=mass_var.get(),
            friction_scale=friction_var.get(),
        )
        state["renderer"] = mujoco.Renderer(
            state["env"].model, height=RENDER_H, width=RENDER_W
        )
        state["obs"] = state["env"].get_obs()
        refresh_manual_action()
        if not preserve_policy:
            state["policy_model"] = None
            state["policy_algo"] = None
            autoplay_var.set(False)
            set_policy_status("Policy: manual control")
        else:
            validate_policy_against_env()

    def make_slider(
        parent: tk.Widget,
        label: str,
        variable: tk.DoubleVar,
        *,
        from_: float,
        to: float,
        resolution: float,
        textvariable: tk.StringVar | None = None,
        command=None,
        length: int = 240,
    ) -> None:
        row = tk.Frame(parent, bg="#2b2b2b")
        row.pack(fill=tk.X, pady=3)
        tk.Label(
            row,
            text=label,
            font=("Helvetica", 11),
            fg="#e0e0e0",
            bg="#2b2b2b",
            width=18,
            anchor="w",
        ).pack(side=tk.LEFT)
        if textvariable is not None:
            tk.Label(
                row,
                textvariable=textvariable,
                font=("Courier", 10),
                fg="#f0c36b",
                bg="#2b2b2b",
                width=16,
                anchor="w",
            ).pack(side=tk.RIGHT, padx=(8, 0))
        tk.Scale(
            row,
            variable=variable,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            length=length,
            showvalue=False,
            font=("Helvetica", 10),
            bg="#3c3c3c",
            fg="#e0e0e0",
            troughcolor="#555",
            highlightthickness=0,
            command=command,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def on_mass_change(_=None) -> None:
        refresh_numeric_labels()
        if state["env"] is not None:
            state["env"].set_mass(mass_var.get())

    def on_friction_change(_=None) -> None:
        refresh_numeric_labels()
        if state["env"] is not None:
            state["env"].set_friction(friction_var.get())

    def on_push_change(_=None) -> None:
        refresh_numeric_labels()

    def apply_push(direction: float) -> None:
        state["env"].apply_external_force(
            fx=direction * push_var.get(),
            duration_steps=3,
        )

    def set_discrete_action(a: float, b: float) -> None:
        values = np.zeros(max(2, state["env"].act_dim), dtype=np.float32)
        values[0], values[1] = a, b
        state["manual_action"] = clamp_action_dims(values, state["env"].act_dim)
        if action_vars:
            action_vars[0].set(a)
            action_vars[1].set(b)
            for extra_var in action_vars[2:]:
                extra_var.set(0.0)

    def build_dr_controls() -> None:
        refresh_numeric_labels()
        left = tk.Frame(dr_frame, bg="#2b2b2b")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        make_slider(
            left,
            "Mass scale",
            mass_var,
            from_=0.3,
            to=2.0,
            resolution=0.05,
            textvariable=mass_text,
            command=on_mass_change,
        )
        make_slider(
            left,
            "Friction scale",
            friction_var,
            from_=0.1,
            to=2.0,
            resolution=0.05,
            textvariable=friction_text,
            command=on_friction_change,
        )
        make_slider(
            left,
            "Push magnitude",
            push_var,
            from_=5.0,
            to=15.0,
            resolution=0.5,
            textvariable=push_text,
            command=on_push_change,
        )

        right = tk.Frame(dr_frame, bg="#2b2b2b")
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))

        tk.Label(
            right,
            text="Action mode",
            font=("Helvetica", 11),
            fg="#e0e0e0",
            bg="#2b2b2b",
            anchor="w",
        ).pack(fill=tk.X)
        dropdown = ttk.Combobox(
            right,
            textvariable=action_mode_var,
            values=ACTION_MODES,
            state="readonly",
            width=22,
        )
        dropdown.pack(fill=tk.X, pady=(4, 8))
        dropdown.bind("<<ComboboxSelected>>", lambda _event: on_action_mode_change())

        button_row = tk.Frame(right, bg="#2b2b2b")
        button_row.pack(fill=tk.X, pady=(2, 8))
        tk.Button(
            button_row,
            text="Push Left",
            command=lambda: apply_push(-1.0),
            font=("Helvetica", 11),
            bg="#5b88c8",
            fg="white",
            width=10,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            button_row,
            text="Push Right",
            command=lambda: apply_push(+1.0),
            font=("Helvetica", 11),
            bg="#5b88c8",
            fg="white",
            width=10,
        ).pack(side=tk.LEFT)

        tk.Checkbutton(
            right,
            text="Policy autoplay",
            variable=autoplay_var,
            font=("Helvetica", 11),
            fg="#e0e0e0",
            bg="#2b2b2b",
            activebackground="#2b2b2b",
            activeforeground="#e0e0e0",
            selectcolor="#2b2b2b",
        ).pack(anchor="w", pady=(0, 6))

    def build_action_controls() -> None:
        for child in manual_frame.winfo_children():
            child.destroy()
        action_vars.clear()

        if args.mode == "continuous":
            labels = ACTION_LABELS[action_mode_var.get()]
            for idx, label in enumerate(labels):
                value_text = tk.StringVar(value=f"{0.0:+0.2f}")
                var = tk.DoubleVar(value=0.0)
                action_vars.append(var)

                def on_value_change(_=None, index=idx, text=value_text, src_var=var):
                    text.set(f"{src_var.get():+0.2f}")
                    refresh_manual_action()

                make_slider(
                    manual_frame,
                    label,
                    var,
                    from_=-1.0,
                    to=1.0,
                    resolution=0.05,
                    textvariable=value_text,
                    command=on_value_change,
                    length=400,
                )

            tk.Button(
                manual_frame,
                text="Zero",
                command=zero_manual_controls,
                font=("Helvetica", 12),
                bg="#f0a030",
                width=8,
            ).pack(pady=8)
        else:
            btn_cfg = dict(font=("Helvetica", 13), width=14, height=2)
            btn_frame = tk.Frame(manual_frame, bg="#2b2b2b")
            btn_frame.pack(pady=5)
            tk.Button(
                btn_frame,
                text="J1+  J2+",
                command=lambda: set_discrete_action(1.0, 1.0),
                **btn_cfg,
            ).grid(row=0, column=0, padx=4, pady=4)
            tk.Button(
                btn_frame,
                text="J1+  J2-",
                command=lambda: set_discrete_action(1.0, -1.0),
                **btn_cfg,
            ).grid(row=0, column=1, padx=4, pady=4)
            tk.Button(
                btn_frame,
                text="J1-  J2+",
                command=lambda: set_discrete_action(-1.0, 1.0),
                **btn_cfg,
            ).grid(row=1, column=0, padx=4, pady=4)
            tk.Button(
                btn_frame,
                text="J1-  J2-",
                command=lambda: set_discrete_action(-1.0, -1.0),
                **btn_cfg,
            ).grid(row=1, column=1, padx=4, pady=4)
            tk.Button(
                manual_frame,
                text="Zero",
                command=zero_manual_controls,
                font=("Helvetica", 12),
                bg="#f0a030",
                width=8,
            ).pack(pady=8)

        refresh_manual_action()

    def on_action_mode_change() -> None:
        build_action_controls()
        rebuild_env()

    build_dr_controls()
    build_action_controls()
    rebuild_env(preserve_policy=False)

    if args.policy is not None:
        try:
            model, algo_name = load_policy(args.policy, args.policy_algo)
            state["policy_model"] = model
            state["policy_algo"] = algo_name
            validate_policy_against_env()
            if state["policy_model"] is not None:
                set_policy_status(
                    f"Policy: {args.policy.name} ({algo_name.upper()}, autoplay {'on' if autoplay_var.get() else 'off'})"
                )
        except Exception as exc:
            autoplay_var.set(False)
            set_policy_status(f"Policy load failed: {type(exc).__name__}: {exc}")

    def reset_env() -> None:
        state["obs"] = state["env"].reset()
        zero_manual_controls()

    tk.Button(
        root,
        text="Reset",
        command=reset_env,
        font=("Helvetica", 12),
        bg="#cc6040",
        fg="white",
        width=8,
    ).pack(pady=(0, 10))

    def get_active_action() -> np.ndarray:
        model = state["policy_model"]
        if autoplay_var.get() and model is not None:
            action, _ = model.predict(state["obs"], deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            return clamp_action_dims(action, state["env"].act_dim)
        return state["manual_action"]

    def tick() -> None:
        action = get_active_action()
        obs, _, terminated, truncated, info = state["env"].step(action)
        state["obs"] = obs
        if terminated or truncated:
            state["obs"] = state["env"].reset()

        state["renderer"].update_scene(state["env"].data, camera="side")
        frame = state["renderer"].render()
        tk_img = ImageTk.PhotoImage(Image.fromarray(frame))
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        tk_img_ref[0] = tk_img

        data = state["env"].data
        j1_a = np.rad2deg(data.qpos[3])
        j2_a = np.rad2deg(data.qpos[4])
        j1_v, j2_v = data.qvel[3], data.qvel[4]
        x = float(info["x"])
        info_var.set(
            f"mode={action_mode_var.get():<20} "
            f"j1={j1_a:+6.1f}° j2={j2_a:+6.1f}° "
            f"j1_vel={j1_v:+5.1f} j2_vel={j2_v:+5.1f} "
            f"action={np.array2string(action, precision=2, floatmode='fixed')} "
            f"x={x:0.3f}"
        )

        if state["policy_model"] is not None:
            set_policy_status(
                f"Policy: {state['policy_path'].name} ({state['policy_algo'].upper()}, autoplay {'on' if autoplay_var.get() else 'off'})"
            )
        root.after(FPS_MS, tick)

    root.after(1, tick)
    print(f"Teleop started ({args.mode} mode). Close the window to quit.")
    try:
        root.mainloop()
    finally:
        if state["renderer"] is not None:
            state["renderer"].close()


if __name__ == "__main__":
    main()
