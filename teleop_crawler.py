"""
Interactive Teleoperation Demo for the 2D MuJoCo Crawler.

Two modes (pick with --mode):
  continuous  : two sliders for arm/hand torque [-1, 1]
  discrete    : four buttons for the 4 coarse actions (+/- combos)

Usage:
  python teleop_crawler.py                  # continuous (default)
  python teleop_crawler.py --mode discrete  # discrete
"""

import argparse
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk

import mujoco
import numpy as np

# ── MuJoCo model (same as notebook) ─────────────────────────

CRAWLER_XML = """
<mujoco model="crawler2d">
  <compiler angle="degree" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>

  <default>
    <geom conaffinity="1" condim="3" friction="1.5 0.5 0.1" density="1000"/>
    <joint armature="0.1" damping="0.5"/>
  </default>

  <asset>
    <texture type="2d" name="grid" builtin="checker" width="512" height="512"
             rgb1="0.7 0.9 0.7" rgb2="0.6 0.85 0.6"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>

  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 -2 3" dir="0 0.5 -1"/>
    <geom name="floor" type="plane" size="50 1 0.1" material="grid"/>

    <!-- Ruler: distance markers along x-axis -->
    <geom name="origin" type="box" size="0.01 0.15 0.002" pos="0 0 0.001" rgba="0.1 0.1 0.8 0.7" contype="0" conaffinity="0"/>
    <geom name="ruler_m4" type="box" size="0.006 0.12 0.002" pos="-2.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_m3" type="box" size="0.003 0.08 0.001" pos="-1.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_m2" type="box" size="0.006 0.12 0.002" pos="-1.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_m1" type="box" size="0.003 0.08 0.001" pos="-0.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_1" type="box" size="0.003 0.08 0.001" pos="0.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_2" type="box" size="0.006 0.12 0.002" pos="1.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_3" type="box" size="0.003 0.08 0.001" pos="1.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_4" type="box" size="0.006 0.12 0.002" pos="2.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_5" type="box" size="0.003 0.08 0.001" pos="2.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_6" type="box" size="0.006 0.12 0.002" pos="3.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_7" type="box" size="0.003 0.08 0.001" pos="3.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_8" type="box" size="0.006 0.12 0.002" pos="4.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_9" type="box" size="0.003 0.08 0.001" pos="4.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_10" type="box" size="0.006 0.12 0.002" pos="5.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_11" type="box" size="0.003 0.08 0.001" pos="5.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_12" type="box" size="0.006 0.12 0.002" pos="6.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_13" type="box" size="0.003 0.08 0.001" pos="6.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_14" type="box" size="0.006 0.12 0.002" pos="7.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_15" type="box" size="0.003 0.08 0.001" pos="7.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_16" type="box" size="0.006 0.12 0.002" pos="8.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_17" type="box" size="0.003 0.08 0.001" pos="8.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_18" type="box" size="0.006 0.12 0.002" pos="9.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_19" type="box" size="0.003 0.08 0.001" pos="9.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_20" type="box" size="0.006 0.12 0.002" pos="10.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_21" type="box" size="0.003 0.08 0.001" pos="10.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_22" type="box" size="0.006 0.12 0.002" pos="11.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_23" type="box" size="0.003 0.08 0.001" pos="11.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_24" type="box" size="0.006 0.12 0.002" pos="12.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_25" type="box" size="0.003 0.08 0.001" pos="12.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_26" type="box" size="0.006 0.12 0.002" pos="13.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_27" type="box" size="0.003 0.08 0.001" pos="13.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_28" type="box" size="0.006 0.12 0.002" pos="14.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_29" type="box" size="0.003 0.08 0.001" pos="14.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_30" type="box" size="0.006 0.12 0.002" pos="15.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_31" type="box" size="0.003 0.08 0.001" pos="15.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_32" type="box" size="0.006 0.12 0.002" pos="16.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_33" type="box" size="0.003 0.08 0.001" pos="16.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_34" type="box" size="0.006 0.12 0.002" pos="17.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_35" type="box" size="0.003 0.08 0.001" pos="17.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_36" type="box" size="0.006 0.12 0.002" pos="18.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_37" type="box" size="0.003 0.08 0.001" pos="18.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_38" type="box" size="0.006 0.12 0.002" pos="19.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>
    <geom name="ruler_39" type="box" size="0.003 0.08 0.001" pos="19.5 0 0.001" rgba="0.25 0.25 0.25 0.4" contype="0" conaffinity="0"/>
    <geom name="ruler_40" type="box" size="0.006 0.12 0.002" pos="20.0 0 0.001" rgba="0.15 0.15 0.15 0.6" contype="0" conaffinity="0"/>

    <camera name="side" pos="0 -0.8 0.25" xyaxes="1 0 0 0 0.3 1" mode="trackcom"/>

    <body name="torso" pos="0 0 0.035">
      <joint name="root_x" type="slide" axis="1 0 0"/>
      <joint name="root_z" type="slide" axis="0 0 1"/>
      <joint name="root_rot" type="hinge" axis="0 1 0"/>

      <geom name="torso_geom" type="box" size="0.08 0.035 0.025"
            rgba="0.3 0.75 0.3 1" density="3000"/>

      <body name="arm" pos="0.08 0 0.01">
        <joint name="arm_joint" type="hinge" axis="0 1 0"
               range="-70 70" limited="true"/>
        <geom name="arm_geom" type="capsule" size="0.012"
              fromto="0 0 0 0.12 0 0" rgba="0.95 0.7 0.1 1"/>

        <body name="hand" pos="0.12 0 0">
          <joint name="hand_joint" type="hinge" axis="0 1 0"
                 range="-70 70" limited="true"/>
          <geom name="hand_geom" type="capsule" size="0.008"
                fromto="0 0 0 0.08 0 0" rgba="0.9 0.15 0.15 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="arm_motor" joint="arm_joint" ctrllimited="true"
           ctrlrange="-1 1" gear="5"/>
    <motor name="hand_motor" joint="hand_joint" ctrllimited="true"
           ctrlrange="-1 1" gear="3"/>
  </actuator>
</mujoco>
"""

# ── Shared state ─────────────────────────────────────────────

ctrl = np.array([0.0, 0.0])

RENDER_W, RENDER_H = 640, 360


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Teleoperate the 2D crawler")
    parser.add_argument("--mode", choices=["continuous", "discrete"],
                        default="continuous")
    args = parser.parse_args()

    # MuJoCo setup
    model = mujoco.MjModel.from_xml_string(CRAWLER_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[3] = np.random.uniform(-0.1, 0.1)
    data.qpos[4] = np.random.uniform(-0.1, 0.1)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    # ── Tkinter window ──────────────────────────────────────
    root = tk.Tk()
    root.title(f"Crawler Teleop — {args.mode}")
    root.configure(bg="#2b2b2b")

    # Render canvas
    canvas = Canvas(root, width=RENDER_W, height=RENDER_H,
                    bg="black", highlightthickness=0)
    canvas.pack(padx=10, pady=(10, 5))
    tk_img_ref = [None]  # prevent GC

    # Info label
    info_var = tk.StringVar(value="")
    tk.Label(root, textvariable=info_var, font=("Courier", 12),
             fg="#e0e0e0", bg="#2b2b2b", justify=tk.LEFT, anchor="w"
             ).pack(fill=tk.X, padx=15, pady=5)

    # ── Controls ────────────────────────────────────────────
    ctrl_frame = tk.Frame(root, bg="#2b2b2b")
    ctrl_frame.pack(padx=10, pady=5, fill=tk.X)

    if args.mode == "continuous":
        tk.Label(ctrl_frame, text="Joint 1 torque", font=("Helvetica", 13),
                 fg="#e0e0e0", bg="#2b2b2b").pack(pady=(5, 0))
        j1_var = tk.DoubleVar(value=0.0)
        tk.Scale(
            ctrl_frame, variable=j1_var, from_=-1.0, to=1.0,
            resolution=0.05, orient=tk.HORIZONTAL, length=400,
            font=("Helvetica", 11), bg="#3c3c3c", fg="#e0e0e0",
            troughcolor="#555", highlightthickness=0,
            command=lambda _: ctrl.__setitem__(0, j1_var.get()),
        ).pack()

        tk.Label(ctrl_frame, text="Joint 2 torque", font=("Helvetica", 13),
                 fg="#e0e0e0", bg="#2b2b2b").pack(pady=(5, 0))
        j2_var = tk.DoubleVar(value=0.0)
        tk.Scale(
            ctrl_frame, variable=j2_var, from_=-1.0, to=1.0,
            resolution=0.05, orient=tk.HORIZONTAL, length=400,
            font=("Helvetica", 11), bg="#3c3c3c", fg="#e0e0e0",
            troughcolor="#555", highlightthickness=0,
            command=lambda _: ctrl.__setitem__(1, j2_var.get()),
        ).pack()

        def zero():
            j1_var.set(0.0)
            j2_var.set(0.0)
            ctrl[:] = 0.0

        tk.Button(ctrl_frame, text="Zero", command=zero,
                  font=("Helvetica", 12), bg="#f0a030", width=8
                  ).pack(pady=8)

    else:  # discrete
        btn_cfg = dict(font=("Helvetica", 13), width=14, height=2)
        btn_frame = tk.Frame(ctrl_frame, bg="#2b2b2b")
        btn_frame.pack(pady=5)

        def make_cmd(a, b):
            def cmd():
                ctrl[0], ctrl[1] = a, b
            return cmd

        tk.Button(btn_frame, text="J1+  J2+", command=make_cmd(1, 1),
                  **btn_cfg).grid(row=0, column=0, padx=4, pady=4)
        tk.Button(btn_frame, text="J1+  J2-", command=make_cmd(1, -1),
                  **btn_cfg).grid(row=0, column=1, padx=4, pady=4)
        tk.Button(btn_frame, text="J1-  J2+", command=make_cmd(-1, 1),
                  **btn_cfg).grid(row=1, column=0, padx=4, pady=4)
        tk.Button(btn_frame, text="J1-  J2-", command=make_cmd(-1, -1),
                  **btn_cfg).grid(row=1, column=1, padx=4, pady=4)
        tk.Button(ctrl_frame, text="Zero", command=make_cmd(0, 0),
                  font=("Helvetica", 12), bg="#f0a030", width=8
                  ).pack(pady=8)

    # Reset button
    def reset():
        mujoco.mj_resetData(model, data)
        data.qpos[3] = np.random.uniform(-0.1, 0.1)
        data.qpos[4] = np.random.uniform(-0.1, 0.1)
        mujoco.mj_forward(model, data)
        ctrl[:] = 0.0
        if args.mode == "continuous":
            j1_var.set(0.0)
            j2_var.set(0.0)

    tk.Button(root, text="Reset", command=reset,
              font=("Helvetica", 12), bg="#cc6040", fg="white", width=8
              ).pack(pady=(0, 10))

    # ── Tick loop ───────────────────────────────────────────
    def tick():
        # Step physics
        data.ctrl[:] = np.clip(ctrl, -1, 1)
        for _ in range(4):
            mujoco.mj_step(model, data)

        # Render to canvas
        renderer.update_scene(data, camera="side")
        frame = renderer.render()
        pil_img = Image.fromarray(frame)
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        tk_img_ref[0] = tk_img  # prevent GC

        # Update info
        j1_a = np.rad2deg(data.qpos[3])
        j2_a = np.rad2deg(data.qpos[4])
        j1_v, j2_v = data.qvel[3], data.qvel[4]
        x = data.qpos[0]
        info_var.set(
            f"j1={j1_a:+6.1f}°  j2={j2_a:+6.1f}°  "
            f"j1_vel={j1_v:+5.1f}  j2_vel={j2_v:+5.1f}   "
            f"action=[{ctrl[0]:+.2f}, {ctrl[1]:+.2f}]   x={x:.3f}"
        )

        root.after(16, tick)  # ~60 fps

    root.after(1, tick)
    print(f"Teleop started ({args.mode} mode). Close the window to quit.")
    root.mainloop()
    renderer.close()


if __name__ == "__main__":
    main()
