"""
Crawler 2D MuJoCo environment — shared module for CMM26 RL demos.

This is a *faithful superset* of the inline class originally duplicated in
L6-1 (tabular / DQN) and L6-2 (REINFORCE / actor-critic). It is extended
with the hooks the L6-3 lecture demos need:

  - Domain randomization: ``mass_scale``, ``friction_scale``,
    ``apply_external_force``.
  - Action-space choice: ``action_mode`` in
    ``{'torque', 'pd_target', 'pd_target_with_gains'}`` — reproduces the
    Peng & van de Panne 2017 comparison inside a single class.
  - Reward-engineering choice: ``reward_mode`` in
    ``{'dense_vel', 'sparse_1m', 'shaped'}`` — L6-3b dense/sparse/shaped
    demo.

Backward-compatibility notes
----------------------------
- The ``include_velocity=False`` default matches the L6-1 inline class.
  L6-2 (which assumed a 4-D observation) must now pass
  ``include_velocity=True`` explicitly.
- ``step(ctrl)`` and ``reset()`` return values are unchanged for
  ``action_mode='torque'`` + ``reward_mode='dense_vel'`` (the legacy path).

References
----------
- X. B. Peng & M. van de Panne (2017). *Learning Locomotion Skills Using
  DeepRL: Does the Choice of Action Space Matter?* SCA.
- J. Hwangbo et al. (2019). *Learning agile and dynamic motor skills for
  legged robots.* Science Robotics.
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# MuJoCo model — identical to the inline XML in L6-1 / L6-2 / teleop_crawler.
# Exposed as a module-level constant so downstream notebooks can import it
# directly (some demos modify the XML to add sensors, obstacles, etc).
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Module-level constants (shared with downstream demos)
# ---------------------------------------------------------------------------

#: Joint indices in ``data.qpos`` / ``data.qvel`` for the two actuated joints.
#: ``qpos`` layout: [root_x, root_z, root_rot, arm_joint, hand_joint].
JOINT_QPOS_IDX = (3, 4)

#: Joint range in radians, matching the ``range="-70 70"`` attribute in XML.
JOINT_ANGLE_LIMIT = np.deg2rad(70.0)

#: Actuator gear ratios from the XML (arm_motor, hand_motor).
ACTUATOR_GEAR = np.array([5.0, 3.0], dtype=np.float64)

#: Default nominal floor-sliding-friction value from the XML ``<default>`` block.
NOMINAL_FRICTION_SLIDING = 1.5

#: Valid action modes for :class:`CrawlerEnv`.
ACTION_MODES = ("torque", "pd_target", "pd_target_with_gains")

#: Valid reward modes for :class:`CrawlerEnv`.
REWARD_MODES = ("dense_vel", "sparse_1m", "shaped")


class CrawlerEnv:
    """2D MuJoCo crawler with two actuated joints (arm and hand).

    State
        Joint angles (and optionally velocities and torso x-position /
        velocity) — always continuous.
    Action
        Semantics depend on ``action_mode``:

        - ``'torque'``: 2-D torque in ``[-1, 1]``, passed to the MuJoCo
          motor actuators directly.
        - ``'pd_target'``: 2-D target joint angle in ``[-1, 1]`` (scaled
          to the joint limit), tracked by an in-env PD controller with
          fixed ``pd_kp`` / ``pd_kd``.
        - ``'pd_target_with_gains'``: 4-D vector ``[target_arm,
          target_hand, gain_arm, gain_hand]``. Gains in ``[-1, 1]``
          modulate ``pd_kp`` by up to ±50 %.
    Reward
        Semantics depend on ``reward_mode``:

        - ``'dense_vel'``: forward x-velocity ``(x_after - x_before)/dt``.
        - ``'sparse_1m'``: 1.0 at the first step where ``x >= 1.0``,
          otherwise 0; terminates the episode on crossing.
        - ``'shaped'``: ``dense_vel - shaping_coef * sum(torque**2)``.
    """

    # ---- construction -----------------------------------------------------

    def __init__(
        self,
        include_velocity: bool = False,
        include_torso_position: bool = False,
        include_torso_velocity: bool = False,
        max_steps: int = 500,
        frame_skip: int = 4,
        # --- DR hooks (new for L6-3) --------------------------------------
        mass_scale: float = 1.0,
        friction_scale: float = 1.0,
        action_noise_std: float = 0.0,
        # --- action-space hook (new for L6-3) -----------------------------
        action_mode: str = "torque",
        pd_kp: float = 4.0,
        pd_kd: float = 0.2,
        # --- reward hook (new for L6-3) -----------------------------------
        reward_mode: str = "dense_vel",
        shaping_coef: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        if action_mode not in ACTION_MODES:
            raise ValueError(
                f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}"
            )
        if reward_mode not in REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}"
            )

        self.model = mujoco.MjModel.from_xml_string(CRAWLER_XML)
        self.data = mujoco.MjData(self.model)

        # --- observation flags --------------------------------------------
        self.include_velocity = include_velocity
        self.include_torso_position = include_torso_position
        self.include_torso_velocity = include_torso_velocity

        # --- episode bookkeeping ------------------------------------------
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.steps = 0

        # --- DR state ------------------------------------------------------
        self.mass_scale = float(mass_scale)
        self.friction_scale = float(friction_scale)
        self.action_noise_std = float(action_noise_std)
        self._nominal_body_mass = self.model.body_mass.copy()
        self._pending_force = np.zeros(6, dtype=np.float64)
        self._pending_force_steps = 0
        self._torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        # --- action-space state -------------------------------------------
        self.action_mode = action_mode
        self.pd_kp = float(pd_kp)
        self.pd_kd = float(pd_kd)

        # --- reward-mode state --------------------------------------------
        self.reward_mode = reward_mode
        self.shaping_coef = float(shaping_coef)
        self._crossed_1m = False

        # --- discretisation ranges (used by L6-1 tabular path) -------------
        self.angle_lo, self.angle_hi = -JOINT_ANGLE_LIMIT, JOINT_ANGLE_LIMIT
        self.vel_lo, self.vel_hi = -8.0, 8.0
        self.x_lo, self.x_hi = -2.0, 2.0
        self.xdot_lo, self.xdot_hi = -4.0, 4.0

        # --- rng ----------------------------------------------------------
        self._rng = np.random.default_rng(seed)

        # Apply DR parameters on construction so obs_dim etc. are consistent.
        self._apply_dr_to_model()

    # ---- static shape helpers --------------------------------------------

    @property
    def obs_dim(self) -> int:
        """Dimensionality of the observation vector under the current flags."""
        return (
            2
            + (2 if self.include_velocity else 0)
            + (1 if self.include_torso_position else 0)
            + (1 if self.include_torso_velocity else 0)
        )

    @property
    def act_dim(self) -> int:
        """Dimensionality of the action vector under the current action mode."""
        if self.action_mode == "pd_target_with_gains":
            return 4
        return 2

    # ---- observation ------------------------------------------------------

    def get_obs(self) -> np.ndarray:
        """Return the continuous observation under the current flags.

        Layout (in order): ``[arm_angle, hand_angle]`` then, depending on
        flags, ``[arm_vel, hand_vel]``, ``torso_x``, ``torso_xdot``.
        """
        obs = [self.data.qpos[JOINT_QPOS_IDX[0]], self.data.qpos[JOINT_QPOS_IDX[1]]]
        if self.include_velocity:
            obs.extend([self.data.qvel[JOINT_QPOS_IDX[0]], self.data.qvel[JOINT_QPOS_IDX[1]]])
        if self.include_torso_position:
            obs.append(self.data.qpos[0])
        if self.include_torso_velocity:
            obs.append(self.data.qvel[0])
        return np.array(obs, dtype=np.float32)

    def discretize(self, obs: np.ndarray, n_bins: int) -> tuple:
        """Convert a continuous observation to a tuple of bin indices.

        Preserved verbatim from the L6-1 inline class so the tabular /
        DP code path keeps working.
        """
        def _bin(val: float, lo: float, hi: float) -> int:
            val = float(np.clip(val, lo, hi))
            idx = int((val - lo) / (hi - lo) * n_bins)
            return min(idx, n_bins - 1)

        arm_b = _bin(obs[0], self.angle_lo, self.angle_hi)
        hand_b = _bin(obs[1], self.angle_lo, self.angle_hi)
        state = [arm_b, hand_b]
        if self.include_velocity:
            arm_vb = _bin(obs[2], self.vel_lo, self.vel_hi)
            hand_vb = _bin(obs[3], self.vel_lo, self.vel_hi)
            state.extend([arm_vb, hand_vb])
        if self.include_torso_position:
            torso_idx = 4 if self.include_velocity else 2
            torso_b = _bin(obs[torso_idx], self.x_lo, self.x_hi)
            state.append(torso_b)
        if self.include_torso_velocity:
            torso_vel_idx = len(state)
            torso_vb = _bin(obs[torso_vel_idx], self.xdot_lo, self.xdot_hi)
            state.append(torso_vb)
        return tuple(state)

    # ---- dynamics ---------------------------------------------------------

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        # Small randomisation of the joint angles, matching L6-1/L6-2 behaviour.
        self.data.qpos[JOINT_QPOS_IDX[0]] = self._rng.uniform(-0.1, 0.1)
        self.data.qpos[JOINT_QPOS_IDX[1]] = self._rng.uniform(-0.1, 0.1)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self._crossed_1m = False
        self._pending_force_steps = 0
        self._pending_force[:] = 0.0
        # Re-apply DR settings in case caller tweaked mass_scale etc.
        self._apply_dr_to_model()
        return self.get_obs()

    def step(self, ctrl: np.ndarray):
        """Advance one policy step.

        ``ctrl`` semantics depend on :attr:`action_mode` — see the class
        docstring.

        Returns ``(obs, reward, terminated, truncated, info)``.
        """
        ctrl = np.asarray(ctrl, dtype=np.float64).reshape(-1)

        # Optional per-step action noise (used by L6-3d DR demo for the
        # "actuator PWM jitter" flavour of dynamic randomisation).
        if self.action_noise_std > 0.0:
            ctrl = ctrl + self._rng.normal(0.0, self.action_noise_std, size=ctrl.shape)

        x_before = self.data.qpos[0]
        torque_cmd = self._resolve_ctrl(ctrl)
        self.data.ctrl[:] = np.clip(torque_cmd, -1.0, 1.0)

        # External-force injection (push) for the configured number of
        # physics sub-steps.
        if self._pending_force_steps > 0:
            self.data.xfrc_applied[self._torso_body_id, :] = self._pending_force
        else:
            self.data.xfrc_applied[self._torso_body_id, :] = 0.0

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        if self._pending_force_steps > 0:
            self._pending_force_steps -= 1
            if self._pending_force_steps == 0:
                self._pending_force[:] = 0.0
                self.data.xfrc_applied[self._torso_body_id, :] = 0.0

        x_after = self.data.qpos[0]
        dt = self.frame_skip * self.model.opt.timestep
        x_vel = (x_after - x_before) / dt

        terminated = False
        if self.reward_mode == "dense_vel":
            reward = x_vel
        elif self.reward_mode == "sparse_1m":
            if (not self._crossed_1m) and (x_after >= 1.0):
                reward = 1.0
                self._crossed_1m = True
                terminated = True
            else:
                reward = 0.0
        else:  # 'shaped'
            # Penalise raw motor torque to discourage high-frequency jitter.
            energy = float(np.sum(self.data.ctrl[:2] ** 2))
            reward = x_vel - self.shaping_coef * energy

        self.steps += 1
        truncated = self.steps >= self.max_steps
        return self.get_obs(), float(reward), terminated, truncated, {"x": x_after}

    # ---- domain-randomisation hooks --------------------------------------

    def set_mass(self, scale: float) -> None:
        """Set the torso-and-limbs mass multiplier, then re-apply to the model.

        Takes effect immediately for the current episode — the next
        :meth:`step` will use the new mass.
        """
        self.mass_scale = float(scale)
        self._apply_dr_to_model()

    def set_friction(self, scale: float) -> None:
        """Set the floor-sliding-friction multiplier.

        Takes effect immediately for the current episode.
        """
        self.friction_scale = float(scale)
        self._apply_dr_to_model()

    def apply_external_force(
        self, fx: float, fz: float = 0.0, duration_steps: int = 1
    ) -> None:
        """Queue an external force on the torso body.

        The force is applied for ``duration_steps`` *policy* steps (each
        of which runs ``frame_skip`` physics sub-steps). Used by L6-3d to
        implement the "push-it-and-watch" DR demo.
        """
        self._pending_force[:] = 0.0
        self._pending_force[0] = float(fx)
        self._pending_force[2] = float(fz)
        self._pending_force_steps = max(1, int(duration_steps))

    def sample_dr(
        self,
        mass_range: tuple = (0.7, 1.3),
        friction_range: tuple = (0.7, 1.3),
        push_prob: float = 0.0,
        push_mag: float = 0.0,
    ) -> None:
        """Sample a domain-randomisation configuration in place.

        Intended to be called inside a training loop before each episode
        (or inside :meth:`reset`) to mimic the Assignment 3 ``domain_rand``
        flags: ``randomize_friction``, ``randomize_base_mass``,
        ``push_robots``.
        """
        self.set_mass(float(self._rng.uniform(*mass_range)))
        self.set_friction(float(self._rng.uniform(*friction_range)))
        if push_prob > 0.0 and self._rng.random() < push_prob:
            direction = 1.0 if self._rng.random() < 0.5 else -1.0
            self.apply_external_force(
                fx=direction * push_mag, fz=0.0, duration_steps=1
            )

    # ---- internal helpers -------------------------------------------------

    def _apply_dr_to_model(self) -> None:
        """Push the current DR parameters into the underlying MuJoCo model."""
        # Scale the non-world bodies' mass (world body is index 0).
        self.model.body_mass[1:] = self._nominal_body_mass[1:] * self.mass_scale
        # Scale the floor's sliding friction. geom_friction is (N,3):
        # (slide, torsional, rolling). Only the slide component is
        # physically meaningful for the crawler demo.
        self.model.geom_friction[self._floor_geom_id, 0] = (
            NOMINAL_FRICTION_SLIDING * self.friction_scale
        )

    def _resolve_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Convert a high-level action to the 2-D torque-in-[-1,1] signal
        expected by the two ``motor`` actuators in the XML.

        - ``'torque'`` → identity (clipped).
        - ``'pd_target'`` → PD tracking with fixed gains.
        - ``'pd_target_with_gains'`` → PD tracking with per-joint gain
          modulation (action[2:4] modulates ``pd_kp`` by up to ±50 %).
        """
        if self.action_mode == "torque":
            if action.size < 2:
                raise ValueError("torque action must have at least 2 components")
            return action[:2]

        # Shared PD-tracking logic for the two PD modes.
        q_arm = float(self.data.qpos[JOINT_QPOS_IDX[0]])
        q_hand = float(self.data.qpos[JOINT_QPOS_IDX[1]])
        qd_arm = float(self.data.qvel[JOINT_QPOS_IDX[0]])
        qd_hand = float(self.data.qvel[JOINT_QPOS_IDX[1]])

        target_arm = float(np.clip(action[0], -1.0, 1.0)) * JOINT_ANGLE_LIMIT
        target_hand = float(np.clip(action[1], -1.0, 1.0)) * JOINT_ANGLE_LIMIT

        if self.action_mode == "pd_target_with_gains":
            if action.size < 4:
                raise ValueError(
                    "pd_target_with_gains action must have 4 components"
                )
            # Map action[2:4] ∈ [-1, 1] to a multiplicative factor in [0.5, 1.5].
            kp_arm = self.pd_kp * (1.0 + 0.5 * float(np.clip(action[2], -1.0, 1.0)))
            kp_hand = self.pd_kp * (1.0 + 0.5 * float(np.clip(action[3], -1.0, 1.0)))
        else:
            kp_arm = self.pd_kp
            kp_hand = self.pd_kp

        torque_arm = kp_arm * (target_arm - q_arm) - self.pd_kd * qd_arm
        torque_hand = kp_hand * (target_hand - q_hand) - self.pd_kd * qd_hand

        # Convert joint-space torque → motor ctrl in [-1, 1] (divide by gear).
        ctrl_arm = torque_arm / ACTUATOR_GEAR[0]
        ctrl_hand = torque_hand / ACTUATOR_GEAR[1]
        return np.array([ctrl_arm, ctrl_hand], dtype=np.float64)


__all__ = [
    "CrawlerEnv",
    "CRAWLER_XML",
    "ACTION_MODES",
    "REWARD_MODES",
    "JOINT_ANGLE_LIMIT",
    "ACTUATOR_GEAR",
]
