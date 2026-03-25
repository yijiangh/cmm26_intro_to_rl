# rl_lectures

Teleop demo for crawler:
```
.venv/bin/python teleop_crawler.py --mode continuous
.venv/bin/python teleop_crawler.py --mode discrete
```

## Physics notes (useful for in-class discussion)

### Why does the robot sag at zero torque?
The robot is **not** in static equilibrium at zero torque. Gravity constantly pulls the arm/hand joints downward. Joint damping (`damping=0.5`) makes the settling very slow (overdamped), so it *looks* static but is actually drifting toward the ground. Holding a pose requires active torque — exactly the kind of thing an RL policy needs to learn.

### Why does it snap to a position at max torque then stabilize?
When you slam torque to ±1, the motor applies a constant force (via the gear ratio — 5× for arm, 3× for hand). As the joint accelerates, damping (`F = -0.5 * velocity`) grows and opposes motion. Eventually damping + gravity balances the motor torque, and the joint reaches terminal velocity then stops. This is overdamped dynamics — like dropping a ball in honey.

### Why doesn't reducing torque from max cause the arm to move?
At max torque the arm hits the **joint limit** (+70°). The joint limit is a hard constraint — MuJoCo applies whatever force is needed to prevent the joint from exceeding it. Any positive torque, no matter how small, keeps the arm pressed against this wall. Gravity at 70° is weak (~0.01 Nm) while even the slider's smallest step (0.05 × gear 5 = 0.25 Nm) is 20× that. Only at exactly zero torque does gravity win and pull the arm away from the stop.

### What does the action space represent?
The actions are **torques**, not target positions. The MuJoCo XML uses `motor` actuators which apply torque directly to each joint, scaled by a gear ratio:
- `ctrl[0] = +1.0` → +5 Nm on joint 1 (gear=5)
- `ctrl[1] = +1.0` → +3 Nm on joint 2 (gear=3)

The 4 discrete actions are bang-bang torque commands (full ±1 on each joint). Where the joint ends up depends on the dynamic balance of motor torque, gravity, damping, and contacts — there is no target angle.

MuJoCo also supports `position` actuators (PD control to a target angle), which is what many real servos use. This model uses torque control, which is more common in RL research.

