# CMM 2026 RL Lectures -- Project Context

## Who & Why

**Instructor/TA:** Yijiang Huang, TA for "Computational Models of Motion" (CMM) at ETH Zurich (CRL lab).
**Course focus:** Applied optimization and computational tools for animated character and robot motion. RL is covered in two guest lectures by Yijiang.
**Student background:** Graduate-level, strong in optimization/physics/numerics, but NOT RL specialists. They have already done a hexpod walking competition using trajectory optimization + PD control earlier in the course.

## Feedback from 2025

Last year's two RL lectures (57 + 70 slides) tried to cover too many concepts too fast. Students felt overwhelmed. For 2026, the goals are:
- **Fewer concepts, deeper dives** -- ~2 algorithms per major section, explained well
- **More interactive demos** -- live Jupyter notebooks (Colab-compatible) shown in class, with a "works then breaks" progression to motivate each next algorithm
- **Less math-heavy** -- keep derivations but emphasize intuition, diagrams, and running code over walls of equations
- Materials adapted from Pieter Abbeel's DeepRL Foundation YT series (recommended to students)

## Two-Lecture Structure (2026)

### Lecture 1: Foundation of RL pt.1 -- MDPs, Value-based, and Policy-based Methods

**Outline:**
1. Motivation (slides 1-20 from 2025 lecture 1, largely unchanged)
   - Hexpod competition results from earlier in the course
   - Feedforward control limitations -> need physics + online actions
   - Optimization-based control -> RL motivation
   - RL = learn emerging behaviour through interaction
   - Supervised Learning vs RL comparison
   - Deep RL highlights timeline (2013 DQN -> 2025 Humanoid Kung-Fu)
   - "Typical skeleton of an applied RL paper"
2. Markov Decision Processes (MDPs) (slides 22-29)
   - Agent-Environment loop (Sutton & Barto diagram)
   - MDP definition: S, A, P(s'|s,a), R(s,a,s'), s_0, gamma, H
   - Modeling examples: Grid-world, **2D Crawler**, Hopper, Humanoid
   - Solving MDPs: policy, optimal policy, deterministic vs stochastic
3. Exact Solution Methods (slides 30-35) + **Demo 0**
   - Value function V^pi(s), Optimal value function V*(s)
   - Connection to dynamic programming (optimal substructure, overlapping subproblems)
   - Bellman Optimality Equation -> Value Iteration algorithm
   - Value Iteration in action (grid-world visualization)
   - **[DEMO 0] Value Iteration on 2D Crawler** -- builds model via teleportation, VI converges, but policy barely works (Markov property violated by angle-only state)
   - Policy Iteration (evaluate -> improve cycle)
   - **Key limitations:** need state sweeping, need dynamics model, need correct state representation

**--- Break ---**

4. Value-based Methods (slides 36-45)
   - Q-Values and Q-Bellman equation
   - Q-Value Iteration -> what if no dynamics model?
   - Tabular Q-Learning: replace expectation with samples, running average update
   - epsilon-greedy exploration, off-policy learning
   - **Q-learning on Crawler Bot** (live demo moment)
   - "Can Tabular Methods Scale?" -- Crawler 10^2, Hopper 10^10, Humanoid 10^100
   - Approximate Q-Learning: parametrize Q with neural network
   - The deadly triad (function approx + bootstrap + off-policy) -> overestimation bias
   - Solutions: replay buffer, double Q-learning
   - DQN on Atari [DeepMind 2013]
   - **Key limitations:** discrete actions only, high bias

5. Policy-based Methods (slides 46-57 from lecture 1 + slides 9-16 from lecture 2)
   - **Paradigm shift:** value-based (off-policy) vs policy-based (on-policy)
   - Parametrizing stochastic policy: categorical (discrete) vs Gaussian (continuous)
   - Likelihood Ratio Policy Gradient derivation
   - Intuition: increase prob of good paths, decrease prob of bad paths
   - Decompose path into states and actions -> no dynamics model required
   - Likelihood Ratio Gradient Estimate (unbiased)
   - **Variance reduction:** the gradient is unbiased but very noisy
   - Baseline subtraction: still unbiased (proof sketch), reduces variance
   - Baseline choices: constant, time-dependent, state-dependent V^pi(s)
   - **Key limitation:** high variance

6. **NEW in 2026 -- Vanilla Policy Gradient / REINFORCE** (moved from lecture 2 slides 9-16)
   - REINFORCE algorithm (Algorithm 1 pseudocode)
   - Vanilla Policy Gradient training curves (learning rate sensitivity)
   - REINFORCE with baseline (actor-critic version)
   - Advantage function A(s,a) = Q(s,a) - V(s) -- "how much better than average"
   - Estimating V^pi(s): Monte-Carlo vs Bootstrap (TD)
   - REINFORCE with baseline training curve (faster, less noisy)
   - Actor-Critic: critic network update + actor network update
   - **Note:** "can get unstable if network is large" -> motivates TRPO/PPO in lecture 2

7. Summary slide: three pillars with limitations
   - Exact: state exploration + need dynamic model
   - Value-based: discrete-only + high bias
   - Policy-based: high variance
   - Arrow to: Actor-Critic Methods (bridge to lecture 2)

### Lecture 2: Foundation of RL pt.2 -- Practical Algorithms and Considerations

**Outline (2026, to be refined):**
- Recap of key concepts (Policy, Value Function, Q Function) and the three method families
- TRPO -> GAE -> PPO story (trust regions, clipped surrogate, advantage estimation)
- SAC (brief mention, 1-2 slides, no dedicated demo)
- Secret Sauces: state/action space design, domain randomization, reward engineering, imitation reward (DeepMimic), simulator choice
- Model-based RL (brief overview)

## Demo Strategy

### Pedagogical Design Principle
Use the **same custom 2D MuJoCo crawler** throughout all demos. Only change the state/action representation or algorithm. Each demo has a **"works -> breaks"** progression: the algorithm succeeds on one configuration, then visibly fails when complexity increases, motivating the next algorithm.

### Lecture 1 Demo: `demo_crawler_rl.ipynb`
Custom 2D MuJoCo crawler (3-body: torso + arm + hand, 2 hinge joints, constrained to 2D via slide joints).

| Demo | Algorithm | State | Actions | Result | Limitation Shown |
|------|-----------|-------|---------|--------|-----------------|
| 0 | **Value Iteration** | 2D (angles only), 81 states | 4 discrete | Barely works (~random) | Needs model access + correct state representation |
| 1 | Tabular Q-learning | 2D (angles only), 81 states | 4 discrete | Works (~10 reward) | -- |
| 2 | Tabular Q-learning | 4D (angles + velocities), 6561 states | 4 discrete | Breaks | Curse of dimensionality |
| 3 | DQN | 4D continuous | 4 discrete | Works | -- |
| 4 | DQN | 4D continuous | 9/49/121 discretized continuous | Breaks | Discrete actions only |

**Demo 0 details (Value Iteration):** Builds transition model by teleporting crawler to all 81 states x 4 actions = 324 simulator calls. VI converges instantly, but the policy barely moves because the 2D state (angles only, no velocity) violates the Markov property -- same angles with different velocities lead to different outcomes. The model is wrong, so the solution is wrong. This teaches TWO lessons: (1) model access is expensive/impractical, (2) model accuracy matters (garbage in = garbage out). Q-learning (Demo 1) on the same 2D state works much better (~10 reward) because it learns from real trajectories where velocity dynamics are implicitly present.

**Each limitation motivates the next algorithm:**
- VI needs model access -> Q-learning learns from experience
- Tabular Q can't scale -> DQN uses neural network
- DQN needs discrete actions -> Policy Gradient works with continuous actions

### Lecture 2 Demo: `demo_policy_gradient.ipynb`
| Demo | Algorithm | Environment | Key Point |
|------|-----------|-------------|-----------|
| 1 | REINFORCE (from scratch) | Crawler, continuous actions | Policy gradient works with continuous actions |
| 2 | PPO (stable-baselines3) | Crawler | Same env, industrial-strength algorithm |
| 3 | PPO (stable-baselines3) | Ant-v4 (111D state, 8D action) | Same algorithm scales to complex bodies |
| 4 | Reward engineering | Crawler with 3 reward functions | Shows how reward design changes behavior |

## Technical Setup

- **Physics:** MuJoCo (custom XML for 2D crawler, built-in Ant-v4)
- **RL framework:** PyTorch (manual DQN + REINFORCE), stable-baselines3 (PPO)
- **Environment wrapper:** Gymnasium
- **Target platform:** Google Colab (also runnable locally)
- **Local venv:** `.venv/` with Python 3.13 (from miniconda), mujoco 3.6.0, torch 2.11, gymnasium 1.2.3, stable-baselines3 2.7.1

## Slide-to-Content Mapping (2025 -> 2026)

### Lecture 1 slides (CMM25_RL_lecture1.pdf, 57 pages)
| 2025 Slides | Content | 2026 Disposition |
|-------------|---------|------------------|
| 1-2 | Title + Outline | Update outline to include REINFORCE |
| 3-8 | Hexpod results + motivation videos | Keep (update with 2026 competition results) |
| 9-10 | Feedforward + optimization-based control | Keep |
| 11-12 | RL intro + SL vs RL | Keep |
| 13-20 | Deep RL highlights timeline | Keep, possibly update with 2025-2026 entries |
| 21 | "Typical skeleton of applied RL paper" | Keep |
| 22-29 | MDP definition + examples | Keep, emphasize 2D crawler example (slide 26) |
| 30-35 | Exact methods (Bellman, VI, PI) | Keep lighter on math + **Demo 0: VI on crawler** |
| 36-40 | Q-learning (tabular) | Keep + **add live demo** |
| 41-42 | Crawler bot + "Can tabular scale?" | Keep + **live demo showing it breaks** |
| 43-45 | Approximate Q / DQN | Keep + **add live demo** |
| 46-48 | Policy-based paradigm shift | Keep |
| 49-53 | Policy gradient derivation | Keep but reduce math density |
| 54-56 | Variance reduction + baseline | Keep |
| 57 | Summary | Update to include REINFORCE |

### Lecture 2 slides 1-16 (CMM25_RL_lecture2.pdf) -> MOVED to Lecture 1
| 2025 Slides | Content | 2026 Disposition |
|-------------|---------|------------------|
| 1-8 | Title + Outline + Recap | Remove (now in lecture 1) |
| 9 | Outline with REINFORCE highlighted | Absorbed into lecture 1 |
| 10 | REINFORCE algorithm + pseudocode | Move to lecture 1 |
| 11 | VPG training curves (learning rate) | Move to lecture 1 |
| 12 | VPG with baseline (actor-critic) | Move to lecture 1 |
| 13 | Advantage function definition | Move to lecture 1 |
| 14 | Estimating V^pi (MC vs Bootstrap) | Move to lecture 1 |
| 15 | VPG with baseline training curve | Move to lecture 1 |
| 16 | Actor-critic algorithm details | Move to lecture 1 (ends with "unstable if large" -> bridge to L2) |

## Key References & Resources
- Pieter Abbeel's DeepRL Foundation YT series (primary adaptation source)
- Sutton & Barto, Reinforcement Learning: An Introduction (policy gradient: chap 13)
- Daniel Seita's notes on policy gradient and GAE
- RL guy's note on training locomotion with SAC (araffin.github.io)
- The real crawling robot for Q-learning (futurismo.biz)
- See `wishlist.md` for additional TODOs and references

## Files in This Directory
- `CMM25_RL_lecture1.pdf/pptx` -- 2025 lecture 1 slides (57 pages)
- `CMM25_RL_lecture2.pdf/pptx` -- 2025 lecture 2 slides (70 pages)
- `demo_crawler_rl.ipynb` -- Lecture 1 Colab notebook (Q-learning + DQN)
- `demo_policy_gradient.ipynb` -- Lecture 2 Colab notebook (REINFORCE + PPO)
- `wishlist.md` -- TODOs and reference links
- `CONTEXT.md` -- this file
