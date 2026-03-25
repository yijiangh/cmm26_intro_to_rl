# Lecture 1 Script: Foundation of RL pt.1

> Per-slide speaking notes for CMM 2026 RL Lecture 1.
> Sections marked **[DEMO]** are live notebook moments.
> Slide numbers reference 2025 deck (CMM25_RL_lecture1.pdf) unless noted as NEW.

---

## Part 1: Motivation (slides 1-21)

*[These slides are well-established from 2025. Brief notes below.]*

### Slide 1-2: Title + Outline
> Update the outline slide to include REINFORCE at the end.

### Slides 3-8: Hexpod competition + motivation videos
> Recap what students did in the hexpod walking competition.
> "You already solved locomotion with trajectory optimization + PD control. Today: what if we DON'T have a good dynamics model, or the problem is too complex for trajectory optimization?"

### Slides 9-10: Feedforward -> optimization-based -> RL
> "RL = learn emerging behaviour through interaction with the environment."

### Slides 11-12: RL intro, SL vs RL comparison
> Key difference: no labeled data, delayed reward, exploration-exploitation.

### Slides 13-20: Deep RL highlights timeline
> Breeze through: DQN Atari (2013) -> AlphaGo (2016) -> OpenAI Five (2019) -> DeepMimic -> recent highlights.
> "By the end of these two lectures, you'll understand every algorithm behind these results."

### Slide 21: Typical skeleton of an applied RL paper
> Frame: MDP formulation -> reward design -> domain sprinkle -> simulator -> algorithm.
> "This is the recipe. Let's understand each ingredient."

---

## Part 2: Markov Decision Processes (slides 22-29)

### Slide 22: Outline (MDP section highlighted)

### Slide 23: MDP diagram (Sutton & Barto)
> "The agent observes the state, takes an action, receives a reward, and lands in a new state. That's the full loop."
> "Key assumption: the state tells you everything you need -- Markov property."

### Slide 24: MDP definition
> List: S, A, P(s'|s,a), R(s,a,s'), s_0, gamma, H.
> "Don't memorize the notation. The intuition is: states, actions, what happens when you act, how good it is, and how far ahead you care."

### Slide 25: Grid-world example
> Walk through: 12 cells, 4 actions, stochastic transitions (noise=0.2), +1/-1 terminal states.
> "Goal: find policy pi that maximizes expected discounted sum of rewards."

### Slide 26: 2D Crawler example (KEY SLIDE)
> "This is our running example for the rest of both lectures. A planar robot with a torso, an arm, and a hand -- 2 joints."
> "State: arm angle and hand angle (discretized). Action: fixed +/- torque on each joint (4 options). Reward: speed in the forward direction."
> "Transition function is crossed out -- we'll come back to why."

**[DEMO: Show the crawler at rest in the notebook. Run the visualization cell so students see what they're working with.]**

### Slide 27-28: Hopper and Humanoid examples
> "Same MDP framework, just bigger. Hopper: 11D state, 3D action. Humanoid: 348D state, 17D action."
> Purpose: plant the seed for "can our algorithms scale?"

### Slide 29: Solving MDPs
> "Our goal: find pi* that maximizes expected sum of discounted rewards."
> "A policy can be deterministic pi(s) = a or stochastic pi(a|s) = probability."

---

## Part 3: Exact Solution Methods (slides 30-35)

### Slide 30: Outline (Exact Solution Method highlighted)

### Slide 31: Value Function V^pi(s)
> "Before we can find the best policy, we need a way to evaluate how good a policy is."
> "V^pi(s) = expected sum of discounted rewards, starting from state s, acting according to policy pi."
> "Think of it as: if I'm in state s and I follow this game plan forever, how much total reward do I expect?"
> Intuition: "It's a prediction of future reward. High V means good state to be in."

### Slide 32: Optimal Value Function V*(s)
> "V*(s) = the value when you act OPTIMALLY. The best you could ever do from state s."
> **Connect to dynamic programming:** "You've seen this before in this course. Remember shortest-path? Same structure."
> - Optimal substructure: if A->E->M is optimal, then A->E and E->M are each optimal
> - Overlapping subproblems: V(v) = min_u d(u,v) + V(u)
> "The Bellman equation is literally the same idea applied to MDPs."

### Slide 33: Bellman Optimality Equation + Value Iteration
> **Bellman equation:** V*(s) = max_a sum_{s'} P(s'|s,a) [R(s,a,s') + gamma * V*(s')]
> "Read it as: the value of a state equals the best action's expected immediate reward plus discounted future value."
> **Value Iteration algorithm:**
> - Init V_0(s) = 0 for all s
> - For k = 1, 2, ...: update V_k using V_{k-1}
> - Extract policy: pi*(s) = argmax_a [sum over s' of P(s'|s,a)(R + gamma V*(s'))]
>
> "Three things to notice (in red):"
> 1. Need to sweep through ALL states
> 2. Need the transition model P(s'|s,a)
> 3. Convergence guaranteed (V is a contraction)

### Slide 34: Value Iteration in action (grid-world)
> Walk through the grid-world animation: values start at 0, propagate backwards from terminal states.
> "After 100 iterations, we have the optimal value for every cell."
> "Beautiful -- but we needed to sweep every state and we needed to know the transition probabilities."

**[DEMO: Value Iteration on the 2D Crawler]**

> "Let's try this on our crawler. We have 81 states, 4 actions. Can we run VI?"
>
> **Run the VI demo cell (Demo 0).** Walk through three steps:
>
> **Step 1 -- Build the model (the expensive part):**
> "First, we need P(s'|s,a) for every state-action pair. How? We TELEPORT the crawler to each of 81 states, try each of 4 actions, and record what happens. That's 324 simulator calls."
> "In the real world, you can't pick up a robot, freeze it mid-air at angle (30, -15), and test all possible torques."
>
> **Step 2 -- Run VI (the easy part):**
> "Given the model, VI converges in milliseconds. The algorithm itself is beautiful and fast."
> Show the value function heatmap -- it looks structured, not random.
>
> **Step 3 -- Roll out the policy (the disappointing part):**
> "The crawler barely moves! Reward ~0.7, about the same as random."
> "Why? Our 2D state ignores velocity. The same joint angles lead to different outcomes depending on how fast the joints are moving. The Markov property is violated."
> "The model we built is WRONG -- not because VI failed, but because our state representation is incomplete. Garbage in, garbage out."
>
> **Two lessons from this demo:**
> 1. **Model access is expensive** -- we needed to teleport to every state (324 calls for 81 states; 26K for 4D; impossible for humanoid)
> 2. **Model accuracy matters** -- even with perfect access, if the state doesn't capture the full dynamics, the solution is useless
>
> "Next: Q-learning avoids BOTH problems. It learns from real experience (no teleportation) and implicitly captures velocity effects (because the agent actually moves through the environment)."

### Slide 35: Policy Iteration
> "Quick mention: Policy Iteration alternates between evaluating a fixed policy and improving it."
> "Same limitations as VI. Converges faster in some cases."
> "Both are exact methods -- elegant but impractical for real problems."

---

## Part 4: Value-based Methods (slides 36-45)

### Slide 36: Outline (Value-based Methods highlighted)

### Slide 37: Q-Values
> "Instead of V(s) = value of state, let's track Q(s,a) = value of taking action a in state s."
> "Q*(s,a) = sum_{s'} P(s'|s,a)(R + gamma max_{a'} Q*(s',a'))"
> "Why Q instead of V? Because pi*(s) = argmax_a Q*(s,a) -- no need for the dynamics model to extract the policy!"

### Slide 38: Q-Value Iteration
> "Same Bellman update, but for Q. Still needs the model for the sum over s'."
> Show Q-values grid (each cell has 4 triangles for each action direction).

### Slide 39: Tabular Q-Learning
> **Key insight:** "What if we don't have P(s'|s,a)? Just REPLACE the expectation with a single sample!"
> - Take action a in state s, observe s' and reward r
> - Compute target = r + gamma * max_{a'} Q(s', a')
> - Update Q(s,a) toward the target with learning rate alpha
>
> "This is Q-learning: learn from interaction, no model needed."

### Slide 40: Q-Learning algorithm pseudocode
> Walk through: epsilon-greedy exploration, running average update.
> Notes: off-policy (can learn optimal policy while acting suboptimally), but need enough exploration.

### Slide 41: Q-learning on Crawler Bot

**[DEMO: Tabular Q-learning on 2D Crawler -- WORKS]**

> "Same 81-state crawler. But now we DON'T build a model. The agent just acts, observes, and learns."
> Run Demo 1 cell. Show training curve going up. Show Q-table heatmap.
> "Compare to VI: no teleportation needed. The agent learned by crawling around."
> Roll out the policy -- similar performance to VI.

### Slide 42: Can Tabular Methods Scale?
> Crawler 10^2, Hopper 10^10, Humanoid 10^100.
> "The Q-table would have more entries than atoms in the universe."

**[DEMO: Tabular Q-learning on 4D Crawler -- BREAKS]**

> "Let's add velocity to the state. Now 9^4 = 6561 states."
> Run Demo 2 cell. Show training curve -- flat, doesn't learn.
> "Most states never visited. The table is too sparse."

### Slide 43: Approximate Q-Learning
> "Replace the Q-table with a neural network: Q_theta(s, a)."
> "Instead of updating a table entry, update network weights to minimize (Q_theta(s,a) - target)^2."

### Slide 44: The Deadly Triad
> "Three things combine to cause instability: function approximation + bootstrap updates + off-policy learning."
> "Solutions: replay buffer (decorrelate samples), double Q-learning (reduce overestimation)."

### Slide 45: DQN on Atari
> "DeepMind 2013: same CNN plays 49 Atari games from pixels. Human-level on 29."

**[DEMO: DQN on 4D Crawler -- WORKS]**

> Run Demo 3. "Same 4D state that broke tabular Q. But the neural network generalizes across similar states."
> Show training curve -- learns!

**[DEMO: DQN with fine continuous actions -- BREAKS]**

> "But what about continuous actions? DQN needs argmax over all actions. If we discretize torques into 7x7=49 or 11x11=121 actions..."
> Run Demo 4. Show it degrades as actions increase.
> "DQN fundamentally needs discrete actions. For robotics, we need continuous torques. We need a different paradigm."

---

## Part 5: Policy-based Methods (slides 46-57)

### Slide 46: Outline (Policy-based Methods highlighted)

### Slide 47: Paradigm Shift
> "Value-based: learn Q -> extract policy via argmax. Limited to discrete actions."
> "Policy-based: directly learn a policy network pi_theta(a|s). The network outputs actions."
> "On-policy: uses data from the current policy (must re-collect after each update)."

### Slide 48: Parametrizing Stochastic Policy
> "Discrete actions: softmax over logits (categorical distribution)."
> "Continuous actions: network outputs mean mu and std sigma of a Gaussian. Sample action from N(mu, sigma)."
> "This is how we handle continuous action spaces!"

### Slide 49: Likelihood Ratio Policy Gradient
> "Our goal: max_theta U(theta) = sum_tau P(tau; theta) R(tau)."
> "tau is a trajectory (s_0, a_0, s_1, a_1, ...). P(tau; theta) is the probability of that trajectory under our policy."

### Slide 50: Policy Gradient Derivation
> Walk through the log-trick derivation. Focus on intuition, not mechanics:
> "We use the log-derivative trick: grad P = P * grad log P. This turns an intractable sum into an expectation we can estimate with samples."
> "Result: grad U ≈ (1/m) sum_i grad_theta log P(tau_i; theta) * R(tau_i)"

### Slide 51: Likelihood Ratio Gradient -- Intuition
> "What does this gradient do? It increases the probability of trajectories with HIGH reward and decreases the probability of trajectories with LOW reward."
> "It's like natural selection for trajectories: good runs get reinforced."

### Slide 52: Decompose Path into States and Actions
> "Key step: log P(tau) = sum of log P(s_{t+1}|s_t, a_t) + sum of log pi(a_t|s_t)."
> "Take gradient w.r.t. theta: the dynamics model terms vanish! Only the policy terms remain."
> "NO DYNAMICS MODEL REQUIRED. This is the magic of policy gradient."

### Slide 53: Likelihood Ratio Gradient Estimate
> "Final formula: grad U ≈ (1/m) sum_i [sum_t grad log pi(a_t|s_t)] * R(tau_i)"
> "Unbiased estimate of the true gradient. Just collect trajectories and compute."

### Slide 54: Variance Reduction
> "Problem: this estimate is unbiased but VERY noisy. Small sample of all possible trajectories."
> "Tends to over-prioritize trajectories with high absolute reward."
> Draw on board/show slide: sample trajectories scattered, gradient direction is noisy.

### Slide 55: Baseline Subtraction
> "Key idea: subtract a baseline b from the reward. grad U ≈ (1/m) sum grad log P(tau) * (R(tau) - b)."
> "Proof that it's still unbiased: E[grad log P * b] = b * grad sum P(tau) = b * grad(1) = 0."
> "Intuition: instead of 'make good trajectories more likely', it's 'make BETTER-THAN-AVERAGE trajectories more likely'."

### Slide 56: Baseline Choices
> - Constant baseline: b = average reward (simplest)
> - Time-dependent baseline: average future reward from time t
> - State-dependent baseline: b(s_t) = V^pi(s_t) -- the value function!
> "When b(s) = V^pi(s), we get Q^pi(s,a) - V^pi(s) = A^pi(s,a), the ADVANTAGE function."
> "This leads us to Actor-Critic methods..."

### Slide 57: Summary (2025 version)
> Exact: state exploration + need dynamic model
> Value-based: discrete-only + high bias
> Policy-based: high variance
> Arrow to Actor-Critic

---

## Part 6: REINFORCE (from 2025 lecture 2, slides 9-16)

### NEW outline slide: show we're now covering REINFORCE

### Slide L2-10: Vanilla Policy Gradient (REINFORCE) Algorithm
> Show Algorithm 1 pseudocode.
> "This is the simplest policy gradient algorithm. Collect trajectories, compute returns, update policy."
> "For each timestep: compute return R_t = sum of future discounted rewards. Estimate advantage A_t = R_t."
> "Update: theta = theta + alpha * grad_hat"

### Slide L2-11: VPG Training Curves
> Show learning rate sensitivity plot (alpha = 2^-12, 2^-13, 2^-14).
> "Very sensitive to learning rate. Too big = unstable. Too small = doesn't learn."

### Slide L2-12: VPG with Baseline (Actor-Critic)
> "Add a baseline: A_t = R_t - b(s_t). Re-fit baseline by minimizing ||b(s_t) - R_t||^2."
> "Replace the sum of future rewards with Q^pi(s_t, a_t) - b(s_t)."
> "Popular choice for b(s): V^pi(s_t)."

### Slide L2-13: Advantage Function
> "A(s,a) = Q(s,a) - V(s) = 'how much better is this action compared to average?'"
> "This is the key quantity we want to estimate."

### Slide L2-14: Estimating V^pi(s)
> Monte-Carlo estimation: regress V against empirical returns (high variance, unbiased)
> Bootstrap estimation: fitted V iteration using TD targets (lower variance, some bias)

### Slide L2-15: Training curve comparison
> REINFORCE with baseline converges faster and more stably than without.

### Slide L2-16: Actor-Critic Algorithm
> "Two networks: actor (policy) and critic (value function)."
> "Critic update: minimize ||b(s_t) - R_t||^2. Actor update: theta += alpha * grad_hat."
> **"Can get unstable if network is large!"** -- this is the cliffhanger for lecture 2.

### Updated Summary Slide
> Same three-column summary, but now REINFORCE and Actor-Critic are included.
> "Next lecture: How to make this stable and practical -- TRPO, GAE, PPO."
> "And: the secret sauces that really make RL work in practice."
