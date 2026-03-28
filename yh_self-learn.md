# Self-Learn Notes

## Bootstrapping in RL

Bootstrapping means updating an estimate using another estimate, instead of waiting for the full observed return.

Example:

- Value Iteration updates `V(s)` using old estimates `V(s')`
- Q-learning updates `Q(s,a)` using estimated next-state action values `max_a' Q(s', a')`

Mathematically:

Value Iteration:

\[
V_{k+1}(s)=\max_a \sum_{s'} P(s'|s,a)\left[r(s,a,s')+\gamma V_k(s')\right]
\]

Here `V_{k+1}(s)` is updated using the previous estimate `V_k(s')`.

Q-learning:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha\Big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\Big)
\]

Here the target contains `Q(s',a')`, which is again an estimate rather than a fully observed return.

So "bootstrapping through the model" means:

- use the transition model `P(s'|s,a)` to decide which successor states matter
- use current value estimates of those successor states to update the current state

This is different from Monte Carlo methods, which do **not** bootstrap. They wait for the full trajectory return.

Monte Carlo policy evaluation:

\[
V(s_t) \leftarrow V(s_t) + \alpha\big(G_t - V(s_t)\big)
\]

where

\[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
\]

The key difference is that `G_t` is a sampled full return, not a target that contains another value estimate.

## Do all DP methods bootstrap?

Yes, dynamic programming methods like value iteration and policy evaluation do bootstrap.

But not all RL algorithms bootstrap.

Algorithms that bootstrap:

- Value Iteration
- Policy Iteration / DP policy evaluation
- TD(0)
- SARSA
- Q-learning
- DQN
- actor-critic critics

Algorithms that do not bootstrap:

- Monte Carlo policy evaluation
- Monte Carlo control
- REINFORCE / vanilla policy gradient with full returns

Another common bootstrap example from TD learning:

\[
V(s_t) \leftarrow V(s_t) + \alpha\big(r_t + \gamma V(s_{t+1}) - V(s_t)\big)
\]

This sits between DP and Monte Carlo:

- it does not use a full transition model like DP
- but it still bootstraps because the target contains `V(s_{t+1})`

## Why can Q-learning beat Value Iteration in the crawler demo?

Important refinement from the demos:

- It is **not only** that Demo 0A misses velocity and is therefore non-Markov.
- Demo 0B adds velocity and still does not help much.

So the deeper problem is:

- Value Iteration needs an explicit transition model over a discretized state space.
- Even with a richer state, that discretized model is still a crude approximation of the real continuous crawler dynamics.
- Contact-rich dynamics are sensitive, and repeated Bellman sweeps can propagate model errors everywhere.

Q-learning still bootstraps, but it does **not** need an explicit global transition model.

Instead, it updates from real sampled transitions in the actual simulator:

- it only learns from states the policy actually visits
- it avoids solving a bad tabular model exactly
- it can learn a useful reactive policy for the experienced state distribution

Short takeaway:

> Value Iteration solves the abstract tabular model exactly. Q-learning learns approximately from real trajectories. When the tabular model is poor, approximate learning on the real system can beat exact planning on the wrong model.

## Why can stochastic aggregated Value Iteration still fail?

A stochastic aggregated model is more sensible than a single deterministic representative transition, but it still does not remove the core problem.

What that model is trying to do is:

- take one coarse discrete bin `s`
- acknowledge that many different continuous microstates live inside that bin
- estimate a distribution over next bins `P(s' | s, a)` instead of one fixed next state

That helps a bit, but VI can still fail for a few reasons.

First, the aggregation itself is still too lossy.  
If one bin contains many physically different microstates, then the true outcome after action `a` may depend strongly on which microstate you were actually in. A single aggregated distribution over next bins washes those differences together. The model becomes “correct on average” in a weak sense, but not predictive enough for control.

Second, the hidden-state mixture is policy-dependent.  
The distribution of microstates inside a bin is not fixed once and for all. It depends on how the agent arrived there, which actions it has been taking, momentum, contact phase, etc. So the aggregated transition model
\[
P(s'|s,a)
\]
is often not really stationary at the coarse level. The same coarse bin under a different policy may contain a different mixture of underlying continuous states. Value iteration assumes one fixed MDP model, but the aggregated process may not actually behave like one.

Third, contact dynamics are highly nonlinear.  
For the crawler, tiny differences in velocity, body orientation, or contact configuration can cause very different next outcomes. Even a stochastic model over bins may be too coarse to preserve the action distinctions that matter. If two actions have meaningfully different effects only in certain microstates, aggregation can blur that away.

Fourth, Bellman planning amplifies model errors.  
Even if the one-step stochastic model is only a little wrong, value iteration repeatedly bootstraps through it. Those local errors get propagated many steps into the future. So a mildly inaccurate aggregated model can still produce a quite poor policy after many sweeps.

Fifth, reward aggregation can also become misleading.  
Even if the transition probabilities are averaged reasonably, the expected reward assigned to a coarse `(s,a)` can still hide important structure inside the bin. For locomotion, reward often depends on subtle short-term motion and contact effects. Averaging that at the bin level can flatten useful distinctions.

So the short version is:

- deterministic aggregation fails because one representative transition is too crude
- stochastic aggregation is better, but still assumes each coarse bin behaves like a proper Markov state with one fixed transition law
- in this crawler, that assumption is still too wrong, because each bin mixes together very different continuous situations

That is why VI can still struggle even with a stochastic aggregated model, while Q-learning can do better:

- Q-learning does not need one globally correct coarse transition law
- it only updates from the actual sampled transitions seen under the current rollout distribution

A concise summary:

“Making the tabular model stochastic helps, but it still assumes that each coarse bin has one well-defined transition distribution. In the crawler, each bin hides many different continuous microstates, and their mixture changes with the trajectory and policy. So even a stochastic aggregated model is still too crude, and Value Iteration ends up planning accurately on an inaccurate abstract MDP.”

## Correction: why the "policy-dependent mixture" wording needs care

If the coarse state were truly Markov, then the distribution of what happens next should depend only on the current coarse state `s` and action `a`, not on how you got there. In that case, talking about a history-dependent mixture inside the bin would be inappropriate.

So the statement

> “the distribution of microstates inside a bin depends on how the agent arrived there”

is really another way of saying:

> the coarse bin is not actually a Markov state.

That is the key point.

A better way to phrase it is:

- In the true continuous simulator, the full physical state is Markov.
- After we aggregate many continuous states into one coarse bin, that coarse representation may no longer be Markov.
- If it is not Markov, then two visits to the same bin can correspond to different underlying continuous states with different future behavior.
- In that case, there is no single correct coarse transition law `P(s' | s, a)` that works for all visits to that bin.

Under the Markov assumption, history should not matter. If history seems to matter at the coarse level, that is evidence that the coarse aggregation broke the Markov property.

That is why a stochastic aggregated model can still fail:

- it tries to force the aggregated bins into an MDP description
- but the aggregated process may actually be a POMDP or non-Markov abstraction
- so the estimated `P(s'|s,a)` is only an average over incompatible situations

Then value iteration solves that averaged coarse model exactly, but that exact solution need not control the real system well.

So the corrected explanation is:

“An MDP does not care about the past if the state is truly Markov. The problem here is that after coarse discretization or aggregation, the bins may no longer be Markov states. Then different underlying continuous states can collapse into the same bin, and there may be no single transition distribution `P(s'|s,a)` that captures them well. A stochastic aggregated model helps, but it still assumes the coarse bins define a valid MDP, which may be false.”

This wording is cleaner and more correct.
