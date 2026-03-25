# Lectuer 1


Crawler 2d demo:
- The current Mujoco robot looks really ugly. That was the original design. 
- 

- [] [the real crawling robot](https://futurismo.biz/archives/6596/) for Q learning
- [ ] Some code experiments to show the variance without baseline subtraction
- [ ] Why baseline works for reducing the variance

# Theory of Policy Gradient:
- [ ] Sutton Barton book Policy grad chapter, detailed derivation
- [ ] Daniel Seita's note on [policy grad](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/)
- [ ] Daniel Seita's [Note on GAE](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/)

# SAC
- [] [RL guy's note on train loco with SAC](https://araffin.github.io/post/sac-massive-sim/)
	- PPO's gaussian distr only uses < 3% of the action space
	- PPO clip actions to fit the boundary, but many implementations have overly large clip values
	- SAC by defaults squashed gaussian leads to almost uniform on the entire action space
	- SAC/TQC and derivatives are tuned for sample efficiency, not fast wall clock time. In the case of massively parallel simulation, what matters is how long it takes to train, not how many samples are used. 
	- policy generates joint targets -> PD controller. policy Intentionally generates outside bound to trick PD to produce whatever torque it wants, by hijacking the proportional error term (target -current), regardless of the actual PD gain terms.

