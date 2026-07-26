# Upper Confidence Bound vs. Thompson Sampling

## Overview

Upper Confidence Bound (UCB) and Thompson Sampling are two influential approaches to the stochastic multi-armed bandit problem. Both attempt to maximize cumulative reward while learning which arm has the greatest expected payoff.

They solve the exploration–exploitation problem differently:

- **UCB** uses optimism: choose the arm with the greatest plausible upper value.
- **Thompson Sampling** uses posterior probability matching: sample one plausible value for every arm and choose the largest sample.

Neither method is universally superior. Their performance depends on the reward model, prior information, stationarity, feedback delays, implementation, and business objective.

## Shared problem

Assume there are $D$ arms. At round $t$:

1. select an arm $A_t$;
2. observe its reward $R_t$;
3. update the algorithm;
4. repeat.

The goal is to maximize:

$$
\sum_{t=1}^{T}R_t
$$

For advertising with binary click rewards:

$$
R_t\in\{0,1\}
$$

and the unknown mean reward of arm $i$ is its click-through probability:

$$
\mu_i=P(R_t=1\mid A_t=i)
$$

## UCB decision rule

A common UCB1-style score is:

$$
U_i(t)
=
\widehat{\mu}_i(t-1)
+
\sqrt{\frac{2\ln t}{N_i(t-1)}}
$$

where:

- $\widehat{\mu}_i$ is the observed mean reward;
- $N_i$ is the number of observations for arm $i$;
- the square-root term is an exploration bonus.

The selected arm is:

$$
A_t\in\arg\max_i U_i(t)
$$

UCB therefore chooses the arm with the highest optimistic index.

## Thompson Sampling decision rule

In the Beta–Bernoulli version:

$$
\theta_i\mid\mathcal{H}_t
\sim
\operatorname{Beta}(\alpha_i,\beta_i)
$$

At round $t$, draw:

$$
\widetilde{\theta}_i
\sim
\operatorname{Beta}(\alpha_i,\beta_i)
$$

for every arm, then select:

$$
A_t\in\arg\max_i\widetilde{\theta}_i
$$

After observing the binary reward:

$$
\alpha_{A_t}\leftarrow\alpha_{A_t}+R_t
$$

$$
\beta_{A_t}\leftarrow\beta_{A_t}+(1-R_t)
$$

Thompson Sampling therefore chooses the best arm in one randomly sampled, posterior-plausible world.

## Side-by-side comparison

| Dimension | Upper Confidence Bound | Thompson Sampling |
|---|---|---|
| Main principle | Optimism under uncertainty | Posterior probability matching |
| Action score | Estimate plus exploration bonus | Random draw from posterior |
| Randomness | Basic rule is deterministic given history and tie-breaking | Randomized by posterior sampling |
| Uncertainty representation | Confidence-style index | Full posterior distribution |
| Prior | Usually no explicit Bayesian prior | Explicit prior required |
| Initialization | Commonly sample each arm once or use infinite scores | Proper priors permit sampling immediately |
| Exploration | Arms with large uncertainty bonuses | Arms with meaningful posterior probability of being best |
| Bernoulli update | Counts and empirical means | Beta posterior parameters |
| Reproducibility | Same history and tie rule give the same choice | Same history can produce different choices |
| Model dependence | Requires a valid exploration bound | Requires a likelihood and prior |
| Basic computation | Means, counts, logarithms, square roots | Random sampling from each posterior |
| Standard theory | Strong frequentist regret guarantees | Strong Bayesian and frequentist regret results |

## Deterministic versus randomized behavior

### Basic UCB

Given:

- the same observation history;
- the same UCB formula;
- the same tie-breaking rule;

basic UCB selects the same arm. It is deterministic conditional on its history.

The rewards remain random because the environment is stochastic. Random tie-breaking or a randomized UCB variant can also introduce algorithmic randomness, so it is too broad to claim that every member of every UCB family is deterministic.

### Thompson Sampling

Thompson Sampling generates new posterior draws at every round. Given the same history, it can select different arms on repeated executions.

This randomness is deliberate rather than noise added without purpose. The probability of selecting an arm reflects its posterior probability of being optimal.

### Implication for evaluation

One execution of Thompson Sampling is not enough to characterize performance. Simulations should use many independent runs and report quantities such as:

- mean cumulative reward;
- mean regret;
- variability or uncertainty across runs;
- frequency with which each arm is selected.

UCB should also be evaluated across many independently generated reward sequences, even if its action rule itself is deterministic.

## Delayed feedback: an important correction

The standard sequential versions of **both** algorithms are usually described as receiving feedback before the next update. Neither method automatically solves arbitrary delayed-feedback or batched-feedback problems merely because Thompson Sampling is randomized.

### What happens if Thompson Sampling is not updated?

It can continue drawing new samples from the unchanged posterior and therefore continue making decisions. However:

- it is not learning from the pending outcomes;
- it may repeatedly allocate traffic based on stale uncertainty;
- naïve large batches can over-select an arm before its outcomes arrive;
- standard immediate-feedback guarantees do not automatically carry over.

### What happens if UCB is not updated?

A naïve UCB implementation may repeatedly select the same apparently promising arm while its feedback is pending, or may calculate invalid statistics if selected actions and observed rewards are not tracked separately.

This can be more visibly problematic for the deterministic rule, but it does not mean delayed UCB is impossible.

### Correct conclusion

Both approaches have delayed and batched variants. A production design should explicitly track:

- decisions made;
- rewards observed;
- rewards still pending;
- delay distributions;
- batch boundaries;
- how uncertainty is adjusted for in-flight observations.

Research provides specialized delayed-feedback UCB algorithms and batched or delayed Thompson Sampling methods. The right comparison depends on the delay model and the exact variant, not simply on “deterministic versus probabilistic.”

## Batch updating

Batching can reduce infrastructure overhead because the policy does not need to be recomputed after every event. However, batch size creates a trade-off:

- smaller batches incorporate evidence quickly;
- larger batches reduce update frequency and system cost;
- very large batches may waste traffic on arms that would already look weak if pending feedback were available.

Simply processing every 5,000 observations with an otherwise sequential algorithm is not guaranteed to preserve its usual behavior. Batch schedules and pending-feedback treatment should be designed deliberately.

## Computational considerations

### Basic UCB

For each arm, basic UCB requires:

- a reward count;
- a selection count;
- a logarithm and square-root calculation;
- an `argmax`.

It is generally inexpensive and easy to reproduce.

### Beta–Bernoulli Thompson Sampling

For each arm, the elementary form requires:

- success and failure counts;
- one Beta random draw;
- an `argmax`.

This is also inexpensive for a modest number of arms.

### Complex models

Thompson Sampling can become computationally demanding when:

- the posterior has no closed form;
- approximate inference or Markov chain Monte Carlo is needed;
- arms share a large hierarchical model;
- the context has many dimensions.

UCB can also become difficult when deriving or computing a valid confidence bound for a complex model.

Computational cost is therefore model- and implementation-dependent. Thompson Sampling is not inherently cheaper merely because it permits random resampling between updates.

## Prior knowledge and model specification

### Thompson Sampling

Thompson Sampling can naturally incorporate prior information. This is useful when reliable historical evidence or domain knowledge exists.

It also introduces risks:

- a poor prior can slow learning;
- an overconfident prior can suppress necessary exploration;
- a misspecified likelihood can produce misleading posterior uncertainty.

### UCB

Basic UCB does not require an explicit Bayesian prior and can be attractive when a simple bounded-reward model is appropriate.

It still relies on assumptions:

- reward boundedness or tail behavior;
- stationarity;
- the validity of the selected concentration bound;
- correct handling of initialization and confidence parameters.

Neither algorithm is assumption-free.

## Theoretical performance

Under common stationary stochastic-bandit assumptions, both UCB1 and suitable Thompson Sampling variants achieve logarithmic problem-dependent expected regret.

If:

$$
\Delta_i=\mu^*-\mu_i
$$

is the gap for a suboptimal arm, both methods can concentrate selections on optimal arms while selecting suboptimal arms only logarithmically often in expectation under their respective assumptions.

Their exact regret constants and guarantees depend on:

- reward family;
- prior;
- horizon;
- arm gaps;
- algorithm variant;
- delayed or immediate feedback;
- Bayesian versus frequentist evaluation.

It is therefore inaccurate to say that one algorithm always has a stronger theoretical foundation.

## Empirical performance

Thompson Sampling often performs very well in simulations and applications, and many studies report favorable empirical results. That does not establish universal dominance over UCB.

The outcome can change with:

- the number of arms;
- reward probabilities;
- closeness of the best arms;
- prior choice;
- horizon;
- nonstationarity;
- context;
- delayed observations;
- the chosen UCB and Thompson Sampling variants.

The correct empirical comparison should:

1. use the same environment and reward objective;
2. evaluate many independent simulations or experiments;
3. report variability, not only one trajectory;
4. compare cumulative reward and regret;
5. include sensitivity checks for priors and exploration parameters;
6. reflect operational delays and constraints.

The practical exercise in this course demonstrates one dataset and one setup. Its winner is evidence for that experiment, not a universal ranking.

## When UCB may be attractive

UCB can be a good choice when:

- rewards fit a well-understood bounded stochastic model;
- a transparent index is desirable;
- reproducible decisions are important;
- a prior is difficult to justify;
- simple implementation and debugging are priorities;
- the confidence-bound theory matches the operational setting.

## When Thompson Sampling may be attractive

Thompson Sampling can be a good choice when:

- a credible Bayesian model is available;
- prior information is useful;
- probability matching is operationally appealing;
- randomization is desirable;
- posterior sampling is computationally manageable;
- the method must extend naturally to a richer Bayesian model.

## Situations requiring more than basic UCB or Thompson Sampling

Neither elementary algorithm is sufficient when the problem includes:

- changing reward probabilities;
- strong user or item context;
- delayed or censored conversions;
- several simultaneous actions;
- inventory, budget, or fairness constraints;
- strategic or adversarial behavior;
- long-term consequences beyond immediate reward.

Possible alternatives include:

- contextual UCB or linear UCB;
- contextual Thompson Sampling;
- sliding-window or discounted bandits;
- delayed-feedback bandits;
- combinatorial and constrained bandits;
- adversarial algorithms such as EXP3.

## Production concerns

Before deploying either policy, define:

- the reward being optimized;
- the feedback window;
- how pending outcomes are represented;
- how new and retired arms are handled;
- safety and minimum-exposure constraints;
- monitoring for drift;
- offline and online evaluation procedures;
- rollback criteria;
- statistical reporting requirements.

Optimizing clicks may reduce profit or long-term customer value. The selected reward should reflect the actual business goal.

## Common misconceptions

### “Every UCB algorithm is deterministic”

Basic UCB with deterministic tie-breaking is deterministic conditional on history. Randomized ties and randomized UCB variants are possible.

### “Thompson Sampling works normally without feedback”

It can continue sampling from a stale posterior, but it does not learn until outcomes arrive. Delayed and batched settings require explicit analysis and often specialized design.

### “UCB cannot support delayed feedback”

Delayed-feedback UCB methods exist. Naïve UCB may perform poorly under delay, but the limitation is not absolute.

### “Batch Thompson Sampling is automatically correct”

Batching changes the information pattern. A suitable batch policy and pending-feedback treatment are needed to retain good performance.

### “Thompson Sampling always beats UCB”

No universal ranking exists. Results depend on the environment, assumptions, tuning, and variants.

### “Only Thompson Sampling has uncertainty”

Both algorithms represent uncertainty. UCB uses an explicit optimistic bonus; Thompson Sampling uses a posterior distribution.

## Quick review

1. **What is the central difference?**

   UCB selects the largest optimistic index; Thompson Sampling selects the largest posterior draw.

2. **Why is basic UCB called deterministic?**

   Given the same history, formula, and tie rule, it makes the same selection.

3. **Why is Thompson Sampling randomized?**

   It generates fresh samples from the posterior distributions at every round.

4. **Can Thompson Sampling act before pending rewards arrive?**

   Yes, but it acts using stale beliefs and does not learn from those pending outcomes.

5. **Can UCB be adapted to delays?**

   Yes. Delayed-feedback UCB variants explicitly account for observed and pending information.

6. **Does Thompson Sampling always have better empirical performance?**

   No. It often performs strongly, but comparisons are problem- and variant-dependent.

7. **What extra modeling choice does Thompson Sampling require?**

   A prior and likelihood, or another method of producing posterior samples.

8. **What should a fair simulation report?**

   Results over many independent runs, including cumulative reward, regret, and variability.

9. **Which algorithm should be selected?**

   The one whose assumptions, model, operational behavior, and validation results best fit the actual problem.

## Key takeaway

UCB and Thompson Sampling are two principled ways to explore while exploiting:

$$
\text{UCB: estimate + optimism bonus}
$$

$$
\text{Thompson Sampling: sample a plausible model and act optimally}
$$

Basic UCB is transparent and usually deterministic given its history. Thompson Sampling is randomized and naturally Bayesian. Both can achieve strong regret performance, both require assumptions, and both need deliberate extensions for delayed, batched, contextual, or nonstationary environments.
