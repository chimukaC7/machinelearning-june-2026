# Thompson Sampling: Intuition and Study Notes

## Overview

**Thompson Sampling** is a randomized Bayesian algorithm for balancing exploration and exploitation in a multi-armed bandit problem.

At every round, it:

1. represents current uncertainty about each arm with a posterior distribution;
2. draws one possible value for every arm from its posterior;
3. selects the arm with the greatest sampled value;
4. observes the reward;
5. updates the selected arm's posterior.

The algorithm can be summarized as:

> Sample one plausible world from the current posterior beliefs, then choose the action that would be optimal in that sampled world.

This automatically favors arms that appear rewarding while continuing to explore arms whose values remain uncertain.

## Multi-armed bandit setting

Assume there are $D$ arms:

$$
\mathcal{A}=\{1,2,\ldots,D\}
$$

At round $t$:

1. the learner chooses arm $A_t$;
2. the selected arm produces reward $R_t$;
3. only that reward is observed;
4. the learner updates its beliefs;
5. the process repeats.

The objective is to maximize cumulative reward:

$$
\sum_{t=1}^{T}R_t
$$

For online advertising, each arm can represent an advertisement. In the simplest click model:

$$
R_t=
\begin{cases}
1,&\text{if the displayed ad is clicked}\\
0,&\text{otherwise}
\end{cases}
$$

The unknown expected reward of an ad is then its click-through probability.

## Reward distributions versus posterior distributions

This distinction is the most important part of the intuition.

### Reward distribution

Every arm has a process that generates observed rewards. For a Bernoulli advertising model:

$$
R_t\mid A_t=i,\theta_i\sim\operatorname{Bernoulli}(\theta_i)
$$

where $\theta_i$ is the arm's unknown click probability.

The Bernoulli distribution describes possible **observed rewards**: click or no click.

### Posterior distribution

Thompson Sampling maintains a posterior belief about the unknown parameter $\theta_i$:

$$
p(\theta_i\mid\mathcal{H}_t)
$$

where $\mathcal{H}_t$ represents all observations collected before round $t$.

This distribution describes uncertainty about **where the arm's true mean parameter might be**. It is not the same as the arm's reward-generating distribution.

In the lecture's diagram, the smooth curves should therefore be interpreted as posterior densities over the unknown expected returns.

## Bayesian ingredients

Bayesian inference combines:

- a **prior**, representing beliefs before observing the current data;
- a **likelihood**, describing how rewards are generated for a proposed parameter value;
- a **posterior**, representing updated beliefs after observing the data.

Bayes' rule gives:

$$
p(\theta_i\mid\text{data})
\propto
p(\text{data}\mid\theta_i)\,p(\theta_i)
$$

Thompson Sampling is not a “no prior knowledge” method. Even a deliberately uninformative-looking prior is still a modeling choice.

## Beta–Bernoulli Thompson Sampling

For binary rewards, the Beta distribution is a convenient prior for the Bernoulli success probability:

$$
\theta_i\sim\operatorname{Beta}(\alpha_i,\beta_i)
$$

Its support is $[0,1]$, matching the possible values of a probability.

### Common initial prior

A frequently used starting point is:

$$
\theta_i\sim\operatorname{Beta}(1,1)
$$

This density is uniform over $[0,1]`. It assigns equal density to every possible click probability before data are observed.

### Posterior update

Suppose arm $i$ has produced:

- $S_i$ successes or clicks;
- $F_i$ failures or non-clicks.

Starting from a $\operatorname{Beta}(\alpha_0,\beta_0)$ prior, the posterior is:

$$
\theta_i\mid\mathcal{H}_t
\sim
\operatorname{Beta}(\alpha_0+S_i,\beta_0+F_i)
$$

With the common $\operatorname{Beta}(1,1)$ prior:

$$
\theta_i\mid\mathcal{H}_t
\sim
\operatorname{Beta}(S_i+1,F_i+1)
$$

Its posterior mean is:

$$
\mathbb{E}[\theta_i\mid\mathcal{H}_t]
=
\frac{\alpha_i}{\alpha_i+\beta_i}
$$

As observations accumulate, the posterior generally becomes more concentrated, reflecting reduced uncertainty.

## Thompson Sampling algorithm

Initialize every arm with:

$$
\alpha_i=1,\qquad\beta_i=1
$$

At each round $t$:

1. Sample one value for every arm:

   $$
   \widetilde{\theta}_i
   \sim
   \operatorname{Beta}(\alpha_i,\beta_i)
   $$

2. Choose the arm with the greatest sample:

   $$
   A_t\in\arg\max_i\widetilde{\theta}_i
   $$

3. Observe the binary reward $R_t\in\{0,1\}$.
4. Update only the selected arm:

   $$
   \alpha_{A_t}\leftarrow\alpha_{A_t}+R_t
   $$

   $$
   \beta_{A_t}\leftarrow\beta_{A_t}+(1-R_t)
   $$

5. Repeat.

A click increases the selected arm's success parameter. A non-click increases its failure parameter. The posterior distributions of unselected arms remain unchanged for that round.

## Compact pseudocode

```text
input: number of rounds T, number of arms D

alpha[1..D] = 1
beta[1..D] = 1

for t = 1 to T:
    for each arm i:
        sample[i] = draw from Beta(alpha[i], beta[i])

    arm = argmax(sample)
    reward = observe_reward(arm)

    if reward == 1:
        alpha[arm] += 1
    else:
        beta[arm] += 1
```

Unlike the standard UCB formula, Thompson Sampling does not require a separate “select every arm once” rule when proper priors are used. The initial priors already define valid distributions from which the algorithm can sample.

## The sampled-world interpretation

Suppose the learner maintains one posterior for each of three arms.

For a new round, it draws:

$$
\widetilde{\theta}_1,\quad
\widetilde{\theta}_2,\quad
\widetilde{\theta}_3
$$

These samples form one plausible configuration of the unknown arm means:

$$
(\theta_1,\theta_2,\theta_3)
\approx
(\widetilde{\theta}_1,\widetilde{\theta}_2,\widetilde{\theta}_3)
$$

Within that hypothetical world, the decision is easy: choose the arm with the greatest sampled mean.

After the real reward is observed, the learner updates its posterior and samples a new hypothetical world on the next round.

The samples are not predictions of the next monetary payouts. They are samples of plausible **mean parameters** under the posterior model.

## Probability matching

Thompson Sampling is also called **posterior probability matching**.

Under the model, the probability that Thompson Sampling selects arm $i$ equals the posterior probability that arm $i$ is optimal:

$$
P(A_t=i\mid\mathcal{H}_t)
=
P\left(
\theta_i=\max_j\theta_j
\mid\mathcal{H}_t
\right)
$$

Ignoring zero-probability ties in a continuous model, if the current posterior says an arm has a 70% probability of being best, Thompson Sampling selects it approximately 70% of the time across repeated posterior draws.

This property creates the exploration–exploitation balance:

- an arm that is probably optimal is chosen often;
- an uncertain arm still has a chance to produce the largest posterior draw;
- an arm that is almost certainly inferior is selected rarely.

## Numerical intuition

Suppose three ads currently have these posteriors:

| Arm | Posterior | Posterior mean | Interpretation |
|---|---|---:|---|
| A | $\operatorname{Beta}(9,13)$ | 0.409 | Moderate evidence, moderate estimate |
| B | $\operatorname{Beta}(4,3)$ | 0.571 | Little evidence, high uncertainty |
| C | $\operatorname{Beta}(31,21)$ | 0.596 | More evidence, relatively concentrated |

On one round, the samples might be:

$$
\widetilde{\theta}_A=0.43,\quad
\widetilde{\theta}_B=0.78,\quad
\widetilde{\theta}_C=0.60
$$

Arm B is selected because its sample is largest. Its high uncertainty allowed it to generate an optimistic draw.

On another round, the samples might be:

$$
\widetilde{\theta}_A=0.38,\quad
\widetilde{\theta}_B=0.49,\quad
\widetilde{\theta}_C=0.63
$$

Arm C is then selected.

This randomization is purposeful: it explores in proportion to posterior uncertainty rather than exploring uniformly.

## What happens after an observation?

Suppose arm B has posterior:

$$
\theta_B\sim\operatorname{Beta}(4,3)
$$

If the displayed ad is clicked:

$$
\theta_B\mid R_t=1
\sim
\operatorname{Beta}(5,3)
$$

If the displayed ad is not clicked:

$$
\theta_B\mid R_t=0
\sim
\operatorname{Beta}(4,4)
$$

The update can change the location and shape of the posterior. One observation does not necessarily make its center move toward the unknown truth on that particular round, but repeated observations refine the belief under suitable model assumptions.

## Why uncertain arms are explored

An arm with few observations has a broad posterior. A broad posterior may occasionally generate a very high sample, causing that arm to be selected.

If it performs well, its posterior shifts toward larger values and it is selected more frequently. If it performs poorly, its posterior moves toward smaller values and its chance of winning future posterior draws decreases.

No separate exploration bonus is required. Exploration emerges directly from posterior uncertainty.

## Thompson Sampling compared with UCB

| Aspect | Thompson Sampling | Upper Confidence Bound |
|---|---|---|
| Decision principle | Sample from each posterior | Calculate an optimistic index |
| Type | Randomized | Usually deterministic given the history and tie rule |
| Uncertainty representation | Full posterior model | Confidence-style bonus |
| Selected arm | Largest posterior sample | Largest upper-bound score |
| Exploration | Probability matching | Optimism under uncertainty |
| Prior required | Yes, explicitly or implicitly | Not usually Bayesian |

Both methods increasingly favor strong arms while preserving exploration. Neither universally dominates the other; performance depends on assumptions, implementation, horizon, reward structure, and environment.

## Thompson Sampling is more general than Beta distributions

The Beta–Bernoulli model applies specifically to binary rewards.

Other reward types require other probabilistic models:

- Gaussian likelihoods for approximately continuous normal rewards;
- count models for event counts;
- Bayesian linear models for contextual bandits;
- hierarchical models for related arms;
- approximate posterior methods when exact Bayesian updates are unavailable.

For monetary slot-machine rewards, using a Beta posterior is generally inappropriate unless the reward has first been modeled as a binary event. The likelihood and prior must match the reward-generating assumptions.

## Regret intuition

Let:

$$
\Delta_i=\mu^*-\mu_i
$$

be the expected-reward gap for suboptimal arm $i$.

Under standard stationary stochastic-bandit assumptions, Thompson Sampling has strong theoretical regret guarantees. For common Bernoulli settings, problem-dependent expected regret grows logarithmically with the horizon under suitable conditions.

The practical interpretation is:

- optimal arms receive an increasing share of selections;
- clearly suboptimal arms are selected less often;
- uncertain arms continue to receive occasional trials;
- the fraction of suboptimal selections tends to decrease over time.

The exact guarantee depends on the model, prior, algorithm variant, and assumptions.

## Advertising versus fixed A/B testing

Thompson Sampling can adapt traffic during a campaign, whereas a conventional A/B/n test usually retains prespecified allocation probabilities.

This does not mean that an organization uses Thompson Sampling merely because it lacks time or money for an A/B test. The methods prioritize different goals:

- bandits typically prioritize cumulative reward;
- fixed experiments often prioritize unbiased estimation and hypothesis testing.

Adaptive allocation can improve user outcomes during an experiment, but it also requires analysis methods that account for adaptive data collection.

## Assumptions and practical limitations

The elementary Beta–Bernoulli implementation assumes:

- a finite set of arms;
- binary rewards;
- stationary click probabilities;
- prompt and correctly recorded feedback;
- independent rewards conditional on the selected arm;
- no important user context;
- a correctly specified Bayesian model.

Real systems may involve:

- changing user preferences;
- seasonality and creative fatigue;
- delayed conversions;
- heterogeneous users;
- budget and inventory restrictions;
- several business objectives;
- fairness and brand-safety constraints.

Possible extensions include contextual Thompson Sampling, discounted or sliding-window variants, delayed-feedback models, and constrained Bayesian bandits.

## Common misconceptions

### “The posterior curve estimates the reward distribution”

In the Beta–Bernoulli model, it represents uncertainty about the Bernoulli success probability, not the distribution of observed click/no-click rewards.

### “Thompson Sampling begins with no prior”

It always uses a prior model. $\operatorname{Beta}(1,1)$ is a uniform prior, not the absence of a prior.

### “Every arm must be manually tested several times first”

Not necessarily. Proper priors allow sampling from the first round, although deliberate initialization may still be used in some systems.

### “The sampled value is the reward we expect to receive next”

It is a sampled plausible value of the arm's mean parameter, not a sampled real-world payout.

### “Only the best arm's posterior becomes narrow”

Only selected arms are updated. Frequently selected arms generally become more concentrated; rarely selected arms can remain broad, which preserves some probability of future exploration.

### “Thompson Sampling eventually becomes completely deterministic”

Its posterior draws remain random. As evidence accumulates, the optimal arm may be selected with probability approaching one under appropriate assumptions, but posterior sampling itself remains probabilistic.

## Historical note and further reading

Thompson Sampling dates to William R. Thompson's 1933 paper:

> W. R. Thompson, **“On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples,”** *Biometrika*, 25(3/4), 285–294.

A modern detailed reference is:

> Daniel Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, and Zheng Wen, **“A Tutorial on Thompson Sampling,”** *Foundations and Trends in Machine Learning*, 11(1), 1–96, 2018.

## Quick review

1. **What is sampled at each round?**

   One plausible mean parameter from every arm's posterior distribution.

2. **Which arm is selected?**

   The arm with the largest sampled parameter.

3. **What does a Beta posterior represent in the advertising example?**

   Uncertainty about an ad's unknown click probability.

4. **How is a success recorded?**

   Increment the selected arm's $\alpha$ parameter.

5. **How is a failure recorded?**

   Increment the selected arm's $\beta$ parameter.

6. **Why are uncertain arms explored?**

   Broad posteriors occasionally produce large optimistic samples.

7. **What is probability matching?**

   Selecting an arm with probability equal to its posterior probability of being optimal.

8. **Why is Beta–Bernoulli unsuitable for arbitrary monetary rewards?**

   It models binary outcomes; other reward types require an appropriate likelihood and prior.

9. **How does Thompson Sampling differ from UCB?**

   Thompson Sampling draws from posterior beliefs, whereas UCB chooses the largest deterministic optimistic index.

## Key takeaway

Thompson Sampling repeatedly:

1. samples one plausible value for each arm;
2. acts optimally for that sampled configuration;
3. observes the real reward;
4. updates the corresponding posterior.

Its random choices are not arbitrary. They reflect the current probability that each arm is optimal, naturally combining exploitation of promising arms with exploration of uncertain ones.
