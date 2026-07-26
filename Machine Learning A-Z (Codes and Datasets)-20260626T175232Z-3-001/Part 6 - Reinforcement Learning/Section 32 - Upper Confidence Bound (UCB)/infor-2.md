# Upper Confidence Bound: Intuition and Study Notes

## Overview

The **Upper Confidence Bound** family of algorithms addresses the exploration–exploitation trade-off in a multi-armed bandit problem.

At each round, the learner must choose one of $D$ arms—for example, one advertisement from several candidates. Each selected arm returns a reward. In a click-optimization problem:

$$
r_t=
\begin{cases}
1,&\text{if the displayed advertisement is clicked}\\
0,&\text{otherwise}
\end{cases}
$$

The objective is to maximize cumulative reward:

$$
\sum_{t=1}^{T}r_t
$$

The difficulty is that the expected reward of each arm is unknown. The learner must gather evidence about the arms while already making reward-bearing decisions.

## Core idea: optimism under uncertainty

UCB assigns every arm an **optimistic score** containing two components:

$$
\text{UCB score}
=
\text{estimated mean reward}
+
\text{uncertainty bonus}
$$

An arm receives a high score when:

- it has produced a high average reward;
- it has not been selected often, so its value remains uncertain;
- or both.

At every round, the algorithm selects the arm with the highest score.

This implements the principle of **optimism under uncertainty**:

> When an arm might plausibly be better than the current favorite, give it another opportunity to demonstrate its value.

## UCB1 formula

Assume rewards are bounded, commonly in $[0,1]$. Before round $t$, define:

- $N_i(t-1)$: the number of times arm $i$ has been selected;
- $S_i(t-1)$: the total reward obtained from arm $i$;
- $\widehat{\mu}_i(t-1)$: the empirical mean reward of arm $i$.

The empirical mean is:

$$
\widehat{\mu}_i(t-1)
=
\frac{S_i(t-1)}{N_i(t-1)}
$$

A common UCB1 index is:

$$
\operatorname{UCB}_i(t)
=
\widehat{\mu}_i(t-1)
+
\sqrt{\frac{2\ln t}{N_i(t-1)}}
$$

The course implementation commonly uses the closely related exploration term:

$$
\Delta_i(t)
=
\sqrt{\frac{3\ln t}{2N_i(t-1)}}
$$

and therefore:

$$
\operatorname{UCB}_i(t)
=
\widehat{\mu}_i(t-1)+\Delta_i(t)
$$

The constants vary across UCB variants and derivations, but the structure is the same:

$$
\text{estimate}+\sqrt{\frac{\log(\text{time})}{\text{number of selections}}}
$$

## Meaning of the two terms

### Empirical mean: exploitation

The empirical mean summarizes the rewards observed from an arm:

$$
\widehat{\mu}_i=\frac{\text{reward sum for arm }i}{\text{selections of arm }i}
$$

For a binary advertising reward, this is the observed click-through rate.

An arm with 20 clicks from 100 displays has:

$$
\widehat{\mu}_i=\frac{20}{100}=0.20
$$

All else equal, an arm with a larger empirical mean is more attractive.

### Uncertainty bonus: exploration

The bonus is large when $N_i$ is small:

$$
\sqrt{\frac{\log t}{N_i}}
$$

This gives under-sampled arms a chance to be selected even when their current observed means are not the largest.

As an arm is selected repeatedly, $N_i$ increases and its uncertainty bonus generally shrinks. The algorithm then relies increasingly on its empirical mean.

## Important dynamic behavior

The visual explanation often shows a confidence band shrinking around each arm's observed mean. This is useful intuition, but several details matter.

### 1. Unplayed arms need special handling

The formula is undefined when $N_i=0$. A standard implementation therefore does one of the following:

- select every arm once during initialization;
- assign an infinite UCB score to every unselected arm.

The arms do not begin with a meaningful common estimated reward and a conventional interval. They must first be sampled.

### 2. The observed mean may move in either direction

For a Bernoulli reward:

- a click pushes the empirical mean upward unless it is already 1;
- a non-click pushes it downward unless it is already 0.

The mean does not move toward the true value on every individual round. It fluctuates randomly and converges under suitable stationary assumptions as the sample count grows.

### 3. An unselected arm's bonus can grow

For an arm that is not chosen, $N_i$ stays fixed while $\ln t$ increases. Its UCB score can therefore rise slowly relative to the scores of frequently selected arms.

This is an essential part of UCB's exploration behavior. Neglected arms can become competitive again and receive another trial.

### 4. Exploration does not stop permanently

In the standard stationary stochastic setting, UCB increasingly favors an optimal arm, but it normally continues selecting suboptimal arms occasionally. Under standard assumptions, UCB1 selects a suboptimal arm only logarithmically often in expectation rather than stopping after a fixed discovery point.

## Step-by-step UCB1 algorithm

Suppose there are $D$ advertisements and $T$ rounds.

### Initialization

Set:

$$
N_i=0,\qquad S_i=0
$$

for every arm $i$.

### At round $t$

1. If an arm has never been selected, select an untried arm.
2. Otherwise, calculate each empirical mean:

   $$
   \widehat{\mu}_i=\frac{S_i}{N_i}
   $$

3. Calculate each exploration bonus:

   $$
   \Delta_i(t)=\sqrt{\frac{3\ln t}{2N_i}}
   $$

4. Calculate each upper-bound score:

   $$
   U_i(t)=\widehat{\mu}_i+\Delta_i(t)
   $$

5. Select the arm with the greatest score:

   $$
   A_t\in\arg\max_i U_i(t)
   $$

6. Observe the reward $r_t$.
7. Update the selected arm:

   $$
   N_{A_t}\leftarrow N_{A_t}+1
   $$

   $$
   S_{A_t}\leftarrow S_{A_t}+r_t
   $$

8. Repeat for the next round.

If multiple arms have the same maximum score, use a defined tie-breaking rule, such as selecting the first maximum or choosing randomly among the tied arms.

## Compact pseudocode

```text
input: number of rounds T, number of arms D

counts[1..D] = 0
reward_sums[1..D] = 0

for t = 1 to T:
    if an untried arm exists:
        arm = an untried arm
    else:
        for each arm i:
            mean_i = reward_sums[i] / counts[i]
            bonus_i = sqrt(3 * log(t) / (2 * counts[i]))
            score_i = mean_i + bonus_i

        arm = argmax(score)

    reward = observe_reward(arm)
    counts[arm] += 1
    reward_sums[arm] += reward
```

## Small numerical example

At round $t=100$, suppose two advertisements have:

| Arm | Observed mean | Selections |
|---|---:|---:|
| A | 0.18 | 50 |
| B | 0.12 | 10 |

Using the course exploration term:

$$
U_A
=
0.18+\sqrt{\frac{3\ln(100)}{2(50)}}
\approx 0.552
$$

$$
U_B
=
0.12+\sqrt{\frac{3\ln(100)}{2(10)}}
\approx 0.951
$$

Although B has a lower observed mean, it has been tested much less often. Its larger uncertainty bonus makes it the optimistic choice for this round.

This illustrates how UCB explores without selecting arms uniformly at random.

## Advertising interpretation

For each advertisement $i$:

- $N_i$ is the number of times the ad has been displayed;
- $S_i$ is the number of clicks received;
- $\widehat{\mu}_i=S_i/N_i$ is its observed click-through rate;
- the bonus measures uncertainty caused by limited exposure;
- the UCB score balances estimated performance and uncertainty.

As data accumulate:

- clearly weak advertisements receive fewer impressions;
- promising advertisements receive more impressions;
- insufficiently tested advertisements still receive occasional opportunities;
- cumulative clicks can exceed those from uniform allocation when the assumptions fit the problem.

## Regret behavior

Let:

$$
\Delta_i=\mu^*-\mu_i
$$

be the expected-reward gap between an optimal arm and arm $i$.

For stationary bounded rewards, UCB1 achieves logarithmic problem-dependent expected regret. Informally:

$$
\mathbb{E}[R_T]
=
O\left(
\sum_{i:\Delta_i>0}
\frac{\ln T}{\Delta_i}
\right)
$$

Equivalently, a suboptimal arm is selected roughly $O(\ln T/\Delta_i^2)$ times under the standard analysis.

The practical meaning is that the proportion of rounds spent on clearly suboptimal arms tends toward zero as the horizon grows.

## Is this a conventional confidence interval?

The UCB bonus is motivated by concentration inequalities such as Hoeffding's inequality. It creates a high-probability optimistic index under specified assumptions.

However, it should not automatically be interpreted as an ordinary fixed-level statistical confidence interval:

- the interval is recalculated adaptively;
- arms are selected using previous observations;
- the probability guarantee depends on the exact formula and assumptions;
- the true mean is not guaranteed to lie inside every displayed bound at every time.

“Confidence bound” is therefore best understood here as an algorithmic optimism device derived from probability bounds.

## UCB compared with fixed A/B testing

UCB and fixed randomized experiments optimize different priorities:

- **UCB** primarily seeks high cumulative reward while learning.
- **A/B testing** commonly prioritizes clean statistical inference and effect estimation.

UCB is not automatically more sophisticated or appropriate in every setting. A fixed experiment may be preferable when unbiased estimation, regulatory review, long-delayed outcomes, or a predeclared hypothesis test is the main requirement.

Adaptive allocation also requires suitable statistical analysis; treating adaptively collected observations as if they came from a conventional fixed design can produce misleading conclusions.

## Assumptions and limitations

Basic UCB1 is most appropriate when:

- the number of arms is finite;
- reward distributions are stationary;
- rewards are bounded;
- feedback is observed quickly;
- each arm's rewards are independent across rounds under the model;
- the reward objective is correctly specified.

Potential problems include:

- **nonstationarity:** advertisement performance changes over time;
- **context:** different users prefer different advertisements;
- **delayed rewards:** conversions arrive well after impressions;
- **multiple objectives:** clicks may not correspond to profit or customer value;
- **selection effects:** adaptive exposure changes the collected dataset;
- **constraints:** budgets, fairness, inventory, and brand safety limit choices.

Relevant extensions include discounted or sliding-window UCB, contextual UCB, delayed-feedback bandits, and constrained bandits.

## Common misconceptions

### “The best arm has the most right-shifted-looking distribution”

The best arm is the one with the greatest expected reward for the chosen objective. A visual distribution may be misleading if it has high variance or rare extreme values.

### “Every new observation makes the estimate better”

More data improve estimation overall, but a single random reward can temporarily move the empirical mean farther from the true mean.

### “Every confidence bonus always decreases”

The bonus generally decreases when that arm is selected. For an ignored arm, the time term grows while its selection count remains fixed, so its score can rise.

### “Once UCB finds the winner, it uses only that arm”

Standard UCB continues limited exploration. It increasingly favors optimal arms but does not ordinarily make an irreversible early commitment.

### “UCB should always replace A/B testing”

The right method depends on whether the primary objective is cumulative reward, best-arm identification, causal inference, or another operational goal.

## Quick review

1. **What are the two parts of a UCB score?**

   An empirical reward estimate and an uncertainty bonus.

2. **Why are untried arms selected?**

   Their values are unknown, and the usual formula is undefined when the selection count is zero.

3. **What causes an arm's bonus to shrink?**

   Increasing that arm's number of selections.

4. **Why can a lower-mean arm still be selected?**

   Its uncertainty bonus may be large enough to give it the highest optimistic score.

5. **What does the logarithmic time term accomplish?**

   It allows neglected arms to become competitive again, preserving exploration.

6. **Does UCB guarantee the best arm after a fixed number of rounds?**

   No. It manages uncertainty and regret probabilistically under model assumptions.

7. **What is the advertising empirical mean?**

   The observed click-through rate, calculated as clicks divided by displays.

8. **When may basic UCB1 be inappropriate?**

   When rewards are nonstationary, delayed, strongly contextual, unbounded, or subject to important constraints not included in the model.

## Key takeaway

Upper Confidence Bound selects the arm with the most optimistic plausible value:

$$
\text{estimated reward}+\text{uncertainty bonus}
$$

Frequently successful arms are exploited, while uncertain arms remain eligible for exploration. As evidence accumulates, the uncertainty bonuses narrow and the algorithm concentrates more heavily on arms with high expected rewards, achieving low regret under the standard stochastic-bandit assumptions.
