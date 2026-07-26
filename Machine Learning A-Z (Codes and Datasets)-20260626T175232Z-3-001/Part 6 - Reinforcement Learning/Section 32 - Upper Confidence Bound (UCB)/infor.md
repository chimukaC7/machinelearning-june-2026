# The Multi-Armed Bandit Problem

## Overview

The **multi-armed bandit problem** is a classic model of sequential decision-making under uncertainty. At each round, an agent chooses one action from several alternatives, observes only the reward produced by that choice, and then uses the new information to make a better decision in the next round.

The objective is to maximize total reward over time. To do so, the agent must balance two competing goals:

- **Exploration:** try uncertain actions to learn how rewarding they are.
- **Exploitation:** select the action that currently appears to be the best.

This exploration–exploitation trade-off is central to reinforcement learning, online experimentation, recommendation systems, adaptive clinical trials, resource allocation, and advertising.

## Relationship to reinforcement learning

Reinforcement learning is broader than the multi-armed bandit setting. A general reinforcement-learning problem may include:

- different environmental states;
- actions that change the next state;
- delayed consequences;
- a sequence of rewards;
- long-term planning.

For example, a robot learning to walk must coordinate actions over time. Moving one leg changes its balance and therefore changes the state from which the next action is taken. A reward function might encourage forward movement, stability, energy efficiency, and safety.

Rewards are not restricted to `1` for success and `0` for failure. They may be positive, zero, or negative, and their scale should reflect the task. A fall might receive a negative reward, while stable forward progress might receive a positive reward.

A basic bandit problem is simpler:

- there is usually no changing environmental state;
- each round consists of choosing one arm and observing its reward;
- the action does not determine a new state for the next round.

For this reason, a multi-armed bandit can be viewed as a particularly simple reinforcement-learning or online-learning problem.

## Why is it called a bandit?

A traditional slot machine is sometimes called a **one-armed bandit**:

- **one-armed** refers to the mechanical lever used to operate older machines;
- **bandit** refers humorously to the machine taking the player's money.

A **multi-armed bandit** imagines several slot machines placed side by side. Each machine is an **arm**, and each arm produces rewards according to its own unknown process.

The learner must decide which machine to play at each round.

## Formal stochastic-bandit model

Assume there are $K$ arms:

$$
\mathcal{A}=\{1,2,\ldots,K\}
$$

At round $t$:

1. the learner chooses an arm $A_t$;
2. the selected arm generates a reward $X_t$;
3. the learner observes that reward;
4. the learner updates its knowledge before the next round.

In a standard **stationary stochastic bandit**, each arm $i$ has a fixed but unknown reward distribution with expected reward:

$$
\mu_i=\mathbb{E}[X_t\mid A_t=i]
$$

The optimal arm is any arm with the greatest expected reward:

$$
i^*\in\arg\max_i \mu_i
$$

and its expected reward is:

$$
\mu^*=\max_i\mu_i
$$

The best arm is defined by its **expected reward**, not necessarily by its median, mode, skewness, or largest possible jackpot. Two or more arms may also tie for the highest mean.

## Example: unknown slot machines

Suppose a player can make 1,000 selections from five machines. Each machine has a different, unknown reward distribution.

If those distributions were known, the reward-maximizing strategy would usually select an arm with the greatest mean reward on every round. In the bandit problem, however, the means are unknown and must be estimated from observed rewards.

Selecting an arm provides two benefits:

- an immediate reward;
- information that can improve future decisions.

Trying an unfamiliar arm may sacrifice reward now but reveal that it is better than the current favorite. Repeatedly choosing the current favorite may earn more immediately but leave potentially better alternatives unexplored.

## Exploration versus exploitation

### Exploration

Exploration gathers information. It may involve:

- selecting every arm initially;
- occasionally choosing a less-tested arm;
- favoring arms with high uncertainty;
- deliberately collecting enough evidence to compare alternatives.

Too little exploration can cause the learner to commit prematurely to a suboptimal arm because of lucky early observations.

### Exploitation

Exploitation uses current knowledge by selecting the arm with the highest estimated value.

Too much exploitation can trap the learner on an arm that only appeared good because of random variation. Too much exploration, on the other hand, wastes opportunities on arms that are already known to be inferior.

An effective bandit algorithm does not generally separate learning and earning into two completely independent stages. It adapts its choices as evidence accumulates.

## Regret

Bandit algorithms are commonly evaluated using **regret**: the reward lost by not selecting an optimal arm on every round.

Let $T$ be the number of rounds. One standard definition of expected pseudo-regret is:

$$
R_T
=
T\mu^*
-
\mathbb{E}\left[\sum_{t=1}^{T}\mu_{A_t}\right]
$$

Equivalently, define the suboptimality gap for arm $i$ as:

$$
\Delta_i=\mu^*-\mu_i
$$

If $N_i(T)$ is the number of times arm $i$ is selected by time $T$, then:

$$
R_T
=
\sum_{i=1}^{K}\Delta_i\,\mathbb{E}[N_i(T)]
$$

This expression provides useful intuition:

- selecting an optimal arm adds no expected regret;
- selecting a suboptimal arm adds regret proportional to its gap;
- exploring a nearly optimal arm is less costly than exploring a very poor arm;
- a good algorithm learns to select suboptimal arms less frequently over time.

Regret is not simply the difference between the largest reward observed and the reward received on one round. It is a cumulative comparison with an optimal policy under the assumed model.

## Bernoulli bandits for online advertising

Online advertising provides a simple modern example. Suppose a campaign has five candidate advertisements. Each time an eligible user arrives, the system chooses one advertisement to display.

For a click-based objective, define:

$$
X_t=
\begin{cases}
1,&\text{if the user clicks}\\
0,&\text{otherwise}
\end{cases}
$$

Each advertisement $i$ has an unknown click-through probability:

$$
p_i=P(X_t=1\mid A_t=i)
$$

Because the reward is Bernoulli:

$$
\mu_i=p_i
$$

The objective is to maximize the number of clicks over the campaign while learning which advertisement has the highest click-through rate.

The bandit algorithm should:

1. show every advertisement enough times to obtain useful evidence;
2. update each estimated click-through rate;
3. allocate more impressions to promising advertisements;
4. retain some exploration while meaningful uncertainty remains.

## A/B testing compared with bandit allocation

A conventional randomized A/B/n test often allocates traffic according to fixed probabilities throughout the experiment. Its primary purpose is usually reliable statistical comparison.

A bandit policy adapts allocation while the experiment is running. Better-performing arms may receive more traffic before the experiment ends, potentially increasing cumulative reward.

The distinction should not be overstated:

| Fixed randomized experiment | Multi-armed bandit |
|---|---|
| Commonly prioritizes inference | Commonly prioritizes cumulative reward |
| Allocation is usually fixed in advance | Allocation adapts to observed rewards |
| Straightforward classical analysis | Adaptive data require appropriate analysis |
| Continues sampling weaker variants as planned | May reduce traffic to weak variants early |

An A/B test is not literally reward-free or “pure exploration”—users may still respond to the better option during the test. Rather, it generally does not adapt traffic to exploit the apparent winner as aggressively as a reward-maximizing bandit.

Bandits are also not automatically superior. Fixed experiments may be preferable when:

- unbiased effect estimation is the main objective;
- treatment effects must be reported with conventional statistical guarantees;
- outcomes are delayed;
- policies, regulations, or fairness constraints require stable allocation;
- the environment changes faster than the algorithm can learn.

## Upper Confidence Bound intuition

Upper Confidence Bound algorithms address exploration and exploitation using **optimism under uncertainty**.

For each arm, they construct an index containing:

1. an estimated mean reward;
2. an uncertainty bonus.

Conceptually:

$$
\operatorname{UCB}_i(t)
=
\widehat{\mu}_i(t)
+
\text{uncertainty bonus}_i(t)
$$

An arm receives a high index when:

- its observed average reward is high;
- it has been selected too few times to rule out a high true mean;
- or both.

The algorithm selects the arm with the greatest upper bound. Frequently selected arms develop narrower bounds, while underexplored arms retain larger uncertainty bonuses. This creates exploration without relying entirely on random choice.

The exact UCB formula and its implementation are developed in the following lessons.

## Assumptions and practical complications

The elementary stochastic-bandit model usually assumes:

- a finite set of arms;
- stationary reward distributions;
- rewards that are independent across rounds, conditional on the chosen arm;
- immediate and observable feedback;
- one common objective;
- no important user context.

Real advertising systems may violate these assumptions:

- click-through rates change over time;
- users differ by location, device, interests, and prior exposure;
- conversions may arrive hours or days later;
- showing an advertisement can change future behavior;
- inventory and budgets impose constraints;
- clicks may not represent revenue or long-term value;
- fairness and brand-safety requirements restrict allocation.

Possible extensions include nonstationary bandits, contextual bandits, delayed-feedback models, constrained bandits, and adversarial bandits.

## Correct reference

The paper mentioned in the lecture is:

> Peter Auer, **“Using Confidence Bounds for Exploitation-Exploration Trade-offs,”** *Journal of Machine Learning Research*, 3:397–422, 2002.

A closely related foundational paper introducing and analyzing UCB1 is:

> Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer, **“Finite-time Analysis of the Multiarmed Bandit Problem,”** *Machine Learning*, 47:235–256, 2002.

These are separate papers. The first develops confidence-bound techniques for broader exploration–exploitation settings; the second gives a finite-time analysis of stochastic multi-armed-bandit policies, including UCB1.

## Quick review

1. **What is an arm?**

   One available action, such as a slot machine, advertisement, or recommendation strategy.

2. **What is exploration?**

   Selecting actions partly to learn more about their reward distributions.

3. **What is exploitation?**

   Selecting the action currently believed to have the greatest expected reward.

4. **What makes an arm optimal?**

   It has the highest expected reward under the chosen objective.

5. **What is regret?**

   The cumulative expected reward lost relative to consistently selecting an optimal arm.

6. **Why can greedy selection fail?**

   Early random outcomes may make a suboptimal arm appear best.

7. **Why does UCB explore?**

   Uncertain, under-sampled arms receive larger confidence bonuses.

8. **How does a bandit differ from a standard fixed A/B test?**

   A bandit adapts future allocation using observed rewards, whereas a fixed test normally preserves predetermined allocation probabilities.

9. **Is a basic bandit a complete reinforcement-learning environment?**

   No. It is a simplified setting without state transitions or long-term action consequences.

## Key takeaway

The multi-armed bandit problem asks how to maximize cumulative reward while learning which action is best. Exploration improves knowledge; exploitation uses that knowledge; regret measures the cost of imperfect decisions. Advertising selection is a natural example, and Upper Confidence Bound algorithms provide a principled way to favor strong arms without ignoring uncertain alternatives.
