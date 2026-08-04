# Stochastic Gradient Descent

## Lesson objective

This lesson compares three ways to estimate gradients and update a neural network:

1. full-batch gradient descent;
2. stochastic gradient descent; and
3. mini-batch gradient descent.

They use the same basic update rule but differ in how many training observations contribute to each gradient estimate.

## 1. The shared optimization objective

Suppose the training set contains \(n\) observations. For parameters \(\boldsymbol{\theta}\), define the empirical objective:

\[
J(\boldsymbol{\theta})
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}_i(\boldsymbol{\theta})
\]

where \(\mathcal{L}_i\) is the loss for observation \(i\).

All three methods seek parameter values that reduce this objective. Their generic update is:

\[
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_t
-
\eta_t\mathbf{g}_t
\]

where:

- \(\eta_t\) is the learning rate at step \(t\); and
- \(\mathbf{g}_t\) is a full or estimated gradient.

## 2. Does gradient descent require a convex objective?

No. Ordinary gradient descent can be applied to differentiable non-convex objectives. Convexity is not a requirement for running the algorithm.

Convexity changes the available guarantees:

- for a convex objective, every local minimum is also global;
- under suitable smoothness and learning-rate conditions, convergence analysis is comparatively straightforward;
- for a non-convex objective, the optimizer may encounter saddle points, plateaus, local minima, and many low-loss regions; and
- gradient-based optimization is not generally guaranteed to find the global minimum of a neural-network objective.

Neural-network training routinely uses gradient methods despite non-convexity.

## 3. Why a squared loss can still produce a non-convex objective

The per-example squared loss:

\[
\mathcal{L}(y,\hat{y})
=
\frac{1}{2}(\hat{y}-y)^2
\]

is convex as a function of the scalar prediction \(\hat{y}\). That does not imply it is convex as a function of every neural-network parameter.

If:

\[
\hat{y}=f(\mathbf{x};\boldsymbol{\theta})
\]

is a nonlinear multilayer network, then:

\[
J(\boldsymbol{\theta})
=
\frac{1}{n}
\sum_{i=1}^{n}
\frac{1}{2}
\left(
f(\mathbf{x}^{(i)};\boldsymbol{\theta})-y^{(i)}
\right)^2
\]

is generally non-convex in \(\boldsymbol{\theta}\). Non-convexity comes from the complete composition of the model and objective, not merely from selecting a loss that looks non-convex in its direct input.

## 4. Full-batch gradient descent

Full-batch gradient descent calculates the exact gradient of the empirical training objective before each update:

\[
\mathbf{g}_t
=
\nabla J(\boldsymbol{\theta}_t)
=
\frac{1}{n}
\sum_{i=1}^{n}
\nabla\mathcal{L}_i(\boldsymbol{\theta}_t)
\]

Then:

\[
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_t
-
\eta_t\nabla J(\boldsymbol{\theta}_t)
\]

Conceptually:

```text
all training observations
        ↓
aggregate full-data gradient
        ↓
one parameter update
```

### Advantages

- The gradient is exact for the current finite training set.
- Repeated evaluations at the same parameters are deterministic under deterministic computation.
- The objective often follows a smoother trajectory.
- It can work well for small datasets and some convex optimization problems.

### Limitations

- Each update can be expensive for a large dataset.
- The model must wait for all training examples to contribute before updating.
- Full-data computation can exceed available memory if implemented naively.
- It provides fewer parameter updates per pass through the data.

Full-batch computation does not strictly require loading the entire dataset into memory simultaneously; gradients can be accumulated in chunks. The defining property is that every training observation contributes to each full gradient before the update.

## 5. Stochastic gradient descent

Strict stochastic gradient descent samples one training observation \(i_t\) and uses its gradient:

\[
\mathbf{g}_t
=
\nabla\mathcal{L}_{i_t}(\boldsymbol{\theta}_t)
\]

The update is:

\[
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_t
-
\eta_t
\nabla\mathcal{L}_{i_t}(\boldsymbol{\theta}_t)
\]

Conceptually:

```text
one sampled observation
        ↓
one sample gradient
        ↓
one parameter update
```

If observations are sampled uniformly, the stochastic gradient is an unbiased estimator of the full gradient under ordinary assumptions:

\[
\mathbb{E}_{i_t}
\left[
\nabla\mathcal{L}_{i_t}(\boldsymbol{\theta})
\right]
=
\nabla J(\boldsymbol{\theta})
\]

An individual estimate can nevertheless differ greatly from the full gradient.

### Advantages

- Each update is inexpensive.
- Learning can begin without waiting for a full-data pass.
- It can process streaming or very large datasets.
- Gradient noise may help the trajectory move through flat regions or away from some unstable stationary points.

### Limitations

- Individual updates have high variance.
- The objective can fluctuate rather than decrease at every step.
- A fixed learning rate may cause continuing oscillation around a low-loss region.
- One-example operations usually underuse GPU and accelerator parallelism.
- More updates may be required to reach a stable result.

## 6. Mini-batch gradient descent

Mini-batch gradient descent uses a subset \(B_t\) of the training observations:

\[
\mathbf{g}_t
=
\frac{1}{|B_t|}
\sum_{i\in B_t}
\nabla\mathcal{L}_i(\boldsymbol{\theta}_t)
\]

and updates:

\[
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_t
-
\eta_t\mathbf{g}_t
\]

Conceptually:

```text
small batch of observations
          ↓
average mini-batch gradient
          ↓
one parameter update
```

Mini-batch training is the standard approach for most modern deep-learning workloads because it balances:

- computational parallelism;
- memory usage;
- gradient variance;
- update frequency; and
- throughput on accelerators.

Despite common informal usage, much of what practitioners call “SGD” is actually mini-batch stochastic gradient descent.

## 7. Side-by-side comparison

| Property | Full batch | Stochastic | Mini-batch |
|---|---:|---:|---:|
| Examples per update | All \(n\) | 1 | \(1<B<n\) |
| Gradient variance | None for fixed finite data | High | Intermediate |
| Update frequency per epoch | 1 | \(n\) | Approximately \(\lceil n/B\rceil\) |
| Hardware parallelism | Potentially high | Usually poor | Usually strong |
| Memory pressure | Potentially high | Low | Controllable |
| Objective trajectory | Smoothest | Noisiest | Moderately noisy |
| Typical deep-learning use | Uncommon for very large datasets | Uncommon in strict one-sample form | Standard |

The categories meet at their boundaries:

- batch size \(B=1\) gives strict SGD;
- batch size \(B=n\) gives full-batch gradient descent; and
- values between them give mini-batch training.

## 8. A small numerical comparison

Consider a linear model:

\[
\hat{y}=wx
\]

with two training observations:

\[
(x_1,y_1)=(1,2)
\]

\[
(x_2,y_2)=(2,4)
\]

Use per-example loss:

\[
\mathcal{L}_i=\frac{1}{2}(wx_i-y_i)^2
\]

whose gradient is:

\[
\frac{\partial\mathcal{L}_i}{\partial w}
=(wx_i-y_i)x_i
\]

Let \(w_0=0\) and \(\eta=0.1\).

### One full-batch update

At \(w_0=0\):

\[
g_1=(0-2)(1)=-2
\]

\[
g_2=(0-4)(2)=-8
\]

The mean gradient is:

\[
g_{\text{batch}}=\frac{-2+(-8)}{2}=-5
\]

Update:

\[
w_1=0-(0.1)(-5)=0.5
\]

### Two stochastic updates

Process the first observation:

\[
w_1=0-(0.1)(-2)=0.2
\]

Now the second observation is evaluated using the updated weight \(w_1=0.2\):

\[
g_2=((0.2)(2)-4)(2)=-7.2
\]

\[
w_2=0.2-(0.1)(-7.2)=0.92
\]

The full-batch and stochastic methods follow different trajectories because SGD updates the model between observations. Changing the SGD sample order can change the trajectory again.

## 9. Does SGD find the global minimum?

Not necessarily. Gradient noise can sometimes help an optimizer move away from saddle points, flat areas, or narrow basins, but SGD provides no general guarantee of reaching the global minimum of a non-convex neural-network objective.

The lecture’s intuition should therefore be softened:

```text
stochasticity may improve exploration
```

is reasonable, whereas:

```text
stochasticity ensures the global minimum is found
```

is incorrect.

In high-dimensional neural networks, saddle points and broad low-loss regions can be as important as isolated local minima. Furthermore, the parameter set with the smallest training loss need not generalize best.

## 10. Is SGD always faster?

There are several meanings of “faster”:

- time per update;
- number of updates to convergence;
- number of examples processed;
- wall-clock training time;
- hardware utilization; and
- time to reach a target validation metric.

Strict SGD has a cheap individual update and a low memory requirement, but it performs many sequential updates and usually makes poor use of vectorized hardware. Full-batch descent performs expensive updates but uses highly aggregated gradients. Mini-batches often achieve the best practical throughput.

No method is universally fastest. Performance depends on the dataset, model, batch size, storage pipeline, accelerator, communication overhead, optimizer, and stopping criterion.

## 11. Randomness and determinism

Full-batch gradient descent can be deterministic when:

- initialization is fixed;
- data and preprocessing are fixed;
- computation uses deterministic operations; and
- all numerical and software conditions are controlled.

SGD becomes stochastic when examples are randomly sampled or shuffled. However, using the same random seed and deterministic operations can reproduce the same sample order and trajectory.

Conversely, two runs can still differ because of nondeterministic accelerator kernels, parallel reductions, distributed scheduling, floating-point ordering, data-loader workers, or other random components.

The name **stochastic gradient descent** describes the gradient estimator, not a guarantee that every software run must be irreproducible.

## 12. Data order and shuffling

If data is ordered by class, time, source, or difficulty, processing it in a fixed sequence can bias successive updates. Ordinary independent-and-identically-distributed training commonly shuffles observations at the beginning of each epoch.

Shuffling is not always appropriate. Time-series forecasting, online learning, curriculum learning, and grouped data can require structure-aware sampling.

The sampling scheme should preserve the assumptions of the task and prevent leakage between related observations.

## 13. Epochs and steps

For \(n\) training observations and batch size \(B\), one ordinary epoch contains approximately:

\[
\left\lceil\frac{n}{B}\right\rceil
\]

optimizer steps.

Example with \(n=8{,}000\):

| Batch size | Approximate steps per epoch |
|---:|---:|
| 1 | 8,000 |
| 32 | 250 |
| 100 | 80 |
| 8,000 | 1 |

One epoch means a pass through the training examples under the chosen sampling convention. One step means one parameter update.

## 14. Batch size as a hyperparameter

Batch size influences:

- gradient variance;
- memory consumption;
- accelerator utilization;
- number of updates per epoch;
- interaction with normalization layers;
- suitable learning-rate scale; and
- sometimes generalization behavior.

Larger batches reduce sampling noise but do not automatically improve convergence or accuracy. Very large-batch training may require learning-rate scaling, warm-up, distributed communication, or other adjustments.

The correct value should be selected using hardware constraints and validation evidence rather than a rule that batches must contain 5, 10, 100, or any other fixed number.

## 15. Learning-rate schedules and SGD convergence

With a fixed learning rate, stochastic updates may continue to fluctuate around a low-loss region. Reducing the learning rate over time can support finer convergence:

\[
\eta_t \downarrow \text{ as training progresses}
\]

Common schedules include:

- step decay;
- exponential decay;
- cosine decay;
- warm-up followed by decay; and
- reductions triggered by validation plateaus.

Momentum is also frequently combined with SGD:

\[
\mathbf{v}_{t+1}
=
\beta\mathbf{v}_t
+
\mathbf{g}_t
\]

\[
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_t
-
\eta\mathbf{v}_{t+1}
\]

Momentum can smooth noisy directions and accelerate movement along directions with persistent gradients.

## 16. Online learning versus shuffled SGD

The terms are related but not identical:

- **shuffled SGD** repeatedly samples from a fixed training dataset;
- **online learning** updates as new observations arrive over time, potentially from a changing distribution.

Online systems must consider concept drift, delayed labels, data quality, monitoring, and whether older observations should remain influential.

## 17. Practical training workflow

A typical mini-batch training loop is:

```python
for epoch in range(number_of_epochs):
    shuffle_training_data_if_appropriate()

    for X_batch, y_batch in mini_batches:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_function(y_pred, y_batch)
        loss.backward()      # backpropagation
        optimizer.step()     # parameter update

    evaluate_validation_metrics()
```

This is conceptual pseudocode. Exact method names vary by framework, and some APIs handle gradient clearing or update steps differently.

## 18. Selecting the best model

The objective is not necessarily to find the absolute lowest training loss. Model selection should consider:

- validation loss and task-specific metrics;
- robustness across random seeds;
- generalization to relevant subgroups;
- calibration;
- training cost and inference constraints;
- sensitivity to data shift; and
- final performance on an untouched test set.

Different stochastic trajectories can produce models with similar training loss but different validation behavior.

## Common pitfalls

- Claiming gradient descent requires a convex objective.
- Assuming squared error guarantees convexity in neural-network parameters.
- Saying SGD reliably escapes every local minimum or finds the global minimum.
- Treating every stationary point as a local minimum.
- Assuming strict one-example SGD is always faster in wall-clock time.
- Saying full-batch descent must load the entire dataset into memory at once.
- Confusing a cheap update with fast overall convergence.
- Calling a mini-batch method “batch gradient descent” without stating the batch size.
- Assuming stochastic training cannot be reproduced with controlled seeds and deterministic operations.
- Forgetting to shuffle ordinary independent training data when order may be harmful.
- Comparing methods only by epochs when their numbers of updates differ.
- Ignoring the interaction between batch size and learning rate.
- Using the test set to choose batch size or stopping time.
- Referring to Andrew Trask as “Andrew Truk” or the article as “13 Lives of Python.”

## Key takeaways

- Gradient methods can optimize both convex and non-convex objectives.
- Convexity provides stronger global guarantees; it is not required to run gradient descent.
- Full-batch descent uses every training observation for each gradient.
- Strict SGD uses one sampled observation per update.
- Mini-batch descent uses an intermediate batch size and is standard in deep learning.
- Stochastic gradients introduce variance that can help exploration but do not guarantee a global optimum.
- Strict SGD is cheap per update but is not universally fastest overall.
- All observations share the same model parameters.
- Batch size affects memory, throughput, gradient noise, and the number of steps per epoch.
- Random seeds improve reproducibility, but other nondeterministic operations may still matter.
- Validation performance, not minimum training loss alone, should guide model selection.

## Review questions

1. Why can gradient descent be applied to a non-convex objective?
2. How can squared prediction error become non-convex in neural-network parameters?
3. Write the gradient estimator for full-batch, stochastic, and mini-batch training.
4. Under what sampling condition is a one-example stochastic gradient unbiased?
5. Why can SGD’s objective value fluctuate between steps?
6. Why does stochasticity not guarantee discovery of the global minimum?
7. In what senses can an optimization method be described as “faster”?
8. Why does strict SGD often underuse a GPU?
9. How do batch size 1, an intermediate batch size, and batch size \(n\) relate?
10. How many optimizer steps occur in an ordinary epoch with 8,000 observations and batch size 100?
11. What factors besides the random seed can affect reproducibility?
12. Why might the learning rate be reduced during stochastic training?
13. How does online learning differ from repeatedly shuffling a fixed dataset?

## Further reading

- [Andrew Trask: *A Neural Network in 13 Lines of Python (Part 2 — Gradient Descent)*](https://iamtrask.github.io/2015/07/27/python-network-part2/)
- [Michael Nielsen: *Neural Networks and Deep Learning*](https://neuralnetworksanddeeplearning.com/)
