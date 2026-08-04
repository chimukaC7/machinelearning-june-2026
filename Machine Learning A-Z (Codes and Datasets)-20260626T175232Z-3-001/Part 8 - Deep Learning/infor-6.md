# Gradient Descent

## Lesson objective

Gradient descent is an optimization method used to reduce a model’s objective function. During neural-network training, backpropagation computes gradients, and an optimizer uses those gradients to update the weights, biases, and other trainable parameters.

The central update rule is:

\[
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_{t}
-
\eta\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}_{t})
\]

where:

- \(\boldsymbol{\theta}\) represents all trainable parameters;
- \(J\) is the objective function;
- \(\nabla J\) is the gradient;
- \(\eta\) is the learning rate; and
- \(t\) identifies the current optimization step.

## 1. Recap: prediction, loss, and gradients

For an input \(\mathbf{x}\), a neural network produces:

\[
\hat{y} = f(\mathbf{x};\boldsymbol{\theta})
\]

The prediction \(\hat{y}\) is compared with the true target \(y\) using a loss function. For a simple regression example:

\[
\mathcal{L}(y,\hat{y})
=
\frac{1}{2}(\hat{y}-y)^2
\]

Across a training set, these per-example losses are aggregated into an objective such as:

\[
J(\boldsymbol{\theta})
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}\left(y^{(i)},\hat{y}^{(i)}\right)
\]

The objective may also contain regularization terms. The goal of optimization is to find parameter values that make this objective suitably small while retaining good performance on unseen data.

## 2. Backpropagation is not gradient descent

These two ideas work together but have different roles:

| Stage | Purpose |
|---|---|
| Forward pass | Compute predictions and the loss |
| Backpropagation | Compute derivatives of the loss with respect to trainable parameters |
| Optimizer step | Use those derivatives to update the parameters |

Backpropagation applies the chain rule efficiently. Gradient descent is one rule for moving the parameters using the resulting gradient.

It is therefore more precise to say:

```text
backpropagation computes the gradient;
gradient descent uses it to update parameters.
```

The squared error itself is not literally propagated backward. What moves backward through the computational graph are derivative calculations that quantify parameter sensitivity.

## 3. Why not search every possible weight?

Imagine that a model has only one trainable weight \(w\). One could choose many candidate values, evaluate the objective for each, and keep the best:

```text
candidate weight → prediction → objective value
```

If this relationship is plotted, the horizontal axis should represent the candidate weight \(w\), and the vertical axis should represent \(J(w)\):

```text
vertical axis:   J(w)
horizontal axis: w
```

Plotting cost against \(\hat{y}\) illustrates the shape of squared prediction error, but it is not the direct picture of brute-force parameter optimization. The optimizer chooses parameters, and those parameters indirectly determine \(\hat{y}\).

For one parameter over a small bounded interval, exhaustive grid search may be possible. Neural networks, however, contain many parameters over continuous spaces.

## 4. Combinatorial growth of grid search

Suppose each parameter is restricted to 1,000 candidate values. For \(d\) parameters, a full grid contains:

\[
1000^d
\]

combinations.

For 25 parameters:

\[
1000^{25}
=
(10^3)^{25}
=
10^{75}
\]

candidate configurations.

This exponential growth makes a dense grid infeasible. It is one manifestation of the difficulties associated with high-dimensional spaces, although the phrase **curse of dimensionality** also covers broader statistical and geometric phenomena.

### Counting the parameters correctly

For a fully connected network with:

- 4 input features;
- 5 hidden units; and
- 1 output unit;

the connection weights are:

\[
(4\times5)+(5\times1)=25
\]

If every hidden and output unit also has a bias, there are:

\[
5+1=6
\]

bias parameters. The total trainable parameter count is therefore:

\[
25+6=31
\]

The lecture’s count of 25 includes connection weights but omits biases.

## 5. The Sunway TaihuLight illustration

Sunway TaihuLight was ranked first on the June 2016 TOP500 list, achieving approximately 93 petaflop/s on the High Performance Linpack benchmark. That ranking belongs to 2016 and should not be presented as current.

Even under the unrealistic assumption that one candidate network could be evaluated with a single floating-point operation, checking \(10^{75}\) candidates at \(93\times10^{15}\) operations per second would require approximately:

\[
\frac{10^{75}}{93\times10^{15}}
\approx
1.08\times10^{58}\text{ seconds}
\]

or roughly:

\[
3.4\times10^{50}\text{ years}
\]

depending on the rounding convention. A real network evaluation requires many operations, so this already enormous lower bound is extremely optimistic.

As of the June 2026 TOP500 list, LineShine holds the top HPL ranking at 2.198 exaflop/s. That substantial hardware improvement does not make exhaustive search over a \(10^{75}\)-point grid practical.

The lesson is not that supercomputers are slow. It is that exhaustive enumeration scales exponentially with the number of parameters.

## 6. The derivative in one dimension

For a one-parameter objective \(J(w)\), the derivative:

\[
\frac{dJ}{dw}
\]

describes the local slope.

- If \(dJ/dw>0\), increasing \(w\) locally increases the objective, so gradient descent decreases \(w\).
- If \(dJ/dw<0\), increasing \(w\) locally decreases the objective, so gradient descent increases \(w\).
- If \(dJ/dw=0\), the point is stationary; it may be a minimum, maximum, saddle-like flat point, or part of a plateau.

The update is:

\[
w_{t+1}
=
w_t
-
\eta\frac{dJ}{dw}\bigg|_{w=w_t}
\]

Subtracting the derivative moves in the locally downhill direction.

## 7. The gradient in multiple dimensions

For parameters:

\[
\boldsymbol{\theta}
=
[\theta_1,\theta_2,\ldots,\theta_d]^{\mathsf{T}}
\]

the gradient is the vector of partial derivatives:

\[
\nabla J(\boldsymbol{\theta})
=
\begin{bmatrix}
\frac{\partial J}{\partial\theta_1} \\
\frac{\partial J}{\partial\theta_2} \\
\vdots \\
\frac{\partial J}{\partial\theta_d}
\end{bmatrix}
\]

Under the usual Euclidean interpretation, the gradient points in the direction of steepest local increase. Its negative points in the direction of steepest local decrease:

\[
-\nabla J(\boldsymbol{\theta})
\]

All parameters should conceptually be updated using derivatives evaluated at the same current parameter state. Software performs these updates efficiently with vectors, matrices, and automatic differentiation.

## 8. A worked one-dimensional example

Consider:

\[
\hat{y}=wx
\]

with one observation:

\[
x=2, \qquad y=6
\]

and loss:

\[
J(w)=\frac{1}{2}(wx-y)^2
\]

The derivative is:

\[
\frac{dJ}{dw}=(wx-y)x
\]

Let the initial weight be \(w_0=0\) and the learning rate be \(\eta=0.1\).

### Step 0

\[
\hat{y}_0=(0)(2)=0
\]

\[
J(w_0)=\frac{1}{2}(0-6)^2=18
\]

\[
\frac{dJ}{dw}=(0-6)(2)=-12
\]

Update:

\[
w_1=0-(0.1)(-12)=1.2
\]

### Step 1

\[
\hat{y}_1=(1.2)(2)=2.4
\]

\[
J(w_1)=\frac{1}{2}(2.4-6)^2=6.48
\]

\[
\frac{dJ}{dw}=(2.4-6)(2)=-7.2
\]

Update:

\[
w_2=1.2-(0.1)(-7.2)=1.92
\]

### Step 2

\[
\hat{y}_2=(1.92)(2)=3.84
\]

\[
J(w_2)=\frac{1}{2}(3.84-6)^2=2.3328
\]

The loss decreases over these steps. For this simple example, the minimum occurs at \(w=3\), where \(wx=6\).

## 9. The learning rate

The learning rate \(\eta\) controls the scale of each update.

### Too small

- Training may progress very slowly.
- The optimizer may require many steps to reach a useful region.
- Limited training time may end before convergence.

### Too large

- Updates may overshoot a minimum.
- The objective may oscillate or diverge.
- Numerical instability may occur.

### Suitable scale

- The objective generally decreases at a useful rate.
- Updates remain stable enough to continue learning.

The lecture’s “take 50 steps, then 40” analogy suggests shrinking steps, but plain gradient descent does not automatically reduce the learning rate. With a fixed \(eta\), the actual update magnitude may shrink near a smooth minimum because the gradient itself becomes smaller:

\[
\|\Delta\boldsymbol{\theta}\|
=
\eta\|\nabla J\|
\]

Learning-rate schedules and adaptive optimizers can also change the effective step sizes explicitly.

## 10. Why the path may zigzag

In more than one dimension, an objective can have directions with very different curvature. Gradient descent may bounce across a steep, narrow direction while progressing slowly along a flatter direction. On a contour plot, this produces a zigzagging path.

Feature scaling and appropriate parameter initialization can improve the geometry seen by the optimizer. Momentum can accumulate useful direction across steps and reduce some oscillation.

Zigzagging is possible but not guaranteed; the path depends on the objective surface, initialization, learning rate, batch noise, and optimizer.

## 11. Convex and non-convex objectives

For a simple convex quadratic objective, every local minimum is global, and appropriately configured gradient descent has comparatively straightforward behavior.

Neural-network objectives are generally non-convex and may contain:

- many low-loss regions;
- saddle points;
- flat plateaus;
- sharp and broad valleys;
- symmetries caused by interchangeable hidden units; and
- noisy gradients from mini-batches.

Gradient descent is not guaranteed to find a unique global minimum. In practice, obtaining parameters that generalize well is more important than proving that the smallest possible training objective has been found.

## 12. Local information, not a global map

Gradient descent does not inspect the entire objective surface or know where the global minimum lies. At each step it uses local first-order information:

\[
\nabla J(\boldsymbol{\theta}_t)
\]

It then takes a step and recalculates the gradient at the new location.

The hill analogy is useful, but real optimization differs from a physical ball:

- the parameter space may have millions or billions of dimensions;
- the gradient can be noisy;
- “downhill” depends on a mathematical objective, not gravity;
- momentum is optional and algorithmic; and
- the objective may change statistically as different mini-batches are sampled.

## 13. Batch gradient variants

The name “gradient descent” can refer broadly to related update strategies.

### Full-batch gradient descent

Computes the gradient using the complete training set before each update:

\[
\nabla J(\boldsymbol{\theta})
=
\frac{1}{n}
\sum_{i=1}^{n}
\nabla\mathcal{L}^{(i)}(\boldsymbol{\theta})
\]

The estimate is deterministic for fixed data and parameters but can be expensive on large datasets.

### Stochastic gradient descent

Uses one randomly selected observation per update. The gradient is noisy but inexpensive.

### Mini-batch gradient descent

Uses a subset of observations:

\[
\nabla J_B(\boldsymbol{\theta})
=
\frac{1}{|B|}
\sum_{i\in B}
\nabla\mathcal{L}^{(i)}(\boldsymbol{\theta})
\]

This is the standard approach in deep learning because it combines efficient hardware use with manageable memory and useful stochasticity.

## 14. Beyond plain gradient descent

Common optimizers build on gradient information:

- **SGD with momentum** accumulates a moving direction;
- **Nesterov momentum** evaluates a look-ahead style correction;
- **AdaGrad** adapts learning rates using historical squared gradients;
- **RMSProp** uses a decaying average of squared gradients; and
- **Adam** combines momentum-like first moments with adaptive second moments.

An optimizer’s popularity does not make it universally best. The choice interacts with the learning rate, regularization, batch size, architecture, and task.

## 15. Parameter initialization

Gradient descent needs a starting point. Neural-network weights are commonly initialized randomly using a scale chosen for the architecture and activation function.

Initialization matters because it can affect:

- activation magnitudes;
- gradient magnitudes;
- symmetry between units;
- convergence speed; and
- which region of the objective landscape is explored.

Initializing every weight in the same layer to the same value can prevent hidden units from learning distinct features. Methods such as Glorot/Xavier or He initialization choose variances intended for particular activation behaviors.

## 16. When to stop training

The goal is not simply to keep descending until the training objective cannot be reduced further. Continued optimization can overfit the training data.

Practical stopping criteria may include:

- no meaningful improvement in validation loss;
- early stopping with a patience interval;
- a maximum epoch or step budget;
- a minimum learning rate;
- convergence of relevant metrics; or
- computational and energy constraints.

Training loss and validation performance should be monitored together.

## 17. Gradient checking

When implementing backpropagation manually, gradients can be checked using finite differences:

\[
\frac{\partial J}{\partial\theta_j}
\approx
\frac{
J(\theta_j+\varepsilon)
-
J(\theta_j-\varepsilon)
}{2\varepsilon}
\]

This numerical estimate is useful for debugging small models, but it is much too expensive for ordinary training and can be sensitive to the choice of \(\varepsilon\). Automatic differentiation provides analytical gradients efficiently for real training workloads.

## Common pitfalls

- Saying that backpropagation itself updates the weights.
- Plotting \(J\) against \(\hat{y}\) when explaining a search over candidate weights.
- Forgetting biases when counting trainable parameters.
- Treating Sunway TaihuLight’s 2016 ranking as current.
- Confusing peak FLOP/s, benchmark FLOP/s, and the cost of evaluating one model configuration.
- Assuming that a zero derivative always identifies a minimum.
- Saying the gradient points downhill; it points toward steepest local increase, so its negative points downhill.
- Believing a fixed learning rate automatically shrinks after every step.
- Assuming the gradient-descent path must zigzag.
- Assuming neural-network optimization always reaches the global minimum.
- Using only training loss to decide when the model is ready.
- Treating finite-difference gradient checking as a practical training algorithm.

## Key takeaways

- Gradient descent updates parameters in the direction opposite the objective gradient.
- Backpropagation computes gradients; the optimizer performs updates.
- Exhaustive parameter grids grow exponentially and are infeasible for neural networks.
- A 4–5–1 dense network has 25 connection weights and, with ordinary biases, 31 trainable parameters in total.
- In one dimension, the derivative gives the local slope; in many dimensions, partial derivatives form the gradient vector.
- The learning rate controls update scale and strongly affects stability and speed.
- Mini-batch gradients are the standard practical approach in deep learning.
- Neural-network objectives are non-convex, so useful generalization matters more than proving global optimality.
- Initialization, scaling, optimizer choice, and stopping criteria all affect training.

## Review questions

1. What is the difference between backpropagation and gradient descent?
2. Why should a brute-force parameter plot use \(w\) rather than \(\hat{y}\) on the horizontal axis?
3. How many configurations result from 1,000 candidates for each of 25 parameters?
4. How many weights and biases are present in a fully connected 4–5–1 network?
5. In one dimension, what does the sign of \(dJ/dw\) tell us?
6. Why does gradient descent subtract the gradient?
7. What role does the learning rate play?
8. Why can a gradient-descent path zigzag?
9. How do full-batch, stochastic, and mini-batch gradients differ?
10. Why is a stationary point not necessarily a minimum?
11. Why is the global minimum of training loss not the only goal?
12. What is finite-difference gradient checking used for?

## Further reading

- [TOP500: June 2016 ranking led by Sunway TaihuLight](https://www.top500.org/lists/top500/2016/06/)
- [TOP500: June 2026 ranking](https://www.top500.org/lists/top500/2026/06/)
