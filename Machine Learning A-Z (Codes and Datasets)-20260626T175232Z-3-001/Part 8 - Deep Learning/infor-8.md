# Backpropagation: Completing the Training Loop

## Lesson objective

Backpropagation efficiently computes how a neural network’s loss changes with respect to every differentiable trainable parameter. An optimizer then uses those gradients to update the parameters.

The complete learning cycle is:

```text
initialize parameters once
        ↓
forward pass
        ↓
compute loss
        ↓
backpropagate gradients
        ↓
optimizer update
        ↓
repeat over batches and epochs
```

## 1. Forward propagation

During a forward pass, the input moves through the network one layer at a time.

Let:

\[
\mathbf{a}^{(0)}=\mathbf{x}
\]

For layer \(l\):

\[
\mathbf{z}^{(l)}
=
\mathbf{W}^{(l)}\mathbf{a}^{(l-1)}
+
\mathbf{b}^{(l)}
\]

\[
\mathbf{a}^{(l)}
=
\phi^{(l)}\left(\mathbf{z}^{(l)}\right)
\]

where:

- \(\mathbf{W}^{(l)}\) is the layer’s weight matrix;
- \(\mathbf{b}^{(l)}\) is its bias vector;
- \(\mathbf{z}^{(l)}\) contains pre-activations;
- \(\phi^{(l)}\) is the activation function; and
- \(\mathbf{a}^{(l)}\) contains output activations.

At the final layer \(L\), the network produces \(\hat{\mathbf{y}}\). The loss compares this prediction with the true target \(\mathbf{y}\):

\[
\mathcal{L}
=
\mathcal{L}\left(\mathbf{y},\hat{\mathbf{y}}\right)
\]

The forward pass stores intermediate values needed for gradient calculation.

## 2. What backpropagation actually does

Backpropagation applies the chain rule from the loss back through the sequence of operations that produced it. It computes quantities such as:

\[
\frac{\partial\mathcal{L}}{\partial\mathbf{W}^{(l)}}
\qquad\text{and}\qquad
\frac{\partial\mathcal{L}}{\partial\mathbf{b}^{(l)}}
\]

for every differentiable layer.

It is common to say that “the error is propagated backward,” but the precise objects being propagated are derivative signals. These derivatives describe how sensitive the loss is to intermediate values and parameters.

Backpropagation does **not** directly update weights. It computes gradients. The optimizer performs the update afterward.

## 3. Backpropagation and the chain rule

Suppose a scalar loss depends on a weight through several intermediate operations:

\[
w \rightarrow z \rightarrow a \rightarrow \hat{y} \rightarrow \mathcal{L}
\]

The chain rule gives:

\[
\frac{\partial\mathcal{L}}{\partial w}
=
\frac{\partial\mathcal{L}}{\partial\hat{y}}
\frac{\partial\hat{y}}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w}
\]

Backpropagation reuses intermediate derivative calculations. This reuse is what makes it far more efficient than independently recomputing the full derivative path for every parameter.

## 4. Layerwise backpropagation equations

Define the layer error signal:

\[
\boldsymbol{\delta}^{(l)}
=
\frac{\partial\mathcal{L}}{\partial\mathbf{z}^{(l)}}
\]

### Output layer

At the final layer:

\[
\boldsymbol{\delta}^{(L)}
=
\nabla_{\mathbf{a}^{(L)}}\mathcal{L}
\odot
\phi^{(L)\prime}\left(\mathbf{z}^{(L)}\right)
\]

where \(\odot\) denotes elementwise multiplication.

For certain matched activation-and-loss combinations, such as softmax with cross-entropy, this expression simplifies.

### Hidden layer

For an earlier layer \(l\):

\[
\boldsymbol{\delta}^{(l)}
=
\left(
\mathbf{W}^{(l+1)\mathsf{T}}
\boldsymbol{\delta}^{(l+1)}
\right)
\odot
\phi^{(l)\prime}\left(\mathbf{z}^{(l)}\right)
\]

This equation transfers the downstream sensitivity through the next layer’s weights and the current layer’s activation derivative.

### Parameter gradients

Once \(\boldsymbol{\delta}^{(l)}\) is known:

\[
\frac{\partial\mathcal{L}}{\partial\mathbf{W}^{(l)}}
=
\boldsymbol{\delta}^{(l)}
\mathbf{a}^{(l-1)\mathsf{T}}
\]

\[
\frac{\partial\mathcal{L}}{\partial\mathbf{b}^{(l)}}
=
\boldsymbol{\delta}^{(l)}
\]

For a mini-batch, gradients are summed or averaged across the batch according to the loss definition.

## 5. “All weights at the same time”

The important efficiency claim is that one reverse traversal can calculate gradients for all differentiable parameters while reusing shared intermediate results.

Mathematically, the usual optimizer step treats gradients as having been evaluated at the same parameter state:

\[
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_t
-
\eta
\nabla_{\boldsymbol{\theta}}
J(\boldsymbol{\theta}_t)
\]

This is often described as updating all parameters simultaneously. An implementation may execute operations in parallel, fused kernels, or a particular sequence, but the conceptual update uses one consistent gradient snapshot.

Backpropagation does not assign a simple percentage of “blame” to each weight. A gradient is a local derivative, not a causal responsibility score. Parameters also interact, so changing several together can have effects not captured by treating each in isolation.

## 6. Backpropagation and automatic differentiation

Modern frameworks construct a computational graph during the forward pass and use **reverse-mode automatic differentiation** to evaluate gradients efficiently.

This is closely related to backpropagation:

- backpropagation is reverse-mode differentiation specialized to layered computational structures;
- automatic differentiation handles a broader graph of differentiable operations; and
- the framework tracks dependencies so users do not manually derive every partial derivative.

Automatic differentiation is not symbolic algebra and is not finite-difference approximation. It evaluates derivatives through elementary operations using chain-rule composition.

## 7. Historical context

The mathematical foundations of reverse-mode differentiation and related gradient calculations predate the 1980s, and multiple researchers contributed to their development and application.

The influential 1986 paper **“Learning representations by back-propagating errors”** by **David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams** played a major role in popularizing backpropagation for training multilayer networks with hidden units.

It is therefore reasonable to describe the 1980s as a period when backpropagation became highly influential, but not to claim that the underlying idea appeared for the first time then.

## 8. Step 1—initialize parameters once

Weights need an initial value before training starts. They are usually initialized randomly using a scale chosen for the architecture and activation function.

Common principles include:

- avoid setting every hidden-unit weight to the same value, because identical units can remain symmetric;
- use Glorot/Xavier-style initialization for some sigmoid or tanh networks;
- use He-style initialization for many ReLU networks; and
- initialize biases to zero or small values when appropriate.

The instruction “small random values close to zero” is a useful first intuition but is not a complete initialization rule. Values that are too small can shrink activations and gradients; values that are too large can cause saturation or instability.

Most importantly, initialization happens **once before the training loop**. Reinitializing the weights for every observation or epoch would erase what the network had learned.

Exceptions include deliberate restarts, ensembling independently initialized models, or fine-tuning a pretrained parameter set.

## 9. Step 2—prepare a mini-batch

Select a batch containing input rows and their targets:

\[
\mathbf{X}_B,\mathbf{Y}_B
\]

Each row represents one observation, and each feature column must match the order and preprocessing expected by the model.

In practice, this stage may include:

- shuffling or structured sampling;
- loading data from storage;
- applying training-only augmentation;
- transforming inputs with training-fitted preprocessing; and
- moving arrays to the compute device.

The batch size determines whether the method is strict SGD, mini-batch training, or full-batch training.

## 10. Step 3—run the forward pass

Propagate the batch from the input layer to the output:

\[
\hat{\mathbf{Y}}_B
=
f(\mathbf{X}_B;\boldsymbol{\theta})
\]

Weights do not merely “limit” activations. Their sign and magnitude determine how earlier activations contribute to later pre-activations. A negative weight can suppress or reverse a contribution; a positive weight can reinforce it.

Intermediate activations and pre-activations may be cached for the backward pass.

## 11. Step 4—compute the batch loss

Compare predictions and targets:

\[
J_B
=
\frac{1}{|B|}
\sum_{i\in B}
\mathcal{L}
\left(
\mathbf{y}^{(i)},
\hat{\mathbf{y}}^{(i)}
\right)
\]

The loss must match the task and output design. Examples include:

- mean squared error for regression;
- binary cross-entropy for binary or multilabel classification; and
- categorical cross-entropy for mutually exclusive multiclass classification.

Regularization penalties may be added to form the complete training objective.

## 12. Step 5—run the backward pass

Start from the scalar loss and traverse the graph in reverse to compute:

\[
\nabla_{\boldsymbol{\theta}}J_B
\]

Frameworks commonly accumulate gradients into parameter-specific buffers. These buffers must be cleared at the correct time unless deliberate gradient accumulation is being used.

Conceptual sequence:

```text
loss derivative
      ↓
output-layer gradients
      ↓
hidden-layer gradients
      ↓
input-side gradients when required
```

The backward pass follows dependencies in reverse; it does not require physically copying the network or training each layer independently.

## 13. Step 6—update the parameters

An optimizer uses the computed gradients. Basic gradient descent applies:

\[
\boldsymbol{\theta}
\leftarrow
\boldsymbol{\theta}
-
\eta
\nabla_{\boldsymbol{\theta}}J_B
\]

The learning rate \(\eta\) controls update scale. Optimizers such as momentum SGD, RMSProp, or Adam modify the update using gradient history and other state.

The correct sequence is:

```text
compute all required gradients using current parameters
then perform the optimizer update
```

Updating an upper-layer weight and then using that new value to calculate a lower-layer gradient would generally be inconsistent with ordinary backpropagation unless a specialized algorithm explicitly defines that behavior.

## 14. Worked two-layer example

Consider a one-input, one-hidden-unit regression network:

\[
z_1=w_1x+b_1
\]

\[
h=\operatorname{ReLU}(z_1)
\]

\[
\hat{y}=w_2h+b_2
\]

\[
\mathcal{L}=\frac{1}{2}(\hat{y}-y)^2
\]

Let:

\[
x=2,\quad y=1
\]

\[
w_1=0.5,\quad b_1=0,\quad w_2=0.4,\quad b_2=0
\]

### Forward pass

\[
z_1=(0.5)(2)+0=1
\]

\[
h=\operatorname{ReLU}(1)=1
\]

\[
\hat{y}=(0.4)(1)+0=0.4
\]

\[
\mathcal{L}
=
\frac{1}{2}(0.4-1)^2
=
0.18
\]

### Output-layer gradients

Because the output is linear:

\[
\delta_2
=
\frac{\partial\mathcal{L}}{\partial\hat{y}}
=
\hat{y}-y
=
-0.6
\]

\[
\frac{\partial\mathcal{L}}{\partial w_2}
=
\delta_2h
=
(-0.6)(1)
=
-0.6
\]

\[
\frac{\partial\mathcal{L}}{\partial b_2}
=
\delta_2
=
-0.6
\]

### Hidden-layer gradients

Because \(z_1=1>0\), the ReLU derivative is 1:

\[
\delta_1
=
w_2\delta_2\operatorname{ReLU}'(z_1)
=
(0.4)(-0.6)(1)
=
-0.24
\]

\[
\frac{\partial\mathcal{L}}{\partial w_1}
=
\delta_1x
=
(-0.24)(2)
=
-0.48
\]

\[
\frac{\partial\mathcal{L}}{\partial b_1}
=
\delta_1
=
-0.24
\]

### Simultaneous conceptual update

With learning rate \(\eta=0.1\):

\[
w_2\leftarrow0.4-(0.1)(-0.6)=0.46
\]

\[
b_2\leftarrow0-(0.1)(-0.6)=0.06
\]

\[
w_1\leftarrow0.5-(0.1)(-0.48)=0.548
\]

\[
b_1\leftarrow0-(0.1)(-0.24)=0.024
\]

All four gradients were calculated using the original parameter state before the update.

With the updated parameters:

\[
z_{1,\text{new}}
=(0.548)(2)+0.024
=1.12
\]

\[
\hat{y}_{\text{new}}
=(0.46)(1.12)+0.06
=0.5752
\]

The new loss is approximately:

\[
\frac{1}{2}(0.5752-1)^2
\approx0.0902
\]

It decreased from 0.18 after one gradient step.

## 15. Step 7—repeat over batches and epochs

After an update, process the next batch. When the training procedure has made one complete pass through the training set, it has completed one **epoch**.

The next epoch uses the parameters learned so far. It does not restart from the initialization.

For \(n\) observations and batch size \(B\), an ordinary epoch contains approximately:

\[
\left\lceil\frac{n}{B}\right\rceil
\]

updates.

More epochs do not automatically produce a better model. Training loss may continue to decrease after validation performance begins to deteriorate.

## 16. Correct training terminology

Updating after every observation in a labeled prediction problem is **stochastic gradient descent**, **incremental learning**, or sometimes **online supervised learning**, depending on the data-arrival setting.

It is not reinforcement learning.

Reinforcement learning involves an agent interacting with an environment, taking actions, receiving rewards, and optimizing expected return. A labeled row with a known target belongs to supervised learning.

The gradient variants are:

| Batch size | Common term |
|---:|---|
| 1 | Strict stochastic gradient descent |
| Between 1 and \(n\) | Mini-batch gradient descent |
| \(n\) | Full-batch gradient descent |

The phrase “batch learning” can be ambiguous, so state the actual batch size or update rule.

## 17. A corrected end-to-end training algorithm

```python
# Split data and fit preprocessing using training data only.
# Build the model.
initialize_parameters_once()

for epoch in range(max_epochs):
    model.train_mode()
    shuffle_training_data_if_appropriate()

    for X_batch, y_batch in training_batches:
        clear_old_gradients()

        y_pred = model(X_batch)              # forward pass
        loss = loss_function(y_pred, y_batch)
        loss.backward()                      # backpropagation
        optimizer.step()                     # parameter update

    model.evaluation_mode()
    validation_metrics = evaluate(validation_data)
    apply_early_stopping_if_needed()

final_metrics = evaluate_once(test_data)
```

The names are framework-neutral pseudocode; real APIs differ.

## 18. Why old gradients may need clearing

Many automatic-differentiation frameworks accumulate gradients by default:

\[
\mathbf{g}_{\text{buffer}}
\leftarrow
\mathbf{g}_{\text{buffer}}
+
\mathbf{g}_{\text{new}}
\]

If the intent is one optimizer step per mini-batch, old gradients must be cleared before or after the appropriate update.

Deliberate **gradient accumulation** sums gradients across several smaller micro-batches before one optimizer step. This can simulate a larger effective batch when memory is limited, provided loss scaling and normalization are handled correctly.

## 19. Vanishing and exploding gradients

Backpropagation multiplies many local derivatives. In deep or recurrent networks, these products can become:

- extremely small, producing **vanishing gradients**; or
- extremely large, producing **exploding gradients**.

Possible mitigations include:

- appropriate activations;
- Glorot/Xavier or He initialization;
- normalization layers;
- residual connections;
- gated recurrent architectures;
- gradient clipping; and
- careful learning-rate selection.

Backpropagation may be mathematically correct while optimization still fails because the resulting gradients are numerically unhelpful.

## 20. Nondifferentiable points

Some common operations, including ReLU at zero, are not differentiable at every point. Frameworks select a valid practical convention or subgradient for such cases.

Truly discrete operations, hard decisions, sampling steps, or non-differentiable external systems may block ordinary gradient flow. Specialized estimators, relaxations, surrogate gradients, or different optimization methods may then be required.

## 21. Gradient checking

For a small model, an analytical gradient can be compared with a finite-difference estimate:

\[
\frac{\partial J}{\partial\theta_j}
\approx
\frac{
J(\theta_j+\varepsilon)
-
J(\theta_j-\varepsilon)
}{2\varepsilon}
\]

This is useful for debugging a manual backward implementation. It is too expensive for normal training and can be sensitive to floating-point precision and \(\varepsilon\).

## 22. When training should stop

Do not train merely until the training loss appears minimized. Use evidence from validation data and the problem’s requirements.

Stopping criteria may include:

- validation loss no longer improving;
- a task metric reaching an acceptable level;
- early-stopping patience being exhausted;
- a maximum epoch or compute budget;
- learning rate becoming very small; or
- signs of overfitting or instability.

The final test set should be used after model-selection decisions are complete.

## Common pitfalls

- Saying that backpropagation itself changes the weights.
- Describing a gradient as a literal share of causal blame.
- Omitting bias gradients.
- Updating one layer before calculating gradients for earlier layers with the original parameter state.
- Reinitializing weights inside the observation or epoch loop.
- Assuming every weight should merely be a tiny random number regardless of activation or layer width.
- Setting all hidden-layer weights to zero and expecting units to learn different features.
- Saying biases must never be initialized to zero.
- Repeating “steps one to five” when step one is initialization.
- Calling per-observation supervised training reinforcement learning.
- Assuming more epochs always improve validation performance.
- Forgetting to clear accumulated gradients.
- Confusing backpropagation with finite differences or symbolic differentiation.
- Claiming the method was invented entirely in 1986 without acknowledging earlier foundations.

## Key takeaways

- Forward propagation computes activations, predictions, and loss.
- Backpropagation applies the chain rule to compute gradients for every differentiable trainable parameter.
- The optimizer—not backpropagation—uses those gradients to update parameters.
- One reverse pass efficiently reuses intermediate derivatives across the network.
- Gradients are local sensitivities, not causal responsibility percentages.
- Weights are initialized once before training and retained across batches and epochs.
- Initialization scale should match the architecture and activation function.
- Strict SGD, mini-batch descent, and full-batch descent differ by the number of examples per update.
- Per-example supervised updates are not reinforcement learning.
- One epoch is a pass through the training data; more epochs can eventually overfit.
- Automatic differentiation implements reverse derivative propagation through computational graphs.
- Validation evidence should determine stopping and model selection.

## Review questions

1. What is stored during the forward pass for use during backpropagation?
2. What does \(\boldsymbol{\delta}^{(l)}\) represent?
3. How is a hidden layer’s error signal calculated from the next layer?
4. Write the equations for the weight and bias gradients of one dense layer.
5. In what sense are all parameter gradients computed “at the same time”?
6. Why is a gradient not a causal blame percentage?
7. What is the difference between backpropagation and an optimizer step?
8. Why should hidden-layer weights not all be initialized identically?
9. Why must initialization occur outside the epoch loop?
10. Why is one-example supervised learning not reinforcement learning?
11. What happens if gradients are accumulated unintentionally?
12. What causes vanishing or exploding gradients?
13. Why can additional epochs reduce model quality on unseen data?
14. What role did the influential 1986 paper play in backpropagation’s history?

## Further reading

- [Rumelhart, Hinton, and Williams (1986): *Learning representations by back-propagating errors*](https://doi.org/10.1038/323533a0)
- [Michael Nielsen: *How the backpropagation algorithm works*](https://michaelnielsen.org/ddi/how-the-backpropagation-algorithm-works/)
