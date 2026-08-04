# How Neural Networks Learn

## Lesson objective

This lesson introduces supervised neural-network training. A model receives inputs, produces predictions, measures its errors with a loss function, computes gradients through backpropagation, and updates its parameters with an optimizer.

The central training loop is:

```text
inputs → predictions → loss → gradients → parameter update → repeat
```

## 1. Rule-based programming and machine learning

In traditional rule-based programming, a developer specifies explicit instructions:

```text
data + hand-written rules → output
```

In supervised machine learning, the developer supplies examples containing inputs and desired outputs:

```text
labeled data + model + learning objective → learned parameters
```

This contrast is useful, but the approaches are not completely separate. A machine-learning system still contains many human decisions and rules, including:

- how data is collected and labeled;
- which inputs are included;
- the architecture of the model;
- the loss function;
- optimization settings;
- constraints and regularization;
- evaluation criteria; and
- deployment safeguards.

Real systems often combine learned components with explicit rules. A neural network does not independently determine the goal; people define the task and provide the objective and data from which it learns.

## 2. The cat-versus-dog example

A traditional image classifier might rely on manually designed rules concerning ears, noses, whiskers, shapes, or colors. Such rules are difficult to enumerate and may fail when animals appear at unusual angles, under different lighting, or with substantial variation.

A supervised neural-network approach uses labeled examples:

```text
cat images labeled “cat”
dog images labeled “dog”
```

During training, the network adjusts its parameters so that its predictions better match these labels. It can learn useful visual features rather than requiring a programmer to specify every visual rule.

However, the network does not simply “understand everything it needs” after being pointed at a folder. Successful training depends on:

- accurate and representative labels;
- enough variation in the examples;
- a suitable model and training procedure;
- separate validation and test data;
- protection against leakage and duplicated images; and
- evaluation under the conditions expected after deployment.

A model can learn shortcuts—for example, backgrounds, image watermarks, or camera artifacts—instead of animal anatomy. Generalization must be measured rather than assumed.

## 3. The perceptron

Frank Rosenblatt described the perceptron in 1957, and his major journal paper, *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*, appeared in 1958.

In modern introductory notation, a perceptron is a single linear binary classifier:

\[
z = \mathbf{w}^{\mathsf{T}}\mathbf{x} + b
\]

\[
\hat{y} =
\begin{cases}
1, & z \ge 0 \\
0, & z < 0
\end{cases}
\]

The term is sometimes used loosely for a single-layer network, but a classical perceptron is more specific than any arbitrary feedforward network with one layer.

Because its decision boundary is linear, a single perceptron cannot solve every binary problem. Multilayer networks with nonlinear activations are more expressive.

## 4. Actual values and predicted values

The standard notation is:

- \(y\): the true or observed target; and
- \(\hat{y}\), pronounced “y-hat”: the model’s prediction.

For student-exam prediction, one observation might be:

\[
\mathbf{x} =
\begin{bmatrix}
\text{study hours} \\
\text{sleep hours} \\
\text{midterm quiz score}
\end{bmatrix}
\]

with final exam score \(y=93\). The network maps \(\mathbf{x}\) to a prediction \(\hat{y}\), which is compared with 93.

For one sample:

```text
features x → network with parameters θ → prediction ŷ
                                           ↓
                                      compare with y
```

The symbol \(\theta\) is often used to represent all trainable parameters collectively.

## 5. Loss function versus cost function

A **loss function** measures error for one observation or one batch. A **cost** or **objective function** often aggregates losses across a dataset and may include regularization. Terminology varies across textbooks, so the intended definition should always be stated.

### Squared-error loss

For one regression observation, the lesson uses:

\[
\mathcal{L}(y,\hat{y}) = \frac{1}{2}(\hat{y}-y)^2
\]

The factor \(1/2\) is convenient because it cancels the 2 produced during differentiation:

\[
\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y}-y
\]

Squaring also makes positive and negative errors contribute positively and penalizes large errors more strongly.

For \(n\) observations, mean squared error can be written as:

\[
J(\theta) = \frac{1}{2n}\sum_{i=1}^{n}
\left(\hat{y}^{(i)}-y^{(i)}\right)^2
\]

Using a mean rather than a raw sum makes the scale of the objective less dependent on the dataset or batch size.

### Choosing a loss

Squared error is common for regression, but it is not the most appropriate loss for every neural-network task:

| Task | Common loss |
|---|---|
| Regression | Mean squared error, mean absolute error, Huber loss |
| Binary classification | Binary cross-entropy |
| Multiclass classification | Categorical cross-entropy |
| Count modeling | Poisson or another distribution-appropriate loss |

The loss should reflect the output representation, statistical assumptions, and practical cost of different errors.

## 6. What the network can adjust

The transcript focuses on weights, but a standard neural network normally learns both:

- **weights**, which scale connections; and
- **biases**, which shift pre-activations.

Depending on the architecture, it may also learn embeddings, normalization parameters, convolutional filters, attention projections, and other quantities. All trainable parameters affect the loss and can receive gradient-based updates.

## 7. Forward pass

For one observation, the network first produces a prediction using its current parameters:

\[
\hat{y} = f(\mathbf{x};\theta)
\]

This is the **forward pass**. The prediction is then passed to the loss function:

\[
\mathcal{L} = \mathcal{L}(y,\hat{y})
\]

The loss is a scalar summary of current prediction error. It tells us how well the model performed under the chosen objective, but the scalar value alone does not specify how each parameter should change.

## 8. Backpropagation

**Backpropagation** efficiently applies the chain rule to calculate the gradient of the loss with respect to every trainable parameter:

\[
\nabla_{\theta}J
\]

For a particular weight \(w_j\), backpropagation computes:

\[
\frac{\partial J}{\partial w_j}
\]

This derivative describes the local sensitivity of the objective to that parameter. It is not the new weight and does not update the network by itself.

The phrase “feeding the error backward” is useful intuition, but backpropagation passes gradient information through mathematical dependencies. It does not send the target value backward as if it were another input signal.

## 9. Optimization and parameter updates

An **optimizer** uses the gradients to update parameters. Basic gradient descent applies:

\[
w_j \leftarrow w_j - \eta\frac{\partial J}{\partial w_j}
\]

\[
b_j \leftarrow b_j - \eta\frac{\partial J}{\partial b_j}
\]

where \(\eta\) is the **learning rate**.

The learning rate controls the update size:

- too small may make learning extremely slow;
- too large may cause oscillation, divergence, or unstable training; and
- an appropriate value depends on the optimizer, model, data, and schedule.

Gradient descent, stochastic gradient descent, momentum, RMSProp, and Adam are optimization methods. Backpropagation computes gradients; the optimizer decides how to use them. Calling the entire training procedure “backpropagation” hides this important distinction.

## 10. Worked one-parameter update

Consider a simple linear model:

\[
\hat{y} = wx+b
\]

Let:

\[
x=2,\quad y=5,\quad w=1,\quad b=0
\]

### Step 1: Forward pass

\[
\hat{y} = (1)(2)+0=2
\]

### Step 2: Loss

\[
\mathcal{L}=\frac{1}{2}(2-5)^2=4.5
\]

### Step 3: Gradients

\[
\frac{\partial\mathcal{L}}{\partial w}
=(\hat{y}-y)x=(2-5)(2)=-6
\]

\[
\frac{\partial\mathcal{L}}{\partial b}
=\hat{y}-y=-3
\]

### Step 4: Update

With learning rate \(\eta=0.1\):

\[
w \leftarrow 1-(0.1)(-6)=1.6
\]

\[
b \leftarrow 0-(0.1)(-3)=0.3
\]

### Step 5: New prediction

\[
\hat{y}_{\text{new}}=(1.6)(2)+0.3=3.5
\]

\[
\mathcal{L}_{\text{new}}
=\frac{1}{2}(3.5-5)^2=1.125
\]

The loss decreased from 4.5 to 1.125 after one update. A real network applies the same principle to many parameters using automatic differentiation and vectorized computation.

## 11. Training repeatedly on one row

Repeatedly updating the model using one observation can reduce that observation’s loss, possibly to zero in a sufficiently flexible model. This demonstrates optimization but not useful learning.

A model trained on one row may simply memorize that example. A zero training loss does not show that it can predict new students, images, or properties correctly. Generalization requires representative data and evaluation on observations not used for fitting.

The target of training is therefore not merely:

```text
minimize training loss at any cost
```

It is closer to:

```text
learn patterns from training data that perform well on unseen data
```

## 12. Multiple observations share one model

Suppose the training set contains eight students. The diagram may display eight copies of a network to visualize eight forward passes, but there is only one shared parameter set:

\[
\theta = \{\mathbf{W}^{(1)},\mathbf{b}^{(1)},\ldots\}
\]

Every row is processed with the same weights and biases. There is not a separate model for every observation.

This parameter sharing is essential: the network learns one function intended to apply across examples.

## 13. Batch, stochastic, and mini-batch training

There are several ways to organize parameter updates.

### Batch gradient descent

Use the complete training set to calculate one gradient and then update the parameters:

```text
all training rows → aggregate objective → one update
```

The eight-row illustration in the transcript most closely describes this approach.

### Stochastic gradient descent

Use one observation for each update:

```text
one row → loss → one update
```

This produces noisy but frequent gradient estimates.

### Mini-batch gradient descent

Use a small group of observations per update:

```text
one mini-batch → average batch loss → one update
```

Mini-batches are the standard practical compromise. They support efficient parallel computation while producing more stable gradients than single-example updates.

## 14. Epochs, batches, and steps

An **epoch** is one complete pass through the training dataset under the current sampling convention.

If the training set has \(n\) observations and the batch size is \(B\), the number of optimizer steps per ordinary epoch is approximately:

\[
\left\lceil\frac{n}{B}\right\rceil
\]

For 8 rows:

- batch size 8 gives 1 update per epoch;
- batch size 4 gives 2 updates per epoch;
- batch size 1 gives 8 updates per epoch.

An **iteration** or **training step** usually means one optimizer update, not necessarily one full pass through the dataset.

Training data is commonly shuffled between epochs so that mini-batch composition changes. Time-series and other ordered data may require different handling.

## 15. A complete training loop

Conceptually, a supervised training loop is:

```python
initialize_parameters()

for epoch in range(number_of_epochs):
    shuffle_training_data_if_appropriate()

    for X_batch, y_batch in training_batches:
        y_pred = model(X_batch)          # forward pass
        loss = loss_function(y_batch, y_pred)
        gradients = backpropagate(loss) # gradient computation
        optimizer_step(gradients)       # parameter update

    evaluate_on_validation_data()
```

Modern frameworks perform the forward pass, automatic differentiation, and optimizer update through library APIs, but the conceptual stages remain the same.

## 16. Training, validation, and test sets

Data is normally divided by purpose:

- **training set:** fits weights and biases;
- **validation set:** selects hyperparameters, thresholds, architecture, and stopping point; and
- **test set:** provides a final estimate after model development decisions are complete.

Repeatedly checking test performance and changing the model in response turns the test set into another validation set and makes the reported result optimistic.

Preprocessing transformations must also be fitted using training data only and reused unchanged on validation, test, and deployment inputs.

## 17. Why a minimum is not the whole goal

Neural-network objectives are typically high-dimensional and non-convex. Training is not guaranteed to locate a unique global minimum, and it often stops at a useful parameter region rather than a mathematically exact minimum.

Even a global minimum of training loss would not automatically be the best model for unseen data. The model may overfit. Practical stopping decisions use evidence such as:

- validation loss;
- validation metrics relevant to the task;
- early stopping;
- regularization;
- learning-rate schedules; and
- computational constraints.

The “optimal weights for the training dataset” are not necessarily optimal for the real population or future inputs.

## 18. Loss curves and diagnostics

Track both training and validation behavior across epochs:

| Pattern | Possible interpretation |
|---|---|
| Both losses remain high | Underfitting, optimization problems, weak features, or data issues |
| Training loss falls; validation loss also falls | Learning is improving generalization |
| Training loss falls; validation loss rises | Possible overfitting |
| Loss becomes unstable or diverges | Learning rate may be too high, gradients may be unstable, or data may be problematic |
| Loss plateaus early | Learning rate, architecture, activation, scaling, or model capacity may need attention |

Loss curves are diagnostic evidence, not complete proof of model quality. Metrics, error analysis, subgroup checks, and data review are also required.

## Common pitfalls

- Describing rule-based programming and neural networks as the only two possible approaches.
- Saying a network learns entirely on its own without a human-defined objective or data pipeline.
- Assuming labeled images guarantee that the model learns the intended visual features.
- Calling every single-layer feedforward network a classical perceptron.
- Confusing \(y\) with \(\hat{y}\).
- Calling half squared error the most common loss for every task.
- Treating weights as the only trainable parameters and omitting biases.
- Saying the loss value itself directly tells every weight how to change.
- Treating backpropagation and gradient descent as synonyms.
- Assuming one epoch always means one parameter update.
- Giving each training row a separate set of network weights.
- Believing zero training loss proves good generalization.
- Assuming training must locate the global minimum.
- Tuning the model using test-set results.

## Key takeaways

- Supervised learning uses labeled examples and a human-defined objective to learn parameters.
- \(y\) is the true target; \(\hat{y}\) is the prediction.
- A loss function measures prediction error under a chosen objective.
- Backpropagation computes gradients through the network using the chain rule.
- An optimizer uses those gradients to update weights, biases, and other trainable parameters.
- All observations share the same model parameters.
- Batch, stochastic, and mini-batch training differ in how many observations contribute to each update.
- One epoch is a pass through the training data; one step is one optimizer update.
- Low training loss is not enough—validation and test performance measure generalization.
- The correct loss depends on the prediction task and output representation.

## Review questions

1. How does supervised machine learning differ from purely rule-based programming?
2. Why does a cat-versus-dog model still depend on human design decisions?
3. What is the distinction between a classical perceptron and a generic neural network?
4. What do \(y\) and \(\hat{y}\) represent?
5. Why is the factor \(1/2\) convenient in squared-error loss?
6. What is the difference between a per-example loss and a dataset objective?
7. What does backpropagation compute?
8. What does the optimizer do with a gradient?
9. How do batch, stochastic, and mini-batch gradient descent differ?
10. If a dataset contains 1,000 observations and the batch size is 100, how many ordinary optimizer steps occur per epoch?
11. Why can repeatedly training on one row produce a misleading result?
12. Why is the minimum training loss not necessarily the best stopping point?
13. What roles do the training, validation, and test sets play?

## Further reading

- [Smithsonian National Museum of American History: Mark I Perceptron](https://americanhistory.si.edu/collections/object/nmah_334414)
- [Rosenblatt (1958): *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*](https://doi.org/10.1037/h0042519)
- [Cross Validated community discussion: loss functions used in neural networks](https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications)
