# Activation Functions

## Lesson objective

An artificial neuron first calculates a weighted sum plus a bias, then applies an activation function. This lesson introduces four classic activation functions—the threshold function, sigmoid, rectified linear unit, and hyperbolic tangent—and explains when each is useful.

We will also connect the choice of output activation to the prediction task and loss function. This is essential because an activation function does more than determine what value moves to the next layer: it shapes the model’s representation, gradients, and output interpretation.

## 1. Where the activation function fits

For an input vector \(\mathbf{x}\), weights \(\mathbf{w}\), and bias \(b\), a neuron computes:

\[
z = \mathbf{w}^{\mathsf{T}}\mathbf{x} + b
\]

It then applies an activation function \(\phi\):

\[
a = \phi(z)
\]

where:

- \(z\) is the **pre-activation** or weighted sum plus bias;
- \(\phi\) is the activation function; and
- \(a\) is the output activation passed to the next layer or used for prediction.

In a complete layer, the same operation is performed for many units and observations:

\[
\mathbf{Z} = \mathbf{XW} + \mathbf{b}
\]

\[
\mathbf{A} = \phi(\mathbf{Z})
\]

## 2. Why activation functions matter

Without nonlinear activations, a stack of dense layers remains equivalent to a single linear transformation. For example, composing two linear transformations still produces another linear transformation:

\[
\mathbf{W}_2(\mathbf{W}_1\mathbf{x}) = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x}
\]

Nonlinear activation functions allow multilayer networks to model nonlinear relationships and construct more expressive decision boundaries.

An activation function also affects:

- the range of values passed through the network;
- the flow of gradients during backpropagation;
- training speed and stability;
- whether units produce sparse or dense activations; and
- how the final output can be interpreted.

## 3. Threshold or step function

A common binary threshold function is:

\[
\phi(z) =
\begin{cases}
0, & z < 0 \\
1, & z \ge 0
\end{cases}
\]

This creates a rigid yes-or-no output. If the pre-activation is below the threshold, the result is 0; otherwise, it is 1.

The boundary convention is not universal. Some definitions use \(z > 0\) for the positive case or place the threshold somewhere other than zero. The exact definition must therefore be stated.

### Strengths

- Easy to interpret.
- Produces a direct binary decision.
- Historically important in early neuron and perceptron models.

### Limitations

- It is discontinuous at the threshold.
- Its derivative is zero almost everywhere and undefined at the threshold.
- Standard gradient-based backpropagation cannot obtain useful gradients through it.
- It gives no calibrated measure of confidence.

For these reasons, a hard threshold is normally used as a **decision rule after training**, not as the trainable output activation of a modern network optimized with backpropagation.

## 4. Sigmoid function

The logistic sigmoid function is:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Its output lies strictly between 0 and 1:

\[
0 < \sigma(z) < 1
\]

Important values include:

\[
\sigma(0) = 0.5
\]

Large positive inputs approach 1, while large negative inputs approach 0. The function is smooth and differentiable, with derivative:

\[
\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)
\]

### Typical use

Sigmoid is commonly used in the output layer for binary classification. Its output can be interpreted as an estimated probability when the model, loss, data, and calibration support that interpretation:

\[
\hat{p}(y=1\mid\mathbf{x}) = \sigma(z)
\]

A decision threshold can then convert the score into a class label:

```text
predict 1 if probability ≥ 0.5; otherwise predict 0
```

The value `0.5` is conventional, not mandatory. A different threshold may be appropriate when false positives and false negatives have different costs.

### Limitations

- For large positive or negative inputs, sigmoid saturates near 1 or 0.
- Its gradient becomes very small in those saturated regions.
- Repeated sigmoid hidden layers can contribute to vanishing gradients.
- Its outputs are not zero-centered.

Sigmoid remains valuable at binary and multilabel output layers, but it is less common as the default activation for deep hidden layers.

## 5. Rectified Linear Unit

The function called the **rectifier** in the lecture is now usually called the **Rectified Linear Unit**, or **ReLU**:

\[
\operatorname{ReLU}(z) = \max(0, z)
\]

Equivalently:

\[
\operatorname{ReLU}(z) =
\begin{cases}
0, & z < 0 \\
z, & z \ge 0
\end{cases}
\]

For negative inputs, ReLU returns 0. For positive inputs, it returns the input unchanged.

Its derivative is:

\[
\operatorname{ReLU}'(z) =
\begin{cases}
0, & z < 0 \\
1, & z > 0
\end{cases}
\]

The mathematical derivative is undefined at exactly zero. Software libraries choose a practical subgradient convention there, commonly 0.

### Why ReLU became popular

- It is inexpensive to compute.
- Its positive side does not saturate like sigmoid or tanh.
- Gradients can flow well through active units.
- Exact zeros create sparse activations.
- It often works effectively as a default hidden-layer activation.

### Limitations

- A unit can enter a state where it always receives negative inputs and therefore always outputs zero. This is known as a **dying ReLU**.
- Its output is unbounded on the positive side.
- It is not differentiable at zero, although this rarely prevents practical optimization.
- It is not automatically optimal for every architecture or dataset.

Related variants include Leaky ReLU, PReLU, ELU, GELU, and SiLU/Swish. They address different optimization or representation needs.

## 6. Hyperbolic tangent

The hyperbolic tangent activation is:

\[
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
\]

Its range is:

\[
-1 < \tanh(z) < 1
\]

The function approaches -1 for large negative inputs, passes through 0 at \(z=0\), and approaches 1 for large positive inputs. Its derivative is:

\[
\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)
\]

### Strengths

- Smooth and differentiable.
- Zero-centered output.
- Useful in some recurrent-network and bounded-state settings.

### Limitations

- Saturates for inputs with large magnitude.
- Can produce small gradients and contribute to vanishing-gradient problems.

The lecture’s phrase “from zero to one and zero to minus one” is better expressed as a continuous range from approximately -1 to 1.

## 7. Comparing the four functions

| Function | Formula | Output range | Smooth? | Common role |
|---|---|---|---|---|
| Threshold | \(\mathbb{1}[z \ge 0]\) | `{0, 1}` | No | Final decision rule; historical models |
| Sigmoid | \(1/(1+e^{-z})\) | `(0, 1)` | Yes | Binary or multilabel outputs |
| ReLU | \(\max(0,z)\) | `[0, ∞)` | Piecewise; kink at 0 | Hidden layers |
| Tanh | \(\tanh(z)\) | `(-1, 1)` | Yes | Some hidden or recurrent states |

No activation is universally best. The appropriate choice depends on its location in the network, the prediction task, the loss function, and empirical results.

## 8. Additional output activations

The four functions above are important, but modern neural networks also rely on other output designs.

### Linear activation for regression

For unrestricted continuous regression, the final unit commonly uses:

\[
\phi(z) = z
\]

This permits predictions across the real-number line. A bounded or strictly positive target may require a different activation or target transformation.

### Softmax for mutually exclusive classes

For \(K\) mutually exclusive classes, softmax transforms a vector of logits into values that sum to 1:

\[
\operatorname{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
\]

The result is one score per class, commonly interpreted as a categorical probability distribution.

### Independent sigmoids for multilabel classification

If several labels may simultaneously be true, use one sigmoid output per label rather than softmax. Softmax makes classes compete to sum to 1, whereas independent sigmoids allow multiple positive labels.

## 9. Matching output activations and losses

The output activation should be chosen together with the loss:

| Task | Typical output | Typical loss |
|---|---|---|
| Unrestricted regression | Linear | Mean squared error or another regression loss |
| Binary classification | One sigmoid | Binary cross-entropy |
| Mutually exclusive multiclass classification | Softmax over classes | Categorical cross-entropy |
| Multilabel classification | One sigmoid per label | Binary cross-entropy per label |

Many frameworks provide numerically stable loss functions that accept raw logits and internally combine the activation with cross-entropy. For example, a “binary cross-entropy with logits” loss should receive raw logits, not values that have already passed through sigmoid.

This fusion avoids unstable calculations involving probabilities extremely close to 0 or 1.

## 10. The binary-classification exercise

The lecture asks which activation can support a binary target.

### Threshold answer

A threshold function directly returns 0 or 1, so it can represent a binary decision. However, its lack of useful gradients makes it unsuitable as the normal trainable output activation in a backpropagation-based network.

### Sigmoid answer

A sigmoid output provides a smooth score between 0 and 1 and can be trained with binary cross-entropy. After training, a decision threshold converts that score into 0 or 1.

Therefore, the standard modern approach is:

```text
training: raw logit → sigmoid-compatible binary cross-entropy
inference: estimated probability → chosen decision threshold → class label
```

Sigmoid is generally preferable to a hard threshold during gradient-based training.

## 11. A common feedforward configuration

For a basic binary classifier, a common architecture is:

```text
input features
    ↓
dense hidden layer + ReLU
    ↓
one output logit + sigmoid interpretation
    ↓
estimated probability of class 1
```

In equations:

\[
\mathbf{h} = \operatorname{ReLU}(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1)
\]

\[
\hat{p} = \sigma(\mathbf{w}_2^{\mathsf{T}}\mathbf{h} + b_2)
\]

This ReLU-hidden/sigmoid-output pattern is common, but it is not mandatory. Architecture and activation choices should be validated for the actual problem.

## 12. How activation functions affect gradients

Backpropagation repeatedly multiplies gradients by local derivatives. This makes activation derivatives central to learning:

- a derivative near zero weakens the gradient flowing to earlier layers;
- a stable nonzero derivative can help information propagate backward;
- very large derivatives can contribute to unstable gradients; and
- an inactive ReLU has derivative zero on its negative side.

Activation choice interacts with weight initialization, normalization, architecture depth, optimizer settings, and residual connections. It should not be considered in isolation.

## 13. The 2011 rectifier paper

The recommended paper is **“Deep Sparse Rectifier Neural Networks”** by **Xavier Glorot, Antoine Bordes, and Yoshua Bengio**, published at AISTATS in 2011.

The paper helped demonstrate that rectifying units could train deep supervised networks effectively, create truly zero-valued sparse activations, and perform competitively with tanh networks in the experiments studied. It was influential in the broader adoption of rectifier activations.

Its findings are important historical evidence, not proof that ReLU is the best activation for every modern model. Later work has introduced many alternatives and architecture-specific choices.

## Common pitfalls

- Forgetting the bias term when describing the pre-activation.
- Saying an activation merely decides whether a signal passes; most activations produce graded values.
- Assuming a sigmoid output is automatically a well-calibrated probability.
- Treating 0.5 as the only valid binary decision threshold.
- Using a hard threshold inside a network trained by ordinary backpropagation.
- Saying ReLU is differentiable everywhere; it has a kink at zero.
- Assuming the kink at zero makes ReLU unusable in practice.
- Saying tanh ranges from 0 to 1; its range is approximately -1 to 1.
- Applying sigmoid before a loss function that already expects raw logits.
- Using softmax for a multilabel task where several labels may be true.
- Assuming ReLU is always the best hidden activation.
- Attributing the 2011 rectifier paper only to Xavier Glorot rather than all three authors.

## Key takeaways

- An activation function transforms a unit’s weighted sum plus bias.
- Nonlinear activations allow multilayer networks to represent nonlinear functions.
- A threshold function creates a hard decision but does not support ordinary gradient-based training well.
- Sigmoid maps logits to values between 0 and 1 and is common for binary outputs.
- ReLU is a popular hidden-layer activation because it is simple and supports useful gradients on its positive side.
- Tanh is smooth and zero-centered but can saturate.
- ReLU’s derivative is undefined at zero; implementations choose a practical convention.
- Output activation and loss must be selected together.
- Binary classification commonly uses sigmoid behavior with binary cross-entropy.
- Mutually exclusive multiclass classification commonly uses softmax with categorical cross-entropy.
- The optimal activation depends on the task, architecture, and evidence.

## Review questions

1. Where does the activation function appear in the artificial-neuron equation?
2. Why would a network containing only linear activations still be a linear model?
3. Why is the threshold function difficult to train with backpropagation?
4. What is the range of the sigmoid function, and what does \(\sigma(0)\) equal?
5. Why can sigmoid and tanh contribute to vanishing gradients?
6. Write the formula for ReLU and describe its derivative.
7. What is a dying ReLU?
8. How does tanh differ from sigmoid in output range and centering?
9. Which output activation is typical for binary, multiclass, and multilabel classification?
10. Why might a loss function accept logits instead of activated probabilities?
11. Why is a hard prediction threshold conceptually separate from a trainable sigmoid output?
12. What did the 2011 rectifier paper contribute to the adoption of ReLU?

## Further reading

- [Glorot, Bordes, and Bengio (2011): *Deep Sparse Rectifier Neural Networks*](https://proceedings.mlr.press/v15/glorot11a.html)
