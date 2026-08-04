# How Neural Networks Make Predictions

## Lesson objective

This lesson follows a trained feedforward neural network through a complete **forward pass**. The example predicts a property’s value from four input features:

1. floor area;
2. number of bedrooms;
3. distance from the nearest city; and
4. property age.

We assume that training has already taken place. The weights and biases are therefore fixed, and our goal is to understand how one observation is transformed into a prediction.

## 1. The property-valuation task

For one property, define the input vector:

\[
\mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{bmatrix}
=
\begin{bmatrix}
\text{area} \\
\text{bedrooms} \\
\text{distance to city} \\
\text{age}
\end{bmatrix}
\]

The target is a continuous property price:

\[
y = \text{property price}
\]

This is a **supervised regression** problem. During training, the model receives properties with known prices and learns parameters that reduce a regression loss. During inference, it receives a new property’s features and estimates its price.

Four variables are enough for an instructional example, but a useful valuation model may also need location, land area, condition, construction quality, recent comparable sales, neighborhood information, market timing, and other relevant factors. A model cannot recover important information that was never represented in its inputs.

## 2. Preprocessing the inputs

The raw features use very different units:

- area may be measured in square feet or square metres;
- bedrooms are a small integer count;
- distance may be measured in miles or kilometres; and
- age may be measured in years.

Neural networks commonly receive scaled or otherwise transformed values. If a scaler was learned during training, the same scaler must transform every property at inference time:

```text
raw property → fitted preprocessing → neural network → prediction
```

Changing units, column order, missing-value handling, or scaling rules after training changes the meaning of the inputs and can invalidate the prediction.

## 3. A model with no hidden layer

The simplest regression network connects the inputs directly to one output unit:

\[
\hat{y} = \mathbf{w}^{\mathsf{T}}\mathbf{x} + b
\]

Expanded:

\[
\hat{y} = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b
\]

With a linear output, this is a linear regression model expressed using neural-network notation. The weights describe the slope associated with each input, while the bias provides an intercept.

It is too broad to say that nearly every machine-learning algorithm becomes this model merely by changing the formula or activation. Some models can be expressed or approximated by neural architectures, but decision trees, nearest-neighbor methods, probabilistic graphical models, and many other algorithms have distinct structures and learning procedures.

### Choosing the output activation

The output transformation must match the target:

- **linear** for an unrestricted continuous price;
- **softplus** or another positive-valued transformation when a strictly positive prediction is required;
- **sigmoid** only if the target was deliberately scaled to a bounded interval such as 0 to 1.

A threshold function is not suitable for ordinary property-price regression because it can produce only discrete outcomes. Logistic regression is a classification model, not a general property-price estimator.

## 4. Adding a hidden layer

Now insert a hidden layer with \(H\) units. Each hidden unit receives the input vector and learns its own weights and bias:

\[
z_j^{(1)} = \sum_{i=1}^{m} w_{ji}^{(1)}x_i + b_j^{(1)}
\]

\[
h_j = \phi\left(z_j^{(1)}\right)
\]

where:

- \(m\) is the number of input features;
- \(j\) identifies a hidden unit;
- \(w_{ji}^{(1)}\) is the connection weight from input \(i\) to hidden unit \(j\);
- \(b_j^{(1)}\) is that unit’s bias;
- \(\phi\) is its activation function; and
- \(h_j\) is its output activation.

For the complete hidden layer:

\[
\mathbf{h} = \phi\left(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}\right)
\]

The hidden vector \(\mathbf{h}\) is a **learned representation** of the property.

## 5. Computing the output

For a single continuous output, the second layer combines the hidden activations:

\[
\hat{y} = \mathbf{w}^{(2)\mathsf{T}}\mathbf{h} + b^{(2)}
\]

The complete one-hidden-layer model is:

\[
\hat{y} =
\mathbf{w}^{(2)\mathsf{T}}
\phi\left(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}\right)
+ b^{(2)}
\]

This sequence is the forward pass:

```text
scaled features
    ↓
hidden weighted sums
    ↓
hidden activations
    ↓
output weighted sum
    ↓
predicted price
```

## 6. Worked numerical example

Assume the preprocessed feature vector for one property is:

\[
\mathbf{x} = [0.6,\ 0.5,\ -0.2,\ 0.8]^{\mathsf{T}}
\]

These are illustrative scaled values, not the raw square footage, bedroom count, distance, and age.

Suppose a trained hidden layer has three ReLU units:

\[
\mathbf{W}^{(1)} =
\begin{bmatrix}
1.2 & 0.0 & -0.8 & 0.0 \\
0.7 & 1.0 & 0.0 & -0.6 \\
0.0 & 0.0 & 0.0 & 1.5
\end{bmatrix}
\]

\[
\mathbf{b}^{(1)} =
\begin{bmatrix}
-0.1 \\
0.2 \\
-1.0
\end{bmatrix}
\]

### Hidden unit 1

\[
z_1 = (1.2)(0.6) + (0)(0.5) + (-0.8)(-0.2) + (0)(0.8) - 0.1
\]

\[
z_1 = 0.78
\]

\[
h_1 = \operatorname{ReLU}(0.78) = 0.78
\]

### Hidden unit 2

\[
z_2 = (0.7)(0.6) + (1.0)(0.5) + (0)(-0.2) + (-0.6)(0.8) + 0.2
\]

\[
z_2 = 0.64
\]

\[
h_2 = \operatorname{ReLU}(0.64) = 0.64
\]

### Hidden unit 3

\[
z_3 = (1.5)(0.8) - 1.0 = 0.20
\]

\[
h_3 = \operatorname{ReLU}(0.20) = 0.20
\]

The hidden representation is therefore:

\[
\mathbf{h} = [0.78,\ 0.64,\ 0.20]^{\mathsf{T}}
\]

Suppose the output layer uses weights measured in thousands of currency units:

\[
\mathbf{w}^{(2)} = [120,\ 80,\ 50]^{\mathsf{T}}, \qquad b^{(2)} = 150
\]

Then:

\[
\hat{y} = (120)(0.78) + (80)(0.64) + (50)(0.20) + 150
\]

\[
\hat{y} = 304.8
\]

The illustrative predicted property price is therefore **304.8 thousand currency units**.

The numbers above were chosen for explanation. A real trained model would learn all weights and biases from data.

## 7. What hidden units can learn

Nonlinear hidden units can respond to combinations of features rather than treating every feature as having one constant effect everywhere.

For example, a unit might become sensitive to some mixture of:

- large area and short distance to the city;
- several bedrooms and relatively new construction;
- property age beyond a learned boundary; or
- a combination that has no obvious human label.

This allows the model to construct intermediate features automatically. With ReLU activations, the network represents a piecewise-linear function: different units become active in different regions of the input space, producing different local linear behavior.

### A unit does not literally reason

Descriptions such as “this neuron looks for large homes close to the city” are interpretations imposed by a human observer. The unit itself performs only a numerical calculation. It has no intention, awareness, or explicit rule written in natural language.

Even when an interpretation appears plausible, it should be tested using activation analysis, sensitivity methods, feature attribution, or controlled examples.

## 8. How a ReLU unit represents a threshold-like pattern

Consider a unit based only on property age:

\[
h = \operatorname{ReLU}(w_{\text{age}}x_{\text{age}} + b)
\]

If age is used in raw years, a positive weight \(w\) and bias \(b=-100w\) produce:

\[
h = \operatorname{ReLU}\bigl(w(x_{\text{age}}-100)\bigr)
\]

This unit outputs zero for ages below 100 and increases linearly above 100. The bias is what shifts the ReLU boundary away from zero.

However, this does not prove the network has learned the human concept “historic property.” It has learned a numerical breakpoint. Whether that breakpoint represents heritage value, an artifact in the dataset, renovation patterns, location, or another correlated factor requires investigation.

If age was standardized, the corresponding threshold must also be expressed in standardized units. A raw threshold of 100 cannot be inserted directly into a network expecting scaled age values.

## 9. Are unimportant connections exactly zero?

In a standard dense layer, every input is connected to every hidden unit. Training may make some weights small, but it does not ordinarily force them to become exactly zero.

A zero or near-zero weight means one input has little direct influence on that unit under the current parameterization. Exact sparsity may arise through:

- explicit architectural choices;
- pruning;
- masks;
- sparsity-inducing penalties; or
- optimization that happens to reach zero.

Removing lines from a teaching diagram is acceptable for clarity, but the omitted connections should not automatically be interpreted as nonexistent in the implemented dense network.

## 10. Distributed representations

The power of a hidden layer does not require every unit to discover one clean concept. Neural networks often use **distributed representations**:

- one concept may be represented across several units;
- one unit may contribute to several patterns;
- multiple units may overlap or compensate for one another; and
- units can be permuted without changing the network’s function if the next layer is permuted consistently.

This means a hidden unit is not guaranteed to correspond to “urban spaciousness,” “family home,” or “historic building.” Those labels are useful stories for intuition, not promises of interpretability.

## 11. Why the hidden layer adds expressive power

The essential source of additional expressiveness is the combination of:

1. learned linear transformations; and
2. nonlinear activation functions.

If the hidden activation were linear, the two layers would collapse into one linear mapping:

\[
\mathbf{w}^{(2)\mathsf{T}}
\left(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}\right)
+ b^{(2)}
\]

This can be rearranged into another weighted sum of \(\mathbf{x}\). It would still be a linear model.

With nonlinear activations, the hidden layer can model interactions and changing relationships. For instance, the marginal effect of area can differ across regions defined by distance, age, and bedroom count.

Hidden layers can make a model more flexible, but they do not automatically increase test accuracy. Excess capacity can overfit, and successful use depends on appropriate data, regularization, optimization, architecture, and evaluation.

## 12. Inference versus training

This lesson focuses on **inference**:

```text
fixed parameters + new features → prediction
```

Training adds a feedback process:

```text
features → prediction → loss against known price
                           ↓
                gradients via backpropagation
                           ↓
                 update weights and biases
```

During inference, the network does not normally change its weights. It applies the transformations learned during training.

## 13. Batch form of the forward pass

For a batch containing \(n\) properties:

\[
\mathbf{X} \in \mathbb{R}^{n \times m}
\]

With \(H\) hidden units:

\[
\mathbf{W}^{(1)} \in \mathbb{R}^{m \times H}
\]

One common row-oriented convention is:

\[
\mathbf{H} = \phi\left(\mathbf{XW}^{(1)} + \mathbf{b}^{(1)}\right)
\]

\[
\hat{\mathbf{y}} = \mathbf{H}\mathbf{w}^{(2)} + b^{(2)}
\]

Different textbooks may transpose the matrices and observations. Both conventions are valid if the dimensions are consistent.

## 14. Checking a valuation model

A property model should be evaluated on held-out data using regression metrics such as:

- mean absolute error;
- mean squared error or root mean squared error;
- median absolute error;
- \(R^2\), interpreted carefully; and
- error slices by location, price band, property type, and other relevant groups.

Predictions should also be checked for implausible values, sensitivity to unit changes, distribution shift, and systematic errors across populations or neighborhoods. A low average error can conceal serious errors for particular groups or rare property types.

## Common pitfalls

- Omitting biases from the forward-pass equations.
- Feeding raw features to a model trained on scaled features.
- Changing the input-column order at inference time.
- Using an output activation that does not match a continuous regression target.
- Claiming that a one-layer neural diagram represents nearly every machine-learning algorithm.
- Assuming unimportant dense connections become exactly zero automatically.
- Treating hidden units as conscious agents that decide what to inspect.
- Assigning a human-readable meaning to a unit without evidence.
- Forgetting that ReLU thresholds are controlled by both weights and biases.
- Assuming hidden layers always improve accuracy.
- Confusing a forward pass with the training process.
- Interpreting a scaled output as a raw price without applying the required inverse transformation.

## Key takeaways

- A forward pass transforms input features into a prediction using fixed learned parameters.
- A network with a linear output and no hidden layer is a linear regression model.
- A hidden layer computes \(\mathbf{h}=\phi(\mathbf{W}^{(1)}\mathbf{x}+\mathbf{b}^{(1)})\).
- The output layer combines hidden activations to produce the prediction.
- Nonlinear activations—not diagram depth alone—give hidden layers additional expressive power.
- Hidden units can learn useful combinations of property features.
- Interpretations of individual units are hypotheses, not guaranteed explanations.
- Standard dense networks usually retain all connections even when some weights are small.
- The bias controls where a ReLU unit becomes active.
- More flexible models still require sound preprocessing, regularization, and evaluation.

## Review questions

1. Why is property valuation a regression problem?
2. What equation describes a model with no hidden layer and a linear output?
3. What two computations occur inside each hidden unit?
4. Write the full equation for a one-hidden-layer regression network.
5. Why would two linear layers without a nonlinear activation collapse into one linear layer?
6. What does the hidden activation vector represent?
7. How does a bias move the activation boundary of a ReLU unit?
8. Why should a hidden unit’s apparent human-readable meaning be treated cautiously?
9. Why are omitted connections in a teaching diagram not necessarily zero in an implemented dense layer?
10. What is the difference between inference and training?
11. Why must inference reuse the training-time preprocessing rules?
12. Which evaluation checks would you use for a real property-valuation model?
