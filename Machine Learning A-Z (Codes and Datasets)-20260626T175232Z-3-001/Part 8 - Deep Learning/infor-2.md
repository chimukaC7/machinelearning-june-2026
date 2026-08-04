# The Artificial Neuron

## Lesson objective

The neuron is the basic computational unit of an artificial neural network. This lesson uses the structure of a biological neuron as an intuition, then translates that idea into a mathematical model consisting of inputs, weights, a bias, a weighted sum, and an activation function.

The biological analogy is helpful, but it should not be interpreted literally. Artificial neurons are simplified mathematical operations; deep learning does not recreate the complete structure or behavior of a human brain.

## 1. The biological inspiration

A typical biological neuron is commonly described using four main structures:

- the **cell body**, or soma, which contains the nucleus and supports the cell;
- **dendrites**, which commonly receive signals from other cells;
- an **axon**, which carries an electrical signal away from the cell body; and
- **axon terminals**, which communicate with other cells at junctions called synapses.

At a typical chemical synapse, the cells do not directly merge. A narrow synaptic gap separates the sending and receiving cells. When a signal reaches an axon terminal, chemical messengers called neurotransmitters can cross this gap and influence the receiving cell.

This description captures the introductory idea, but real neuroscience is more complex. Neurons have many forms, signals may be excitatory or inhibitory, connections vary in strength, and not every neuron follows a perfectly simple dendrite-to-soma-to-axon pathway.

## 2. Santiago Ramón y Cajal and the neuron doctrine

Spanish neuroscientist **Santiago Ramón y Cajal** used and refined staining methods that made individual nerve cells visible in tissue. His careful observations and drawings helped establish the neuron doctrine: the nervous system is composed of distinct cells rather than one continuous fused network.

Cajal produced influential drawings across many years in the late nineteenth and early twentieth centuries. It is safer to identify a specific illustration by its documented title and date than to assume that every widely reproduced drawing comes from exactly 1899.

Cajal and Camillo Golgi shared the 1906 Nobel Prize in Physiology or Medicine for their work on the structure of the nervous system.

## 3. From a biological neuron to an artificial unit

An artificial neuron receives numerical inputs, combines them using learned parameters, and produces an output:

```text
inputs → weighted sum plus bias → activation function → output
```

The analogy is approximate:

| Biological term | Rough artificial-network analogy |
|---|---|
| Incoming neural activity | Numerical input values or earlier-layer activations |
| Synaptic influence | A learned weight applied to a connection |
| Cell body integration | Weighted sum plus bias |
| Firing response | Activation-function output |
| Axon output | Value passed to later units |

In machine learning, the connecting lines are usually called **connections** or **edges**, not biological synapses. Calling them synapses is an analogy rather than a technical requirement.

## 4. Inputs represent one observation

Suppose a dataset describes bank customers. One observation might contain:

- (x_1): age;
- (x_2): account balance;
- (x_3): credit score; and
- (x_m): another encoded customer feature.

For one customer, the features form an input vector:

\[
\mathbf{x} = [x_1, x_2, \ldots, x_m]
\]

All values in this vector describe the **same observation**. The target associated with that row must describe that same customer—for example, whether the customer left the bank.

During practical training, neural networks normally process several observations together in a **mini-batch**. The one-row explanation describes a single sample conceptually, not a restriction that networks must process only one row at a time.

## 5. Weights and the bias

Every incoming feature has an associated weight:

\[
w_1, w_2, \ldots, w_m
\]

A weight determines how strongly its corresponding input affects the unit. Its sign also matters:

- a positive weight can increase the pre-activation value as its input increases;
- a negative weight can decrease it; and
- a weight close to zero gives that input relatively little direct influence on the unit.

The neuron also usually has a **bias** (b). The bias shifts the activation threshold and allows the unit to produce a nonzero pre-activation even when every input is zero.

The parameters learned during training include both weights and biases. Omitting the bias from the neuron diagram leaves out an important part of the computation.

## 6. The weighted sum

The unit first calculates a linear combination of its inputs:

\[
z = w_1x_1 + w_2x_2 + \cdots + w_mx_m + b
\]

In vector notation:

\[
z = \mathbf{w}^{\mathsf{T}}\mathbf{x} + b
\]

The value (z) is often called the **pre-activation**, **logit** in certain output contexts, or simply the weighted sum plus bias.

### Small numerical example

Let:

\[
\mathbf{x} = [2, 3], \qquad \mathbf{w} = [0.5, -0.25], \qquad b = 0.1
\]

Then:

\[
z = (0.5)(2) + (-0.25)(3) + 0.1 = 0.35
\]

The activation function will transform this value into the neuron’s output.

## 7. The activation function

After computing (z), the unit applies an activation function φ:

\[
a = \phi(z)
\]

The value (a) is passed to the next layer or used as part of the final prediction.

An activation does not always make a binary decision to “pass” or “block” a signal. Many modern activation functions return continuous numerical values. Examples include:

- **ReLU:** \(\max(0, z)\);
- **sigmoid:** maps a value to the interval between 0 and 1;
- **tanh:** maps a value to the interval between -1 and 1; and
- **linear activation:** returns the input without nonlinear transformation.

Nonlinear activation functions are essential because stacking only linear transformations still produces an overall linear transformation. Nonlinearity allows a network to learn more complex relationships.

## 8. The complete artificial-neuron equation

The entire computation can be summarized as:

\[
a = \phi(\mathbf{w}^{\mathsf{T}}\mathbf{x} + b)
\]

where:

- \(\mathbf{x}\) is the input vector;
- \(\mathbf{w}\) is the vector of learned weights;
- \(b\) is the learned bias;
- \(z = \mathbf{w}^{\mathsf{T}}\mathbf{x} + b\) is the pre-activation;
- \(\phi\) is the activation function; and
- \(a\) is the unit’s output activation.

In a full neural network, many units perform this operation in parallel. Their outputs become the inputs to later layers.

## 9. Input, hidden, and output layers

### Input layer

The input layer represents the feature values supplied to the model. Input nodes are often drawn as neurons, but they usually do not perform the same weighted computation as hidden units; they primarily expose the input values to the first learned layer.

The human-senses analogy can be useful for intuition, but machine-learning inputs are not limited to equivalents of sight, hearing, touch, taste, and smell. A network can receive any suitable numerical representation, including financial records, scientific measurements, embeddings, or outputs from another system.

### Hidden layers

Hidden-layer units transform inputs or earlier activations. A hidden unit may receive values directly from the input layer or from another hidden layer. The same weighted-sum-and-activation principle applies in both cases.

### Output layer

The output layer is designed for the prediction task. Common configurations include:

| Task | Typical output design | Typical activation |
|---|---|---|
| Regression | One or more continuous outputs | Linear |
| Binary classification | One probability-like output | Sigmoid |
| Multiclass classification | One score per class | Softmax |
| Multilabel classification | One independent output per label | Sigmoid |

For multiclass classification, the network usually has one output unit per class. The target labels may be stored as integer class indices or one-hot vectors depending on the loss function and software API; one-hot or “dummy” encoding is not universally required.

## 10. Feature scaling

Features may have very different units and ranges. For example, age might range from 18 to 90 while account balance ranges from 0 to hundreds of thousands. Large scale differences can make gradient-based optimization harder.

### Standardization

Standardization commonly uses:

\[
x' = \frac{x - \mu}{\sigma}
\]

where \(\mu\) is the training-feature mean and \(\sigma\) is its standard deviation. The transformed training feature has a mean near 0 and a variance near 1 when the same population convention is used consistently.

### Min-max scaling

Min-max scaling commonly uses:

\[
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\]

This maps training values to the interval from 0 to 1 when the range is nonzero. New values outside the training range can transform to values below 0 or above 1 unless clipping is applied.

### Preventing leakage

The scaler must be fitted on the training set only:

1. calculate means, standard deviations, minima, or maxima from training data;
2. transform the training data with those learned values; and
3. transform validation and test data using the same values.

Calculating scaling statistics from the complete dataset leaks information from evaluation data into training.

Feature scaling is often helpful for neural-network optimization, but “every input must always be standardized” is too absolute. Binary indicators, already bounded measurements, learned embeddings, and architectures containing normalization layers may need different treatment. The correct choice depends on the representation and model.

## 11. How learning changes the neuron

Training does more than arbitrarily decide which signals matter. It optimizes model parameters to reduce a chosen loss function.

At a high level:

1. The network performs a forward pass and produces predictions.
2. A loss function measures prediction error.
3. Backpropagation applies the chain rule to compute gradients of the loss with respect to weights and biases.
4. An optimizer uses those gradients to update the parameters.
5. The process repeats over many mini-batches and epochs.

Gradient descent is a family of optimization ideas. Backpropagation is the efficient procedure used to calculate gradients through a computational graph. They are related, but they are not the same operation.

## 12. A layer in matrix form

Real implementations calculate many units and observations at once. For a batch of examples, a dense layer can be written as:

\[
\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}
\]

\[
\mathbf{A} = \phi(\mathbf{Z})
\]

Here:

- \(\mathbf{X}\) contains a batch of input observations;
- \(\mathbf{W}\) contains the weights for all units in the layer;
- \(\mathbf{b}\) contains their biases and is broadcast across the batch;
- \(\mathbf{Z}\) contains the pre-activations; and
- \(\mathbf{A}\) contains the output activations.

This matrix formulation is how the single-neuron idea scales efficiently to large networks on modern hardware.

## 13. Efficient BackProp

The lecture recommends *Efficient BackProp* by Yann LeCun and collaborators. The chapter discusses practical considerations for training neural networks, including input preprocessing and optimization. It remains historically valuable, although modern architectures and optimizers add many techniques developed since its 1998 publication.

The researcher mentioned earlier in the course is **Geoffrey Hinton**, not Jeffrey Hinton. Personal relationships between researchers are not needed to establish the technical value of their work and should not be asserted without a reliable source.

## Common misconceptions

- **“Deep learning recreates the human brain.”** Artificial networks use simplified mathematical operations and only loose biological inspiration.
- **“A neuron by itself is useless.”** An isolated biological neuron remains a complex living cell; the intended point is that cognition arises from networked activity.
- **“Every connecting line is literally a synapse.”** In artificial networks, it is a weighted mathematical connection.
- **“Dendrites always receive and axons always transmit in one simple direction.”** This is a useful introductory pattern, not a complete account of all neural signaling.
- **“The activation function only decides whether a signal passes.”** Many activations return graded continuous values.
- **“Weights are the only learned quantities.”** Biases and, in many architectures, several other parameter types are also learned.
- **“Standardization means dividing by variance.”** It normally divides by the standard deviation.
- **“Min-max scaling guarantees all future values remain between 0 and 1.”** Values outside the fitted training range may fall outside that interval.
- **“Multiclass targets always require manually created dummy variables.”** The required label format depends on the loss and implementation.
- **“A neural network processes only one row at a time.”** Training commonly uses mini-batches.

## Key takeaways

- A biological neuron has a cell body, dendrites, an axon, terminals, and synaptic connections.
- An artificial neuron is a mathematical abstraction, not a biological simulation.
- Its core computation is \(a = \phi(\mathbf{w}^{\mathsf{T}}\mathbf{x} + b)\).
- Weights control how inputs contribute; the bias shifts the pre-activation.
- Activation functions introduce the nonlinearity needed to learn complex relationships.
- All features in an input vector describe the same observation.
- Neural networks typically train on batches rather than literally one row at a time.
- Feature scaling often improves optimization and must be fitted using training data only.
- Output-layer structure and activation depend on the task.
- Backpropagation computes gradients; the optimizer uses them to update parameters.

## Review questions

1. What are the main structural parts of a biological neuron?
2. What is a synapse, and why do the communicating cells not need to touch directly?
3. Why is the biological-neuron analogy only approximate?
4. What roles do inputs, weights, the bias, and the activation function play?
5. Write the equation for a single artificial neuron.
6. Why does a multilayer network require nonlinear activation functions?
7. How do typical output layers differ for regression, binary classification, multiclass classification, and multilabel classification?
8. What is the difference between standardization and min-max scaling?
9. Why must a scaler be fitted only on training data?
10. How are backpropagation and gradient-based optimization related but distinct?
11. How does the matrix equation for a full dense layer extend the single-neuron equation?

## Further reading

- [Nobel Prize: the life and discoveries of Santiago Ramón y Cajal](https://www.nobelprize.org/prizes/medicine/1906/cajal/article/)
- [NINDS Brain Basics: neuron structure and signaling](https://www.ninds.nih.gov/es/node/8168)
- [Yann LeCun’s publication archive: *Efficient BackProp*](https://yann.lecun.com/exdb/publis/)
