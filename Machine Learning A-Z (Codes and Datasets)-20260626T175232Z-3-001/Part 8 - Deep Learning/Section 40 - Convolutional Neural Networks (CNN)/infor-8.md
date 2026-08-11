# Softmax and Cross-Entropy

Softmax and cross-entropy are central to multiclass classification. They are frequently used at the output of a convolutional neural network, but the same ideas apply to many other classifiers.

Suppose a trained CNN receives an image and reports:

$$
P(\text{dog}) = 0.95, \qquad P(\text{cat}) = 0.05
$$

Why do these outputs lie between 0 and 1 and add up to 1? Ordinary output neurons do not coordinate with one another automatically. Before an output activation is applied, the network produces arbitrary real-valued scores called **logits**. A softmax function transforms those logits into a probability distribution.

## The Softmax Function

For $K$ mutually exclusive classes, softmax converts the logit $z_i$ for class $i$ into:

$$
\operatorname{softmax}(z_i)
= \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

The denominator contains the exponentiated logits for **all** classes. This shared denominator normalizes the outputs, ensuring that:

$$
0 < P(y=i) < 1
$$

and

$$
\sum_{i=1}^{K} P(y=i) = 1
$$

Softmax is sometimes called the **normalized exponential function**. Larger logits receive larger probabilities, but the output depends on the relative differences between all logits.

### Worked softmax example

Suppose the output layer produces these logits:

$$
z_{\text{dog}} = 2, \qquad z_{\text{cat}} = 0
$$

Then:

$$
P(\text{dog})
= \frac{e^2}{e^2 + e^0}
\approx 0.881
$$

$$
P(\text{cat})
= \frac{e^0}{e^2 + e^0}
\approx 0.119
$$

The probabilities add up to 1, and the larger dog logit receives the larger probability.

### When softmax is appropriate

Softmax is suitable when each example belongs to exactly one of several mutually exclusive classes—for example, dog, cat, or bird.

It is not normally used for **multilabel classification**, where several classes can be true at once. A photograph could contain both a dog and a cat, for example. In that setting, each class commonly receives an independent sigmoid output.

For a two-class problem, either of the following designs can be used:

- two softmax outputs with categorical cross-entropy; or
- one sigmoid output with binary cross-entropy.

The single-sigmoid design is usually simpler.

## Cross-Entropy Loss

After softmax generates the predicted class probabilities, cross-entropy measures how well they agree with the true label.

For a one-hot encoded target vector $y$ and a predicted probability vector $\hat{y}$, categorical cross-entropy for one example is:

$$
L(y, \hat{y})
= -\sum_{i=1}^{K} y_i \log(\hat{y}_i)
$$

Only the correct class has a target value of 1; every other target value is 0. The expression therefore simplifies to:

$$
L = -\log(\hat{y}_{\text{correct}})
$$

Cross-entropy focuses on the probability assigned to the true class:

- a high probability for the true class produces a small loss;
- a low probability for the true class produces a large loss; and
- a perfectly confident correct prediction approaches a loss of 0.

For example, if the image is a dog and the model predicts $P(\text{dog})=0.9$, then:

$$
L = -\log(0.9) \approx 0.105
$$

If it predicts only $P(\text{dog})=0.1$, then:

$$
L = -\log(0.1) \approx 2.303
$$

The confidently wrong prediction receives a much larger penalty.

## One-Hot and Sparse Labels

For a dog-versus-cat softmax classifier, a one-hot label might be represented as:

| True class | Dog target | Cat target |
| --- | ---: | ---: |
| Dog | 1 | 0 |
| Cat | 0 | 1 |

With **categorical cross-entropy**, the target is supplied as this full one-hot vector. With **sparse categorical cross-entropy**, the target is supplied as an integer class index. Both represent the same learning objective when configured correctly.

## Comparing Two Neural Networks

Consider three labeled images and two classifiers:

| Image | True class | Network 1: $P(\text{dog})$ | Network 1: $P(\text{cat})$ | Network 2: $P(\text{dog})$ | Network 2: $P(\text{cat})$ |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Dog | 0.90 | 0.10 | 0.60 | 0.40 |
| 2 | Cat | 0.10 | 0.90 | 0.30 | 0.70 |
| 3 | Dog | 0.40 | 0.60 | 0.10 | 0.90 |

Both networks classify two of the three images correctly. Their classification error is therefore identical:

$$
\text{Classification error} = \frac{1}{3} \approx 33.3\%
$$

However, Network 1 is clearly better calibrated to these examples. It is more confident on both correct predictions and less confidently wrong on the final prediction.

### Mean squared error

If squared errors are summed across the two class outputs and then averaged across the three images, the results are:

$$
\operatorname{MSE}_{1}
= \frac{0.02 + 0.02 + 0.72}{3}
\approx 0.253
$$

$$
\operatorname{MSE}_{2}
= \frac{0.32 + 0.18 + 1.62}{3}
\approx 0.707
$$

The exact numerical scale of MSE depends on whether an implementation averages over examples, output units, or both. The important result is that Network 1 receives the lower error.

### Cross-entropy

Categorical cross-entropy uses the probability assigned to the true class:

$$
\operatorname{CE}_{1}
= \frac{-\log(0.9)-\log(0.9)-\log(0.4)}{3}
\approx 0.376
$$

$$
\operatorname{CE}_{2}
= \frac{-\log(0.6)-\log(0.7)-\log(0.1)}{3}
\approx 1.057
$$

Again, Network 1 receives the lower loss. Unlike classification error, cross-entropy captures the quality and confidence of the predicted probabilities.

## Why Classification Error Is Not Used for Backpropagation

Accuracy and classification error depend only on which class has the largest score. Small changes in the probabilities usually do not change the predicted class, so the metric remains constant over large regions.

As a result, classification error does not provide useful gradients for training. It is valuable as an evaluation metric, but differentiable losses such as cross-entropy are used to optimize the model.

## Why Prefer Cross-Entropy to MSE for Classification?

Mean squared error can technically be used for classification, but cross-entropy is generally a better match for probabilistic classification.

There are two main reasons:

1. **Probabilistic interpretation:** Cross-entropy corresponds to the negative log-likelihood of the correct class. Minimizing it is equivalent to making the observed labels more probable under the model.
2. **Useful gradients:** Combining softmax with cross-entropy produces a particularly simple gradient with respect to each logit:

$$
\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i
$$

With softmax followed by MSE, the gradient also contains derivatives of the softmax function. When a softmax output is saturated near 0 or 1, those extra factors can make learning slower.

The logarithm also strongly penalizes assigning a tiny probability to the correct class. If the correct-class probability improves from $10^{-6}$ to $10^{-3}$:

$$
-\log(10^{-6}) \approx 13.816
$$

$$
-\log(10^{-3}) \approx 6.908
$$

Cross-entropy clearly reflects this thousandfold relative improvement even though both probabilities remain numerically small.

This does not mean MSE is universally inferior. It remains a common choice for regression because it models a different kind of prediction problem.

## Loss, Cost, and Objective

The terminology is not determined by whether a model is an ANN or a CNN. A common convention is:

- **loss:** the error for one example or one batch;
- **cost:** an aggregate of losses across a dataset, sometimes including regularization; and
- **objective:** the complete quantity being optimized.

In practice, authors and software libraries often use these terms interchangeably.

## Numerical Stability

Directly calculating exponentials can overflow when logits are large. Softmax is unchanged if the same constant is subtracted from every logit, so implementations commonly use:

$$
\operatorname{softmax}(z_i)
= \frac{e^{z_i-m}}{\sum_j e^{z_j-m}},
\qquad m = \max_j z_j
$$

This is more numerically stable.

Likewise, practical deep-learning libraries often combine softmax and cross-entropy into one operation that works directly from logits. This avoids explicitly evaluating $\log(0)$ and improves numerical stability. A model should not apply softmax twice: either pass logits to a loss configured to accept logits or pass probabilities to a loss expecting probabilities.

## Suggested Further Reading

- Geoffrey Hinton's lecture, *The Softmax Output Function*, provides an intuitive explanation of softmax-based classifiers.
- Rob DiPietro's 2016 article, *A Friendly Introduction to Cross-Entropy Loss*, offers an accessible introduction using information-theoretic intuition.
- Peter Roelants' 2016 neural-network intermezzo on softmax and cross-entropy develops the mathematics in greater detail.

## Study Notes

### Core formulas

**Softmax:**

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

**Categorical cross-entropy:**

$$
L = -\sum_i y_i\log(\hat{y}_i)
$$

**One-hot simplification:**

$$
L = -\log(\hat{y}_{\text{correct}})
$$

**Softmax-cross-entropy gradient:**

$$
\frac{\partial L}{\partial z_i} = \hat{y}_i-y_i
$$

### Essential distinctions

| Term | Meaning |
| --- | --- |
| Logit | An unrestricted score produced before the output activation |
| Probability | A normalized output used to express model confidence |
| Softmax | Converts a vector of logits into a categorical distribution |
| Cross-entropy | Penalizes disagreement between target and predicted distributions |
| Accuracy | Measures how often the selected class is correct |
| Loss | Supplies a differentiable training signal |

### Output-layer decision guide

| Task | Typical output | Typical loss |
| --- | --- | --- |
| Binary, mutually exclusive | One sigmoid neuron | Binary cross-entropy |
| Multiclass, mutually exclusive | One softmax neuron per class | Categorical cross-entropy |
| Multilabel | One sigmoid neuron per label | Binary cross-entropy per label |
| Regression | Linear output neuron(s) | MSE, MAE, or another regression loss |

### Common mistakes

- Treating logits as probabilities before applying an appropriate activation.
- Using softmax for classes that are not mutually exclusive.
- Swapping the target and predicted probability arguments in the cross-entropy formula.
- Applying softmax in the model and then using a loss configured to apply softmax again.
- Computing raw exponentials without numerical stabilization.
- Interpreting a probability as guaranteed real-world confidence without checking calibration.
- Using accuracy as the quantity for gradient-based optimization.

### Quick self-check

1. Why do softmax outputs add up to 1?
2. What is the cross-entropy loss when the correct class receives probability 1?
3. What happens to the loss as the correct-class probability approaches 0?
4. Why can two models have the same accuracy but different cross-entropy losses?
5. When should independent sigmoid outputs be preferred over softmax?
6. Why is a combined logits-based cross-entropy implementation preferable?

### Answers

1. Every exponentiated logit is divided by the same sum of all exponentiated logits.
2. It is $-\log(1)=0$.
3. The loss grows without bound because $-\log(p)\rightarrow\infty$ as $p\rightarrow0$.
4. Accuracy records only whether the winning class is correct, whereas cross-entropy also reflects the probability assigned to the true class.
5. Use independent sigmoids when multiple labels may be true for the same example.
6. It combines the operations using stable mathematics, avoids overflow and $\log(0)$ problems, and prevents accidental double application of softmax.

## Final Takeaway

Softmax converts a model's raw class scores into a probability distribution, while cross-entropy measures how much probability the model assigned to the correct class. Together, they provide an interpretable output and an effective gradient-based objective for mutually exclusive multiclass classification.
