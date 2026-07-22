# Maximum Likelihood in Logistic Regression

A logistic regression model produces an S-shaped curve that estimates the probability of a binary outcome, such as whether a customer will accept an offer. However, many different logistic curves could potentially be fitted to the same data. How do we determine which curve fits best?

The answer is **maximum likelihood estimation (MLE)**.

## Evaluating a Candidate Curve

For every observation, the candidate logistic regression curve predicts a probability:

$$
p_i = P(y_i = 1 \mid x_i)
$$

Here, $p_i$ is the predicted probability that observation $i$ belongs to the positive class—for example, that the person says **yes** to the offer.

Suppose the model predicts the following probabilities for people who actually said **yes**:

$$
3\%,\ 54\%,\ 92\%,\ 95\%,\ 98\%
$$

For these observations, the likelihood calculation uses the predicted probabilities directly because the observed outcome is $y=1$.

Now suppose the model predicts these probabilities of saying **yes** for people who actually said **no**:

$$
1\%,\ 4\%,\ 10\%,\ 58\%,\ 96\%
$$

Because these observations have the outcome $y=0$, we need the corresponding probabilities of saying **no**:

$$
1-p_i
$$

The probabilities assigned to the observed **no** outcomes are therefore:

$$
99\%,\ 96\%,\ 90\%,\ 42\%,\ 4\%
$$

Notice that a prediction of a $96\%$ chance of **yes** gives only a $4\%$ chance to the observed **no** outcome. This sharply reduces the likelihood of that candidate curve.

## Calculating the Likelihood

The likelihood of a candidate curve is the product of the probabilities it assigns to all the outcomes that actually occurred:

$$
L = \prod_{i=1}^{n} p_i^{y_i}(1-p_i)^{1-y_i}
$$

This compact formula handles both possible outcomes:

- If $y_i=1$, the observation contributes $p_i$.
- If $y_i=0$, the observation contributes $1-p_i$.

For the example above, the likelihood is:

$$
L=(0.03)(0.54)(0.92)(0.95)(0.98)(0.99)(0.96)(0.90)(0.42)(0.04)
$$

A curve receives a high likelihood when it consistently assigns high probabilities to the outcomes that were actually observed. Incorrect predictions made with high confidence produce very small factors and strongly reduce the overall likelihood.

## Finding the Best-Fitting Curve

Different values of the logistic regression coefficients produce different curves and therefore different likelihoods. The fitting process adjusts the coefficients iteratively and searches for the values that produce the largest likelihood:

$$
\hat{\boldsymbol{\beta}}
=
\underset{\boldsymbol{\beta}}{\operatorname{argmax}}\;L(\boldsymbol{\beta})
$$

The curve associated with these coefficient values is the **maximum-likelihood estimate** and is treated as the best-fitting logistic regression curve for the observed data.

In practice, software does not manually test every possible curve. It uses a numerical optimization algorithm to find the coefficient values that maximize the likelihood.

## Log-Likelihood

Multiplying many probabilities can produce extremely small numbers. For numerical stability, logistic regression software normally maximizes the **log-likelihood** instead:

$$
\ell(\boldsymbol{\beta})
=
\sum_{i=1}^{n}
\left[
y_i\log(p_i)+(1-y_i)\log(1-p_i)
\right]
$$

Taking the logarithm changes products into sums and does not change which coefficient values give the maximum.

Equivalently, many machine-learning implementations minimize the **negative log-likelihood**, also called **binary cross-entropy** or **log loss**.

## Study Notes

- Logistic regression predicts $P(y=1\mid x)$, a probability between 0 and 1.
- For an observed positive case, use $p_i$ in the likelihood.
- For an observed negative case, use $1-p_i$ in the likelihood.
- The likelihood measures how probable the observed dataset is under a particular set of model coefficients.
- Maximum likelihood estimation chooses the coefficients that make the observed outcomes as probable as possible.
- One confidently incorrect prediction can substantially reduce the likelihood.
- Maximizing likelihood is equivalent to maximizing log-likelihood.
- Maximizing log-likelihood is equivalent to minimizing negative log-likelihood (binary cross-entropy).
- MLE finds the best fit for the observed data, but it does not guarantee good performance on unseen data; validation is still necessary.

### Key Formula

$$
\boxed{
L(\boldsymbol{\beta})
=
\prod_{i=1}^{n}p_i^{y_i}(1-p_i)^{1-y_i}
}
$$

**In one sentence:** the best logistic regression curve is the one that assigns the greatest combined probability to the outcomes that actually occurred.
