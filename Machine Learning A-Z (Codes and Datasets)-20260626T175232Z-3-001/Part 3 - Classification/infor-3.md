# Logistic Regression

## Overview

**Logistic Regression** is a supervised learning algorithm used primarily for classification. It predicts the probability that an observation belongs to a particular category based on one or more independent variables, or **features**.

Despite the word *regression* in its name, Logistic Regression is a classification method. Its output is a probability, which can then be converted into a class label.

## Logistic Regression versus Linear Regression

| Linear Regression | Logistic Regression |
| --- | --- |
| Predicts a continuous value | Predicts the probability of a class |
| Output can be any real number | Probability is constrained between `0` and `1` |
| Fits a straight line or hyperplane | Applies a sigmoid function to a linear score |
| Example: predict income | Example: predict whether insurance will be purchased |

Using Linear Regression directly for a binary outcome is unsuitable because its predictions can fall below `0` or above `1`, which are not valid probabilities.

## Example: Predicting an insurance purchase

Suppose an insurance company wants to predict whether a person will purchase a health insurance plan.

- **Independent variable (`x`):** age
- **Dependent variable (`y`):** purchase decision
- `y = 0`: the person did not purchase the plan
- `y = 1`: the person purchased the plan

The training data contains each person's age and known purchase outcome. Because the target has only two possible values, the observations appear along the horizontal levels `0` and `1`, rather than being distributed continuously as they would be in a regression problem.

## The linear score

Logistic Regression first forms a linear combination of the input features:

$$
z = b_0 + b_1x_1 + b_2x_2 + \cdots + b_kx_k
$$

where:

- `b₀` is the intercept.
- `b₁, b₂, ..., bₖ` are the learned coefficients.
- `x₁, x₂, ..., xₖ` are the feature values.

The linear score `z` can be any real number, so it must be transformed before it can represent a probability.

## The sigmoid function

The **sigmoid**, or **logistic**, function maps the linear score to a value between `0` and `1`:

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

The resulting S-shaped curve approaches `0` for very negative values of `z` and approaches `1` for very positive values.

In binary classification:

$$
p = P(y=1 \mid X)
$$

Therefore, `p` is the estimated probability that an observation belongs to the positive class.

## The log-odds equation

The Logistic Regression model can also be written as:

$$
\ln\left(\frac{p}{1-p}\right)
= b_0 + b_1x_1 + b_2x_2 + \cdots + b_kx_k
$$

The expression:

$$
\frac{p}{1-p}
$$

is called the **odds**, while its natural logarithm is called the **log-odds** or **logit**.

Logistic Regression assumes that the log-odds of the positive outcome are a linear combination of the features. It does **not** assume that the probability itself has a linear relationship with the features.

## From probabilities to class predictions

The model naturally produces probabilities. If a definite class label is required, a decision threshold is applied:

$$
\hat{y} =
\begin{cases}
1, & p \ge t \\
0, & p < t
\end{cases}
$$

where `t` is the chosen threshold.

A common default is `t = 0.50`:

- If `p ≥ 0.50`, predict class `1`.
- If `p < 0.50`, predict class `0`.

For example:

| Person | Predicted purchase probability | Prediction at `t = 0.50` |
| --- | ---: | --- |
| Age 35 | `0.42` | Will not purchase (`0`) |
| Age 45 | `0.81` | Will purchase (`1`) |

The threshold is a business or modeling decision, not an unchangeable part of Logistic Regression. For instance, an insurer might lower it when identifying potential buyers for a low-cost marketing campaign, increasing recall at the expense of more false positives.

## Using multiple features

Logistic Regression can use many independent variables. An insurance-purchase model might include:

- Age
- Income
- Education level
- Employment status
- Family size
- Marital status
- Previous insurance history

With several features, the model becomes:

$$
P(y=1 \mid X)
= \frac{1}{1 + e^{-(b_0+b_1x_1+b_2x_2+\cdots+b_kx_k)}}
$$

For two features, the `0.50` probability threshold corresponds to a straight decision boundary. With more features, it becomes a linear hyperplane in a higher-dimensional feature space.

## Interpreting coefficients

Each coefficient describes how its feature affects the log-odds of the positive class while the other features are held constant.

- A positive coefficient increases the predicted probability of class `1` as the feature increases.
- A negative coefficient decreases the predicted probability of class `1` as the feature increases.
- A coefficient near zero indicates little linear effect on the log-odds.

Exponentiating a coefficient gives an **odds ratio**:

$$
\text{odds ratio for } x_j = e^{b_j}
$$

For a one-unit increase in `xⱼ`, the odds are multiplied by `e^{bⱼ}`, assuming all other features remain constant. This interpretation depends on the feature's units and how it was encoded or scaled.

## How the model is trained

Logistic Regression does not use Ordinary Least Squares. It estimates the coefficients by maximizing the likelihood of the observed class labels, or equivalently by minimizing **log loss** (binary cross-entropy):

$$
L = -\frac{1}{n}\sum_{i=1}^{n}
\left[y_i\ln(p_i) + (1-y_i)\ln(1-p_i)\right]
$$

This loss strongly penalizes confident predictions that are wrong.

## Important assumptions and considerations

- The observations should be independent unless the modeling method explicitly handles dependence.
- The log-odds should have an approximately linear relationship with continuous predictors.
- Predictors should not exhibit severe multicollinearity.
- Extremely influential observations and outliers can distort the fitted coefficients.
- A sufficiently large and representative training sample is needed.
- Categorical variables must be encoded numerically.
- Feature scaling is often beneficial when regularization is used or when features have very different scales.

Logistic Regression may struggle when classes have a strongly nonlinear boundary unless nonlinear transformations or interaction terms are added.

## Evaluating Logistic Regression

Useful evaluation tools include:

- Confusion matrix
- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- ROC curve and ROC-AUC
- Precision-recall curve
- Log loss
- Probability calibration

Accuracy should not be used alone for imbalanced datasets. The appropriate metric and threshold depend on the real-world costs of false positives and false negatives.

## Key takeaway

Logistic Regression transforms a linear combination of features through the sigmoid function to estimate the probability of a categorical outcome. A threshold can convert that probability into a class label, while the model's coefficients provide a useful interpretation in terms of log-odds and odds ratios.
