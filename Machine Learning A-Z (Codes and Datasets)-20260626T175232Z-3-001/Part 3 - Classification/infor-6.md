# False Positives and False Negatives

In binary classification, a model can make two kinds of incorrect prediction: a **false positive** or a **false negative**.

## False Positive: Type I Error

A **false positive** occurs when a model predicts a positive outcome, but the actual outcome is negative.

In other words, the predicted positive result is false:

```text
Predicted: Positive
Actual:    Negative
Result:    False Positive (Type I Error)
```

For example, an earthquake-warning system may predict that an earthquake will occur when no earthquake actually occurs. The warning is unnecessary, but it may still lead to anxiety, disruption, and costly emergency measures.

## False Negative: Type II Error

A **false negative** occurs when a model predicts a negative outcome, but the actual outcome is positive.

In this case, the predicted negative result is false:

```text
Predicted: Negative
Actual:    Positive
Result:    False Negative (Type II Error)
```

For example, an earthquake-warning system may predict that no earthquake will occur when one actually does. People may then be unprepared for the event.

## Which Error Is More Serious?

It may be tempting to remember a Type I error as less dangerous and a Type II error as more dangerous. However, this is not a general rule. The seriousness of each error depends entirely on the problem and its real-world consequences.

- In **disease screening**, a false negative may delay diagnosis and treatment.
- In **spam filtering**, a false positive may hide an important legitimate email.
- In **fraud detection**, a false negative may allow fraud to continue, while a false positive may block an innocent customer's transaction.
- In the **justice system**, a false positive could contribute to an innocent person being treated as guilty.

Both errors can therefore be serious. When designing and evaluating a classifier, we must decide which error is more costly in the specific application and select an appropriate decision threshold and evaluation metrics.

---

# Study Notes

## Quick Comparison

| Error | Model predicts | Actual class | Alternative name |
|---|---|---|---|
| **False Positive (FP)** | Positive | Negative | Type I error |
| **False Negative (FN)** | Negative | Positive | Type II error |

## How to Decode the Names

Read each term from right to left:

1. **Positive** or **negative** describes the model's prediction.
2. **False** means that the prediction was incorrect.

Therefore:

- **False positive:** An incorrect positive prediction.
- **False negative:** An incorrect negative prediction.

## Confusion-Matrix Position

Using actual classes as rows and predicted classes as columns:

| Actual \ Predicted | Negative | Positive |
|---|---:|---:|
| **Negative** | True Negative (TN) | **False Positive (FP)** |
| **Positive** | **False Negative (FN)** | True Positive (TP) |

Always inspect the axis labels because some libraries and diagrams display the matrix in a different orientation.

## Error-Rate Formulas

### False-Positive Rate

The **false-positive rate (FPR)** is the proportion of actual negatives that were incorrectly predicted as positive:

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

It is also:

$$
\text{FPR} = 1 - \text{Specificity}
$$

### False-Negative Rate

The **false-negative rate (FNR)** is the proportion of actual positives that were incorrectly predicted as negative:

$$
\text{FNR} = \frac{FN}{FN + TP}
$$

It is also:

$$
\text{FNR} = 1 - \text{Recall}
$$

## Connection to Precision, Recall, and Specificity

- More false positives generally reduce **precision** and **specificity**.
- More false negatives reduce **recall**, also called **sensitivity**.
- Reducing one type of error may increase the other when the classification threshold changes.

| Goal | Metric to monitor |
|---|---|
| Avoid incorrectly flagging negative cases | Specificity or false-positive rate |
| Make positive predictions more trustworthy | Precision |
| Avoid missing actual positive cases | Recall or false-negative rate |

## Decision-Threshold Trade-Off

Suppose a model produces a probability of belonging to the positive class.

- **Lowering the threshold** usually predicts more cases as positive. This tends to reduce false negatives but increase false positives.
- **Raising the threshold** usually predicts fewer cases as positive. This tends to reduce false positives but increase false negatives.

The best threshold is not automatically `0.5`. It should reflect the model's purpose, the relative cost of each error, class frequencies, and operational constraints.

## Important Terminology Note

The labels **Type I error** and **Type II error** originate in statistical hypothesis testing:

- A **Type I error** means rejecting a true null hypothesis.
- A **Type II error** means failing to reject a false null hypothesis.

In binary-classification discussions, these are commonly mapped to false positives and false negatives respectively. The mapping assumes that the positive prediction corresponds to rejecting the null hypothesis.

## Memory Aid

A reliable memory aid is:

- **False positive = false alarm:** The model raises an alarm for something that is absent.
- **False negative = missed detection:** The model fails to identify something that is present.

Use this only to remember the definitions—not to rank their seriousness.

## Exam-Style Summary

A false positive occurs when a classifier predicts the positive class for an actually negative observation; it is commonly called a Type I error. A false negative occurs when a classifier predicts the negative class for an actually positive observation; it is commonly called a Type II error. Neither error is universally more serious. Their relative cost depends on the application, and changing the decision threshold often creates a trade-off between them.

## Quick Review Questions

1. What does the word “false” indicate in a false positive?
2. What are the predicted and actual classes in a false negative?
3. Which error is commonly called a Type I error?
4. Why is a Type II error not always more serious than a Type I error?
5. How does lowering a classification threshold usually affect false positives and false negatives?
6. Which metric is directly affected by missed positive cases?
7. What is the difference between a false-positive rate and a false-negative rate?
