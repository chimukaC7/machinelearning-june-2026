# Support Vector Machines (SVMs)

Support Vector Machines, usually abbreviated as **SVMs**, are supervised machine-learning algorithms used primarily for classification, although they can also be adapted for regression.

The foundations of SVMs date back to the 1960s, and the method was refined significantly during the 1990s. SVMs remain important because they approach classification differently from many other algorithms: they focus on finding a decision boundary with the largest possible margin between classes.

## Finding a Decision Boundary

Consider a dataset with two input features, \(x_1\) and \(x_2\). Each observation belongs to one of two classes, which we can represent as red and green points on a two-dimensional graph.

Our goal is to draw a boundary that separates the two classes. Once the boundary has been learned, a new observation can be classified according to the side of the boundary on which it falls.

For a linearly separable dataset, many different lines might correctly separate the training observations. We could draw a vertical, horizontal, or diagonal line and still classify every training point correctly. However, these boundaries may produce different predictions when new observations are introduced.

An SVM therefore does not search for just any separating boundary. It searches for the boundary that provides the **maximum margin** between the two classes.

## The Maximum-Margin Hyperplane

In two dimensions, the SVM decision boundary is a line. In three dimensions, it is a plane. More generally, in a space with any number of dimensions, it is called a **hyperplane**.

The selected decision boundary is known as the **maximum-margin hyperplane** or **maximum-margin classifier**. It lies between the two classes and is positioned so that the distance to the nearest observation on either side is as large as possible.

Two parallel boundaries pass through the closest observations from the two classes:

- the **positive hyperplane**, often written as \(w^T x + b = 1\); and
- the **negative hyperplane**, often written as \(w^T x + b = -1\).

The central decision boundary is:

\[
w^T x + b = 0
\]

where:

- \(x\) is an input-feature vector;
- \(w\) is the weight vector, which determines the orientation of the hyperplane; and
- \(b\) is the bias, which shifts the hyperplane's position.

The class prediction is based on the sign of the decision function:

\[
\hat{y} = \operatorname{sign}(w^T x + b)
\]

## Support Vectors

The observations closest to the decision boundary are called **support vectors**. They determine the position and orientation of the maximum-margin hyperplane.

They are called *vectors* because each observation can be represented as a feature vector:

\[
x = [x_1, x_2, \ldots, x_n]
\]

Although an observation appears as a point in a two-dimensional illustration, it is more generally a vector in an \(n\)-dimensional feature space.

For a hard-margin SVM, observations that lie farther from the boundary do not directly affect the final separating hyperplane. If a non-support-vector observation is removed without changing which points are closest to the boundary, the learned boundary remains the same. Moving or removing a support vector, however, may change the boundary.

This dependence on a small set of critical training observations gives the Support Vector Machine its name.

## Intuition: Learning from Difficult Cases

Imagine training a model to distinguish apples from oranges. Some examples are easy to classify: a typical red apple and a typical orange-colored orange are clearly different. Other examples are more ambiguous, such as a green orange or an orange-colored apple.

The ambiguous examples are likely to lie close to the decision boundary. If they become support vectors, they have a direct influence on where the SVM places that boundary.

This does not mean that an SVM simply searches for unusual or extreme observations. More precisely, it focuses on the observations closest to the boundary between the classes. These difficult cases contain the most useful information for defining the separation.

By maximizing the margin around those observations, the SVM aims to create a boundary that generalizes well to unseen data.

## Linearly Separable and Non-Linearly Separable Data

The simplest SVM is the **hard-margin SVM**. It assumes that the classes are perfectly linearly separable, meaning that a hyperplane can divide the classes without making any training errors.

Real datasets often contain noise, overlapping classes, or outliers. A **soft-margin SVM** allows some observations to fall inside the margin or even on the wrong side of the decision boundary. The parameter \(C\) controls the trade-off between:

- maximizing the margin; and
- penalizing classification errors.

A large \(C\) places a stronger penalty on training errors and generally produces a narrower margin. A small \(C\) allows more margin violations and generally produces a wider, more regularized margin.

When a straight hyperplane cannot separate the classes effectively, an SVM can use the **kernel trick**. A kernel measures similarity as if observations had been mapped into a higher-dimensional feature space, without requiring that mapping to be calculated explicitly. Common kernels include:

- linear;
- polynomial; and
- radial basis function (RBF).

## Why SVMs Can Be Effective

SVMs can perform well when:

- the classes have a clear margin of separation;
- the number of features is large relative to the number of observations;
- a non-linear boundary can be represented effectively by a suitable kernel; and
- the features have been scaled appropriately.

Their performance depends on selecting a suitable kernel and tuning parameters such as \(C\) and, for kernels such as RBF, \(\gamma\). SVMs can be sensitive to feature scale and may become computationally expensive on very large datasets.

---

# Study Notes

## Key Terms

| Term | Meaning |
|---|---|
| **SVM** | A supervised algorithm that finds a decision boundary with the largest possible margin between classes. |
| **Decision boundary** | The boundary used to assign observations to different classes. |
| **Hyperplane** | The general name for the decision boundary in a multidimensional feature space. |
| **Margin** | The distance between the decision boundary and the nearest observations from the classes. |
| **Support vectors** | The observations closest to the decision boundary that determine its position. |
| **Hard margin** | An SVM formulation that permits no training misclassifications and requires linearly separable data. |
| **Soft margin** | An SVM formulation that permits margin violations in exchange for better robustness and generalization. |
| **Kernel** | A function that enables an SVM to learn a non-linear boundary through similarity calculations in an implicit feature space. |
| **\(C\)** | A regularization parameter controlling the penalty for margin violations. |
| **\(\gamma\)** | A parameter used by kernels such as RBF that controls how locally each training observation influences the boundary. |

## Core Idea

> An SVM chooses the separating hyperplane that maximizes the distance to the nearest training observations from each class.

For the canonical hard-margin formulation, maximizing the margin is equivalent to minimizing:

\[
\frac{1}{2}\lVert w\rVert^2
\]

subject to:

\[
y_i(w^T x_i + b) \geq 1
\]

for every training observation \(i\), where \(y_i \in \{-1,+1\}\).

The total distance between the positive and negative supporting hyperplanes is:

\[
\frac{2}{\lVert w\rVert}
\]

Therefore, minimizing \(\lVert w\rVert\) maximizes the margin.

## Hard Margin vs. Soft Margin

| Hard-margin SVM | Soft-margin SVM |
|---|---|
| Requires perfect linear separability | Allows errors and margin violations |
| Sensitive to noise and outliers | More suitable for real-world data |
| No slack variables | Uses slack variables \(\xi_i\) |
| Focuses on the widest error-free margin | Balances margin width against classification errors |

The soft-margin objective is commonly written as:

\[
\min_{w,b,\xi}\left(\frac{1}{2}\lVert w\rVert^2 + C\sum_{i=1}^{n}\xi_i\right)
\]

subject to:

\[
y_i(w^T x_i+b) \geq 1-\xi_i,\qquad \xi_i \geq 0
\]

## Effect of Important Parameters

### The \(C\) Parameter

- **Large \(C\):** places a high cost on errors, attempts to classify more training observations correctly, and may overfit.
- **Small \(C\):** accepts more violations, increases regularization, and may generalize better.

### The RBF \(\gamma\) Parameter

- **Large \(\gamma\):** each observation has a short-range influence, producing a more complex boundary that may overfit.
- **Small \(\gamma\):** each observation has a wider influence, producing a smoother boundary that may underfit.

## Practical Workflow

1. Separate the feature matrix \(X\) from the target vector \(y\).
2. Split the data into training and test sets.
3. Scale the features, because SVMs are based on distances and dot products.
4. Select a kernel appropriate for the problem.
5. Tune parameters such as \(C\) and \(\gamma\), preferably using cross-validation.
6. Train the SVM on the training set.
7. Evaluate it using a confusion matrix and suitable classification metrics.
8. Inspect performance on unseen data and watch for overfitting.

## Advantages

- Effective in high-dimensional feature spaces.
- Can model both linear and non-linear decision boundaries.
- Uses only the support vectors to define the final boundary.
- Often performs well on small and medium-sized datasets.

## Limitations

- Feature scaling is usually essential.
- Training can be slow for very large datasets.
- Kernel and parameter selection can require careful tuning.
- Predictions are less interpretable than those from simple linear models or decision trees.
- Standard SVM outputs are not probabilities unless probability calibration is added.

## Quick Review Questions

1. Why can several lines classify the same training data correctly?
2. What makes the SVM decision boundary preferable to another separating boundary?
3. What are support vectors, and why are they important?
4. What is the difference between a line, a plane, and a hyperplane?
5. When should a soft-margin SVM be preferred to a hard-margin SVM?
6. How does increasing \(C\) affect the model?
7. What problem does the kernel trick solve?
8. Why should input features normally be scaled before fitting an SVM?

## One-Sentence Summary

An SVM classifies data by using the observations nearest the class boundary to construct a hyperplane with the largest practical margin between the classes.
