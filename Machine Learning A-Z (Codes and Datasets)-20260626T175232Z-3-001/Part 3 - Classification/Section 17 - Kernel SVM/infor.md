# Kernel Support Vector Machines

Previously, we examined the Support Vector Machine (SVM) algorithm and learned how it selects a decision boundary between classes. Given several possible separating hyperplanes, a linear SVM searches for the one that provides the maximum margin between the closest observations from each class.

Once trained, the model classifies a new observation according to the side of the decision boundary on which it falls.

## The Limitation of a Linear Boundary

A straight line can separate some two-dimensional datasets perfectly. Such data is described as **linearly separable**.

However, many real datasets cannot be divided effectively by a straight line. Imagine one class forming a cluster in the center of a graph while the other class surrounds it. No horizontal, vertical, or diagonal line can separate the two groups without making classification errors.

This is an example of data that requires a **non-linear decision boundary**.

A linear SVM can still be fitted to data that is not perfectly separable by using a soft margin. It will tolerate some classification errors or margin violations. Nevertheless, its decision boundary remains linear, so it may underfit data whose underlying class structure is curved or otherwise non-linear.

Kernel SVMs address this limitation.

## Moving to a Higher-Dimensional Space

Suppose that observations are described by a single feature \(x\), but the two classes cannot be separated by placing one threshold on the number line. We can transform the original feature into an additional feature:

\[
z = x^2
\]

The observations can then be represented using both \(x\) and \(z\). Although they were not linearly separable in the original one-dimensional space, they may become linearly separable in this new two-dimensional feature space.

The same principle applies more generally:

\[
\phi(x): \mathbb{R}^{d} \rightarrow \mathbb{R}^{D}
\]

where:

- \(x\) is an observation in the original \(d\)-dimensional input space;
- \(\phi(x)\) is its transformed representation; and
- \(D\) is the dimension of the new feature space, which may be much larger than \(d\).

An SVM can learn a linear hyperplane in the transformed feature space. When interpreted back in the original space, that hyperplane corresponds to a non-linear decision boundary.

The essential idea is:

> Data that is not linearly separable in its original feature space may become linearly separable after an appropriate transformation.

## Why Explicit Transformation Can Be Expensive

Creating all the transformed features explicitly can be computationally expensive. A transformation may produce hundreds, thousands, or even infinitely many dimensions.

Storing the transformed observations and calculating with every new feature would require substantial memory and processing time. Fortunately, the SVM optimization process depends mainly on dot products between observations rather than on the transformed coordinates themselves.

This makes the **kernel trick** possible.

## The Kernel Trick

A kernel function calculates the inner product between two observations in a transformed feature space without explicitly constructing their transformed feature vectors:

\[
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)
\]

In other words, the kernel tells the SVM how similar two observations would be after the transformation, while avoiding the cost of computing and storing every transformed coordinate.

The SVM can therefore learn a linear separator in an implicit higher-dimensional space. This produces a non-linear boundary in the original input space.

The workflow can be understood as:

1. Begin with data that cannot be separated effectively by a linear boundary.
2. Conceptually map the observations into a higher-dimensional feature space.
3. Find a maximum-margin hyperplane in that space.
4. Interpret the resulting classifier as a non-linear decision boundary in the original space.
5. Use a kernel to perform the necessary similarity calculations without constructing the higher-dimensional representation explicitly.

## Common Kernel Functions

### Linear Kernel

\[
K(x_i,x_j)=x_i^T x_j
\]

The linear kernel does not create a non-linear boundary. It is appropriate when the classes can be separated reasonably well by a hyperplane or when the feature space is already very large.

### Polynomial Kernel

\[
K(x_i,x_j)=(\gamma x_i^T x_j+r)^d
\]

The polynomial kernel can model curved interactions between features.

Its important parameters include:

- \(d\): the polynomial degree;
- \(\gamma\): the scale applied to the inner product; and
- \(r\): an independent constant, sometimes represented by `coef0`.

### Radial Basis Function Kernel

\[
K(x_i,x_j)=\exp\left(-\gamma\lVert x_i-x_j\rVert^2\right)
\]

The **Radial Basis Function (RBF)**, also called the Gaussian kernel, is a widely used choice for non-linear classification. It assigns high similarity to nearby observations and decreasing similarity as the distance between them increases.

The RBF kernel can produce highly flexible decision boundaries without requiring the transformed features to be calculated explicitly.

### Sigmoid Kernel

\[
K(x_i,x_j)=\tanh(\gamma x_i^T x_j+r)
\]

The sigmoid kernel resembles the activation function used in some neural-network models, although it is less commonly selected than the linear and RBF kernels.

## Choosing a Kernel

The best kernel depends on the structure of the data:

- Use a **linear kernel** when a linear boundary is sufficient or the dataset contains many features.
- Use an **RBF kernel** as a strong general-purpose option when the relationship appears non-linear.
- Use a **polynomial kernel** when polynomial feature interactions are plausible.

Kernel choice and parameter values should normally be evaluated through cross-validation rather than selected only from a visualization of the training data.

---

# Study Notes

## Key Terms

| Term | Meaning |
|---|---|
| **Linearly separable** | Classes can be separated perfectly by one linear hyperplane. |
| **Non-linearly separable** | No single linear hyperplane can perfectly divide the classes. |
| **Feature transformation** | A mapping from the original features to a new representation. |
| **Feature space** | The coordinate space in which observations are represented. |
| **Kernel** | A function that computes inner products in an implicit transformed feature space. |
| **Kernel trick** | Using a kernel to work in a higher-dimensional space without explicitly constructing its coordinates. |
| **Kernel SVM** | An SVM that uses a kernel to learn a non-linear boundary in the original input space. |
| **RBF kernel** | A kernel that measures similarity using the distance between observations. |

## Linear SVM vs. Kernel SVM

| Linear SVM | Kernel SVM |
|---|---|
| Learns a linear boundary in the original feature space | Can learn a non-linear boundary in the original feature space |
| Usually faster and easier to interpret | Usually more computationally expensive |
| Suitable for approximately linear class structures | Suitable for complex or curved class structures |
| Uses the ordinary dot product | Replaces the dot product with a kernel function |

## Linearly Separable vs. Non-Linear Structure

These ideas should not be confused:

- **Perfect linear separability** means one hyperplane can classify every training observation correctly.
- A **soft-margin linear SVM** can handle imperfect separation by allowing violations, but its boundary is still linear.
- A **kernel SVM** is useful when the class relationship itself requires a non-linear boundary.

## Core Kernel Identity

The key mathematical relationship is:

\[
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)
\]

The left side is calculated directly by the kernel. The right side is the dot product that would have been calculated after explicitly transforming both observations.

The kernel trick avoids computing \(\phi(x_i)\) and \(\phi(x_j)\) directly.

## Understanding the RBF Kernel

\[
K(x_i,x_j)=\exp\left(-\gamma\lVert x_i-x_j\rVert^2\right)
\]

- When two observations are close, their squared distance is small and the kernel value is near \(1\).
- When two observations are far apart, their squared distance is large and the kernel value approaches \(0\).
- The parameter \(\gamma\) determines how quickly similarity decreases with distance.

### Effect of \(\gamma\)

- **Small \(\gamma\):** each training observation has a broad region of influence, leading to a smoother boundary that may underfit.
- **Large \(\gamma\):** each observation has a narrow region of influence, leading to a more detailed boundary that may overfit.

### Effect of \(C\)

- **Small \(C\):** stronger regularization, a wider effective margin, and more tolerance for training errors.
- **Large \(C\):** weaker regularization, a greater penalty for errors, and a boundary that fits the training data more closely.

\(C\) and \(\gamma\) work together. Both should be tuned using validation data or cross-validation.

## Why Feature Scaling Matters

SVM kernels frequently depend on distances or dot products. If one feature ranges from 0 to 100,000 while another ranges from 0 to 1, the large-scale feature can dominate the calculation.

Standardization is therefore normally applied before training:

\[
x'=\frac{x-\mu}{\sigma}
\]

To prevent data leakage, the scaler must be fitted only on the training data and then used to transform both the training and test sets.

## Practical Workflow

1. Prepare the feature matrix \(X\) and target vector \(y\).
2. Split the observations into training and test sets.
3. Fit a feature scaler using only the training set.
4. Transform the training and test features with the fitted scaler.
5. Select a candidate kernel.
6. Tune parameters such as \(C\), \(\gamma\), and polynomial degree through cross-validation.
7. Train the Kernel SVM.
8. Evaluate it on unseen data using appropriate classification metrics.
9. Compare its performance with a simpler linear baseline.

## Advantages

- Models complex, non-linear class boundaries.
- Avoids explicitly creating a potentially enormous transformed feature space.
- Can perform well in high-dimensional spaces.
- Retains the maximum-margin principle of SVMs.

## Limitations

- Training can be slow and memory-intensive on large datasets.
- Choosing a kernel and tuning its parameters can be difficult.
- Feature scaling is generally essential.
- Flexible kernels can overfit without suitable regularization.
- The resulting decision boundary can be difficult to interpret.

## Common Misconceptions

1. **A kernel physically adds columns to the dataset.**  
   Not necessarily. The kernel computes the relevant inner products without explicitly creating those features.

2. **Mapping to a higher-dimensional space always guarantees perfect separation.**  
   A suitable transformation may make separation possible, but success depends on the data, kernel, and regularization.

3. **A Kernel SVM draws a curved hyperplane in the higher-dimensional space.**  
   The separator in the transformed space is still a linear hyperplane. It appears non-linear only when viewed in the original space.

4. **Kernel SVMs are always better than linear SVMs.**  
   A more flexible model is not automatically better. A linear SVM may generalize better, train faster, and require less tuning when the data is approximately linear.

## Quick Review Questions

1. What does it mean for a dataset to be linearly separable?
2. Why might a linear SVM underfit a dataset?
3. How can a feature transformation make separation easier?
4. What computational problem does the kernel trick solve?
5. What does a kernel function calculate?
6. Why does a linear hyperplane in a transformed space produce a non-linear boundary in the original space?
7. How does \(\gamma\) affect an RBF decision boundary?
8. How does \(C\) affect regularization and margin violations?
9. Why is feature scaling important for an RBF SVM?
10. Why should a Kernel SVM be compared with a linear baseline?

## One-Sentence Summary

A Kernel SVM learns a non-linear decision boundary by applying the maximum-margin principle in an implicit transformed feature space, using a kernel function to avoid calculating that transformation explicitly.
