# Principal Component Analysis (PCA)

## Overview

Principal Component Analysis (PCA) is one of the most widely used unsupervised learning techniques and one of the most popular methods for dimensionality reduction.

PCA is commonly used for:

- Data visualization
- Feature extraction
- Noise reduction and filtering
- Data compression
- Preprocessing for machine-learning models
- Applications such as financial analysis and gene-expression analysis

The main goal of PCA is to detect relationships among features and represent a high-dimensional dataset with fewer variables. It finds new directions in the data along which the variance is greatest, then projects the original observations onto a lower-dimensional subspace while retaining as much useful information as possible.

For a dataset with $D$ original dimensions, PCA projects the data onto a $K$-dimensional subspace, where $K < D$. The new dimensions are called **principal components**.

## Intuition

Suppose a dataset contains strongly correlated features. Those features may carry overlapping information. PCA replaces them with a smaller set of uncorrelated principal components:

- The **first principal component (PC1)** captures the greatest possible variance in the data.
- The **second principal component (PC2)** captures the greatest remaining variance and is perpendicular to PC1.
- Each subsequent component captures the greatest remaining variance while remaining orthogonal to all earlier components.

In two dimensions, PCA can be visualized as rotating the coordinate axes so that one axis follows the direction in which the data varies most. If most points lie close to that axis, the second axis contributes relatively little information and may be discarded.

The same idea applies in three or more dimensions, although the geometry becomes harder to visualize. PCA identifies the most informative axes in the higher-dimensional space and projects the data onto a smaller number of them.

## PCA Workflow

The main steps in PCA are:

1. **Standardize the features.** This is especially important when features use different units or scales.
2. **Calculate the covariance or correlation matrix.** This describes how the features vary together.
3. **Find the eigenvectors and eigenvalues.** The eigenvectors define the principal-component directions, while the eigenvalues indicate how much variance each direction captures.
4. **Sort the components.** Order the eigenvalues and their corresponding eigenvectors from largest to smallest.
5. **Choose the first $K$ components.** Select enough components to retain an acceptable proportion of the total variance.
6. **Construct the projection matrix $W$.** Its columns are the selected eigenvectors.
7. **Transform the data.** Project the standardized data onto the new subspace:

   $$Z = XW$$

Here, $X$ is the standardized dataset, $W$ is the projection matrix, and $Z$ is the lower-dimensional representation.

## PCA Compared with Linear Regression

PCA can look similar to linear regression in a two-dimensional plot, but the two techniques have different goals:

- **Linear regression** is supervised. It models a dependent variable and minimizes prediction error in that variable.
- **PCA** is unsupervised. It has no target variable and finds directions that maximize the variance of the features.

PCA therefore describes the structure of the feature space rather than predicting one variable from another.

## Limitations

PCA has several important limitations:

- It is sensitive to feature scale, so standardization is often required.
- It is sensitive to outliers because extreme observations can strongly affect variance and covariance.
- It captures only linear relationships.
- Principal components are combinations of the original features and can be difficult to interpret.
- A component with low variance is not necessarily irrelevant to a prediction task.

---

## Study Notes

### Key Terms

- **Dimensionality reduction:** Reducing the number of features while preserving as much useful information as possible.
- **Principal component:** A new feature formed as a linear combination of the original features.
- **Variance:** The amount of spread in the data. PCA treats high-variance directions as information-rich.
- **Eigenvector:** A direction that defines a principal component.
- **Eigenvalue:** The amount of variance explained by the corresponding eigenvector.
- **Explained variance ratio:** The proportion of total variance captured by a component.
- **Loading:** The coefficient showing how strongly an original feature contributes to a principal component.
- **Score:** An observation's coordinate after it has been projected onto a principal component.

### Essential Properties

- Principal components are mutually orthogonal and therefore uncorrelated.
- PCA orders components by explained variance.
- The first $K$ components provide the best rank-$K$ linear approximation of the centered data in the least-squares sense.
- PCA changes the feature representation; it does not simply select a subset of the original features.
- The signs of eigenvectors are arbitrary. Flipping the sign of a component does not change the PCA solution.

### Choosing the Number of Components

Common methods include:

- Keeping enough components to reach a chosen cumulative explained-variance threshold, such as 90% or 95%.
- Inspecting a scree plot and looking for an “elbow.”
- Selecting the number of components that gives the best cross-validated performance in the downstream task.

The retained variance after choosing $K$ components is:

$$
\text{Cumulative explained variance} =
\frac{\sum_{i=1}^{K} \lambda_i}{\sum_{j=1}^{D} \lambda_j}
$$

where $\lambda_i$ is the eigenvalue associated with component $i$.

### Practical Checklist

1. Separate features from the target, if a target exists.
2. Split the data into training and test sets before fitting preprocessing steps.
3. Fit the scaler using only the training data.
4. Fit PCA using only the scaled training data.
5. Apply the fitted scaler and PCA transformation to validation or test data.
6. Choose $K$ using explained variance or cross-validation.
7. Inspect loadings when interpretation matters.

> **Important:** Fitting the scaler or PCA on the complete dataset before the train/test split causes data leakage.

### Quick Review Questions

1. Why should features often be standardized before PCA?
2. What do eigenvectors and eigenvalues represent in PCA?
3. Why are principal components uncorrelated?
4. How does PCA differ from feature selection?
5. How can a scree plot help determine the number of components?
6. Why can outliers distort PCA?
7. Why must PCA be fitted only on the training data?

### One-Sentence Summary

PCA transforms correlated features into a smaller set of orthogonal components, ordered by the amount of variance they explain, so that the data can be represented more compactly.
