# Linear Discriminant Analysis (LDA): Intuition and Study Notes

## Introduction

**Linear Discriminant Analysis (LDA)** is a supervised dimensionality-reduction technique commonly used as a preprocessing step for classification and other machine-learning tasks. Like Principal Component Analysis (PCA), LDA projects a dataset into a lower-dimensional space. The crucial difference lies in how the new axes are chosen.

PCA finds directions that preserve as much overall variance as possible. LDA uses the class labels and finds directions that provide the greatest separation between classes. In other words, LDA attempts to keep observations from the same class close together while pushing observations from different classes farther apart.

> **Main takeaway:** PCA focuses on variance in the features, whereas LDA focuses on separation between labeled classes.

## The Goal of LDA

LDA projects the original feature space onto a smaller subspace while preserving as much **class-discriminatory information** as possible. It seeks linear combinations of the original features, called **linear discriminants**, that:

- Minimize the spread of observations within each class.
- Maximize the distance between the class means.

Conceptually, LDA searches for directions that maximize the following ratio:

$$
\text{class separability}
=
\frac{\text{between-class variance}}{\text{within-class variance}}
$$

A large ratio indicates that the projected class centers are far apart and that observations belonging to the same class are tightly grouped.

## PCA Versus LDA

Both PCA and LDA are linear transformation techniques that can reduce the number of dimensions, but they optimize different objectives.

| Property | PCA | LDA |
|---|---|---|
| Learning type | Unsupervised | Supervised |
| Uses class labels? | No | Yes |
| Main objective | Maximize retained variance | Maximize class separability |
| Extracted features | Principal components | Linear discriminants |
| Best suited for | Compression, visualization, noise reduction | Preprocessing for classification |
| Maximum number of components | Up to the number of original features | At most `min(number of features, number of classes - 1)` |

PCA may discover a direction with high variance even if the classes overlap along that direction. LDA may prefer a lower-variance direction when it separates the labeled classes more effectively.

Because LDA uses the dependent variable—the class label—it is supervised. PCA considers only the independent variables and is therefore unsupervised.

## The Main Steps of LDA

A conceptual LDA implementation follows these steps:

1. **Compute the mean vectors.** Calculate the mean feature vector for each class and, when needed, the overall mean.
2. **Compute the scatter matrices.** Construct the within-class scatter matrix and the between-class scatter matrix.
3. **Solve the eigenvalue problem.** Find the eigenvectors and eigenvalues associated with the relationship between the two scatter matrices.
4. **Rank the discriminants.** Sort the eigenvectors by decreasing eigenvalue. Larger eigenvalues correspond to directions with greater discriminatory power.
5. **Project the observations.** Form a transformation matrix from the selected eigenvectors and use it to map the original samples into the lower-dimensional subspace.

In practice, a library such as scikit-learn performs these calculations internally.

## Understanding the Scatter Matrices

The two scatter matrices capture the competing goals of LDA:

- **Within-class scatter, $S_W$:** measures how widely observations are dispersed around their own class mean. LDA tries to make this small after projection.
- **Between-class scatter, $S_B$:** measures how far the class means are from the overall mean. LDA tries to make this large after projection.

The discriminant directions are selected to maximize a criterion based on $S_W^{-1}S_B$. This is the mathematical expression of finding compact classes that are well separated.

## Number of Linear Discriminants

If a dataset contains $C$ classes, LDA can produce at most $C - 1$ useful linear discriminants. The number is also limited by the number of original features.

Therefore:

$$
n_{\text{components}} \leq \min(n_{\text{features}}, C - 1)
$$

For example, a wine dataset with three customer segments can have at most two linear discriminants. This makes a two-dimensional visualization possible.

## Study Notes

### Key Ideas

- LDA is a **supervised feature-extraction** technique.
- It creates new features from linear combinations of the original variables.
- The new features are called **linear discriminants**.
- LDA uses the target labels during fitting.
- Its objective is to maximize between-class separation relative to within-class spread.
- LDA can reduce dimensions before a classifier is trained, but LDA itself can also be used as a classifier.
- LDA is most helpful when the labels contain meaningful class structure that can be separated with linear projections.

### Correct Preprocessing Workflow

To prevent data leakage, use the following order:

1. Split the data into training and test sets.
2. Fit any required preprocessing on the training set.
3. Fit LDA using `X_train` **and** `y_train`.
4. Transform both sets with that same fitted LDA object.
5. Train the downstream classifier on the transformed training set.
6. Evaluate it on the transformed test set.

Never fit LDA on the complete dataset before the train/test split. Because LDA uses both features and labels, doing so would leak information from the test set into the learned projection.

### LDA in scikit-learn

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
```

Notice that `fit_transform` receives both `X_train` and `y_train`. This reflects LDA's supervised nature. The test labels are not supplied to `transform`.

### Assumptions and Practical Considerations

Classical LDA works best when:

- The relationship between features and class separation is approximately linear.
- Each class is approximately normally distributed in feature space.
- The classes have similar covariance matrices.
- The observations are independent.

LDA can still work when these assumptions are not perfectly satisfied, but performance should be validated rather than assumed. Strong outliers, severe nonlinearity, highly imbalanced classes, or poorly estimated covariance matrices can reduce its effectiveness.

### Common Mistakes

- Calling LDA unsupervised or forgetting that it requires labels during fitting.
- Confusing **Linear Discriminant Analysis** with **Latent Dirichlet Allocation**, which also uses the abbreviation LDA but is a different technique.
- Assuming that LDA always produces the same number of components as PCA.
- Requesting more than `number of classes - 1` components.
- Fitting LDA before the train/test split.
- Assuming that maximum feature variance automatically means maximum class separation.
- Evaluating a dimensionality-reduction method only from a visualization instead of using test metrics or cross-validation.

## Quick Review Questions

1. Why is LDA considered a supervised technique?
2. What does LDA attempt to maximize and minimize?
3. How does the objective of LDA differ from that of PCA?
4. What do the within-class and between-class scatter matrices represent?
5. Why can a three-class problem produce at most two linear discriminants?
6. Why must LDA be fitted only on the training data?
7. When might PCA retain variance without separating the classes effectively?
8. What assumptions are commonly associated with classical LDA?

## Summary

LDA maps labeled observations into a lower-dimensional space by finding linear directions that keep each class compact while separating different classes as much as possible. Its use of the target labels is the defining distinction from PCA and makes it especially useful as a supervised preprocessing technique for classification.
