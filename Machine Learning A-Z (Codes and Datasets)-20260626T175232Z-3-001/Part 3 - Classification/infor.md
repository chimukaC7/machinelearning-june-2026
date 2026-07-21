# Part 3: Classification

## What is classification?

**Classification** is a supervised machine learning task used to predict a **category or class label**.

This differs from regression:

| Task | Predicted output | Example |
| --- | --- | --- |
| Regression | A continuous numerical value | Predicting a house price |
| Classification | A discrete class or category | Predicting whether an email is spam |

A classification model learns from observations whose correct labels are already known. After training, it assigns a label to a new observation and may also estimate the probability or confidence associated with that label.

## Types of classification

- **Binary classification:** predicts one of two classes, such as `fraud` or `not fraud`.
- **Multiclass classification:** predicts one of more than two mutually exclusive classes, such as a handwritten digit from `0` to `9`.
- **Multilabel classification:** may assign several labels to one observation, such as tagging an image with `beach`, `sunset`, and `people`.

## Common applications

Classification is used across many domains, including:

- **Medicine:** identifying whether a tumour is benign or malignant.
- **Banking:** detecting fraudulent transactions or estimating default risk.
- **Marketing:** predicting whether a customer will respond to a campaign.
- **Customer retention:** identifying customers who may leave a service.
- **Email filtering:** classifying messages as spam or legitimate.
- **Computer vision:** recognizing objects, faces, or handwritten digits.
- **Manufacturing:** categorizing products as defective or acceptable.

## Classification models covered in this section

### 1. Logistic Regression

Despite its name, Logistic Regression is a **classification algorithm**. It models the probability that an observation belongs to a class and applies a decision threshold to produce the final label.

Key characteristics:

- Produces interpretable coefficients and class probabilities.
- Learns a linear decision boundary in the original feature space.
- Provides a strong, simple baseline for binary classification.
- Can be extended to multiclass problems.

### 2. K-Nearest Neighbors (K-NN)

K-NN predicts the class of a new observation by examining the labels of its `k` nearest training observations. The class receiving the most votes is usually selected.

Key characteristics:

- Simple and intuitive.
- Can learn nonlinear decision boundaries.
- Requires little explicit training but can be slow during prediction.
- Is sensitive to feature scales, the choice of `k`, and the distance metric.

### 3. Support Vector Machine (SVM)

A linear SVM finds the separating hyperplane that creates the largest possible margin between classes. The training observations closest to the boundary are called **support vectors**.

Key characteristics:

- Often effective in high-dimensional feature spaces.
- Focuses on the observations that define the class boundary.
- Uses a linear decision boundary unless combined with a kernel.
- Is sensitive to feature scaling and regularization settings.

### 4. Kernel SVM

Kernel SVM extends SVM to problems that are not linearly separable. A kernel function allows the algorithm to construct a nonlinear decision boundary without explicitly calculating every coordinate in a higher-dimensional feature space. This is known as the **kernel trick**.

Common kernels include:

- Linear
- Polynomial
- Radial Basis Function (RBF)
- Sigmoid

### 5. Naive Bayes

Naive Bayes applies Bayes' theorem and assumes that the predictors are conditionally independent given the class. Although this assumption is often unrealistic, the method performs surprisingly well in many applications.

Key characteristics:

- Fast to train and predict.
- Works well with high-dimensional data.
- Is commonly used for text and document classification.
- Has variants for different feature distributions, including Gaussian, Multinomial, and Bernoulli Naive Bayes.

### 6. Decision Tree Classification

A Decision Tree repeatedly divides the feature space using a sequence of decision rules. Each internal node represents a test, each branch represents an outcome, and each leaf represents a predicted class.

Key characteristics:

- Easy to visualize and explain.
- Captures nonlinear relationships and feature interactions.
- Usually does not require feature scaling.
- Can overfit if the tree grows too deep or is insufficiently constrained.

### 7. Random Forest Classification

A Random Forest combines many Decision Trees. Each tree is trained using a random sample of the observations and a random subset of features. The forest usually selects its final class through majority voting.

Key characteristics:

- Captures nonlinear patterns and interactions.
- Usually generalizes better than a single unrestricted Decision Tree.
- Is robust and effective across many tabular-data problems.
- Is less interpretable and more computationally expensive than one tree.

## Linear and nonlinear classifiers

A classifier is described as linear or nonlinear according to the shape of its decision boundary in the feature space.

- **Typically linear:** Logistic Regression and linear SVM.
- **Able to learn nonlinear boundaries:** K-NN, Kernel SVM, Decision Trees, and Random Forests.
- **Depends on its distributional assumptions:** Naive Bayes.

## General classification workflow

1. Identify the features `X` and categorical target `y`.
2. Split the data into training and test sets.
3. Preprocess the data, including encoding categories and handling missing values.
4. Scale features when required by the selected algorithm.
5. Train the classifier using the training set.
6. Predict classes or class probabilities for unseen observations.
7. Evaluate the model with metrics appropriate to the problem.
8. Tune hyperparameters using validation data or cross-validation.

## Evaluating a classifier

Accuracy alone can be misleading, especially when the classes are imbalanced. Useful evaluation tools include:

- **Confusion matrix:** summarizes correct and incorrect predictions by class.
- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of predicted positives that are truly positive.
- **Recall (sensitivity):** proportion of actual positives detected by the model.
- **Specificity:** proportion of actual negatives correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** measures how well a binary classifier ranks positive cases above negative ones across different thresholds.

The best metric depends on the cost of each type of error. For example, recall may be especially important when failing to detect a disease is more costly than producing a false alarm.

## Key takeaway

Classification predicts discrete categories rather than continuous values. No single classifier is best for every dataset: model choice should reflect the shape and size of the data, interpretability requirements, computational constraints, and the real-world cost of prediction errors.
