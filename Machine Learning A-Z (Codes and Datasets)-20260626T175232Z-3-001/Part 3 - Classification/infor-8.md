# Choosing and Improving Classification Models

In Part 3, we studied seven classification models:

1. Logistic Regression
2. K-Nearest Neighbors (K-NN)
3. Support Vector Machine (SVM)
4. Kernel SVM
5. Naive Bayes
6. Decision Tree Classification
7. Random Forest Classification

After learning this many algorithms, three natural questions arise:

1. What are the advantages and disadvantages of each model?
2. How do I choose the right model for a problem?
3. How can I improve a model's performance?

## 1. Advantages and Disadvantages

### Logistic Regression

**Advantages**

- Simple, fast, and easy to implement.
- Produces class-probability estimates.
- Coefficients can help explain the direction of feature effects.
- Works well when the log-odds are approximately a linear function of the features.
- Supports regularization to control overfitting.

**Disadvantages**

- Has a linear decision boundary unless nonlinear features are engineered.
- Can be affected by multicollinearity and influential observations.
- May underfit complex relationships.
- Coefficient interpretation requires care, especially after feature transformations.

### K-Nearest Neighbors (K-NN)

**Advantages**

- Simple and intuitive.
- Makes few assumptions about the shape of the decision boundary.
- Can model complex, nonlinear class boundaries.
- Naturally supports multiclass classification.

**Disadvantages**

- Prediction can be slow and memory-intensive because the training data must be retained.
- Sensitive to feature scale, irrelevant features, noise, and the choice of `k`.
- Performance often deteriorates in high-dimensional spaces—the curse of dimensionality.
- Requires a meaningful distance metric.

### Support Vector Machine (SVM)

**Advantages**

- Can perform well in high-dimensional spaces.
- Maximizes the margin between classes.
- Regularization helps control the bias–variance trade-off.
- Uses only the support vectors to define the decision boundary.

**Disadvantages**

- Usually requires feature scaling.
- Can be computationally expensive on large datasets.
- Results may be sensitive to the value of `C`.
- Does not naturally produce calibrated probabilities.
- The final model is less interpretable than Logistic Regression or a small Decision Tree.

### Kernel SVM

**Advantages**

- Models nonlinear decision boundaries through the kernel trick.
- Can be highly effective on small or medium-sized datasets with complex boundaries.
- Offers several kernels for different similarity structures.

**Disadvantages**

- Training and prediction may become slow as the dataset grows.
- Requires careful tuning of the kernel and parameters such as `C` and `gamma`.
- Requires feature scaling.
- Is difficult to interpret and debug.
- Can overfit when the kernel is too flexible or poorly tuned.

### Naive Bayes

**Advantages**

- Very fast to train and predict.
- Works well with high-dimensional data, particularly text features.
- Performs well even with relatively small training datasets.
- Produces class-probability estimates.
- Naturally supports multiclass classification.

**Disadvantages**

- Assumes that features are conditionally independent given the class—an assumption that is often unrealistic.
- Probability estimates may be poorly calibrated.
- The chosen variant must match the feature distribution, such as Gaussian, Multinomial, or Bernoulli Naive Bayes.
- Strongly correlated features may distort the evidence.

### Decision Tree

**Advantages**

- Easy to visualize and explain when the tree is small.
- Captures nonlinear relationships and feature interactions.
- Requires little preprocessing and normally no feature scaling.
- Can work with numerical and appropriately encoded categorical data.

**Disadvantages**

- A deep tree can overfit easily.
- Small changes in the data may produce a very different tree.
- Greedy split selection does not guarantee the globally optimal tree.
- Axis-aligned splits may require a complex tree for some boundaries.

### Random Forest

**Advantages**

- Usually generalizes better than a single Decision Tree.
- Reduces variance by averaging many decorrelated trees.
- Captures nonlinear relationships and interactions.
- Requires little preprocessing and normally no feature scaling.
- Can provide out-of-bag evaluation and feature-importance estimates.

**Disadvantages**

- Less interpretable than a single Decision Tree.
- Requires more computation and memory.
- Prediction may be slower when the forest contains many trees.
- Some feature-importance methods can be biased or misleading.
- Probability estimates may need calibration.

## 2. Choosing a Model

A useful starting point is to consider whether a relatively simple linear decision boundary is likely to describe the problem.

- For a **linear or approximately linear problem**, begin with Logistic Regression or a linear SVM.
- For a **nonlinear problem**, consider K-NN, Kernel SVM, Naive Bayes, Decision Tree, or Random Forest.

This is only an initial shortlist—not a final decision rule. Real datasets rarely reveal in advance which algorithm will work best. Compare promising models using the same preprocessing, data splits, cross-validation strategy, and evaluation metrics.

### Business and Operational Considerations

- Use **Logistic Regression** when you need a strong, interpretable baseline or meaningful probability estimates. If decisions depend on probability ranking, verify probability calibration.
- Use **Naive Bayes** for fast probabilistic classification, especially with text or high-dimensional sparse features. Do not assume that its raw probabilities are calibrated.
- Use a **linear SVM** for high-dimensional classification when margin-based separation is useful and probability output is not essential.
- Use a **Kernel SVM** when the dataset is not extremely large and the class boundary is complex.
- Use **K-NN** when local similarity is meaningful and prediction latency and dimensionality are manageable.
- Use a **Decision Tree** when transparent, rule-based explanations are a priority.
- Use a **Random Forest** when strong predictive performance, robustness, and limited preprocessing are more important than simple interpretation.

Customer segmentation itself is normally an unsupervised clustering problem. An SVM can classify new customers into previously defined segments only when labelled segment assignments already exist.

### Validate the Choice

Use **k-fold cross-validation** to estimate how well each candidate generalizes to unseen data. For classification, stratified folds are often appropriate because they preserve class proportions more reliably.

Choose an evaluation metric that matches the real objective:

- Accuracy for reasonably balanced classes with similar error costs.
- Precision when false positives are particularly costly.
- Recall when false negatives are particularly costly.
- F1 score when both precision and recall matter.
- Balanced accuracy or precision–recall metrics for imbalanced data.
- Log loss or Brier score when the quality of probability estimates matters.

Also consider interpretability, training time, prediction latency, memory, scalability, fairness, calibration, and the cost of collecting each feature.

## 3. Improving Model Performance

A model contains two broad kinds of quantities:

- **Learned parameters:** Values estimated during training, such as Logistic Regression coefficients or the split rules in a Decision Tree.
- **Hyperparameters:** Settings chosen before training, such as the regularization strength, number of neighbors, kernel settings, tree depth, or number of trees.

Default hyperparameter values are useful starting points, but they are not guaranteed to be optimal for a particular dataset. **Hyperparameter tuning** searches for settings that improve validation performance.

Common tuning approaches include:

- Grid search
- Randomized search
- Bayesian optimization

The search must be evaluated with cross-validation and performed without leaking information from the test set. Preprocessing steps should be fitted inside the cross-validation pipeline. The final untouched test set should be used only after model and hyperparameter selection.

---

# Study Notes

## Seven-Model Comparison

| Model | Boundary | Scaling? | Interpretability | Typical strength |
|---|---|---:|---|---|
| Logistic Regression | Linear by default | Recommended | High | Baseline and probability modelling |
| K-NN | Nonlinear | Required | Medium | Local similarity patterns |
| Linear SVM | Linear | Required | Low–medium | High-dimensional separation |
| Kernel SVM | Nonlinear | Required | Low | Complex boundaries on moderate data |
| Naive Bayes | Distribution-dependent | Variant-dependent | Medium | Fast text and sparse-data classification |
| Decision Tree | Nonlinear | Not normally needed | High when small | Explainable rules and interactions |
| Random Forest | Nonlinear | Not normally needed | Low–medium | Robust general-purpose performance |

## Representative Hyperparameters

| Model | Important hyperparameters |
|---|---|
| Logistic Regression | Regularization strength, penalty, solver, class weights |
| K-NN | Number of neighbors, distance metric, neighbor weights |
| SVM | `C`, class weights |
| Kernel SVM | `C`, kernel, `gamma`, polynomial degree |
| Naive Bayes | Variant-specific smoothing or variance settings |
| Decision Tree | Maximum depth, split criterion, minimum samples per split or leaf, maximum features |
| Random Forest | Number of trees, maximum depth, maximum features, minimum leaf size, bootstrap setting |

Parameter names and meanings depend on the software implementation.

## Linear vs. Nonlinear Models

A model is not chosen solely by the appearance of a two-dimensional plot:

- A linear model can represent nonlinear patterns when suitable transformed features are supplied.
- A flexible nonlinear model may overfit noise.
- Cross-validation provides stronger evidence than visual intuition alone.
- A simpler model is often preferable when its performance is comparable and it is easier to explain, maintain, and deploy.

## Recommended Selection Workflow

```text
Define the prediction target and business cost
                    ↓
Choose suitable evaluation metrics
                    ↓
Create training and untouched test sets
                    ↓
Build a preprocessing-and-model pipeline
                    ↓
Establish a simple baseline
                    ↓
Compare several candidate algorithms
                    ↓
Tune promising models with cross-validation
                    ↓
Check calibration, errors, stability, and fairness
                    ↓
Evaluate the final choice once on the test set
```

## Preprocessing Reminders

- Scale features for K-NN and SVM-based models.
- Logistic Regression often benefits from scaling, especially with regularization.
- Ordinary Decision Trees and Random Forests do not normally require scaling.
- Handle missing values without using information from validation or test data.
- Encode categorical variables appropriately.
- Fit preprocessing only on training folds by using a pipeline.

## Model Improvement Is More Than Tuning

Hyperparameter tuning is only one way to improve performance. Often, larger gains come from:

- Correcting label and data-quality problems.
- Collecting more representative observations.
- Engineering informative features.
- Removing data leakage.
- Selecting a more appropriate metric.
- Handling class imbalance.
- Adjusting the decision threshold.
- Calibrating predicted probabilities.
- Performing error analysis on important subgroups.

## Probability Ranking and Calibration

Ranking and calibration are related but different:

- **Ranking quality** asks whether higher-risk observations generally receive higher scores.
- **Calibration** asks whether a predicted probability such as 70% corresponds to an event rate close to 70%.

A model may rank observations well while producing poorly calibrated probabilities. If business decisions depend on probability values, evaluate both discrimination and calibration.

## Cross-Validation Essentials

- **k-fold cross-validation:** Divide the training data into `k` folds, train on `k − 1`, validate on the remaining fold, and repeat.
- **Stratified k-fold:** Preserves approximately the same class proportions in every fold.
- **Nested cross-validation:** Separates hyperparameter selection from performance estimation when a particularly rigorous estimate is required.
- **Time-series or grouped data:** Require specialized splits; ordinary random folds may leak information.

## Common Mistakes

- Choosing a model using training accuracy.
- Tuning hyperparameters on the test set.
- Scaling the full dataset before cross-validation.
- Comparing models with different data splits.
- Using accuracy alone for imbalanced classes.
- Assuming all probability outputs are calibrated.
- Selecting the most complex model when a simpler model performs similarly.
- Ignoring deployment constraints and the real costs of errors.

## Exam-Style Summary

The seven classification models differ in flexibility, assumptions, computational cost, interpretability, and preprocessing requirements. Linear problems can begin with Logistic Regression or a linear SVM, while nonlinear candidates include K-NN, Kernel SVM, Naive Bayes, Decision Trees, and Random Forests. However, algorithm selection should ultimately be based on consistent cross-validation, appropriate metrics, and operational requirements. Model performance can be improved through sound data preparation, feature engineering, hyperparameter tuning, threshold selection, and probability calibration while keeping the test set untouched until final evaluation.

## Quick Review Questions

1. Which seven classification models were introduced in Part 3?
2. Which models normally require feature scaling?
3. Why is “linear versus nonlinear” only a starting point for model selection?
4. When might a Decision Tree be preferred over a Random Forest?
5. What is the difference between a learned parameter and a hyperparameter?
6. Why should preprocessing be performed inside a cross-validation pipeline?
7. What is the difference between probability ranking and calibration?
8. Why must the test set remain untouched during model selection?
