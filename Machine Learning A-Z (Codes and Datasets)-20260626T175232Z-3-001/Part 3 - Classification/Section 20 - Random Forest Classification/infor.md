# Random Forest Classification

Today, we are going to learn about random forests and introduce an important concept called **ensemble learning**.

## Ensemble Learning

Ensemble learning combines the predictions of multiple machine learning models to produce one stronger and more reliable final model. The individual models may use different algorithms, or they may be multiple instances of the same algorithm.

Random Forest is an ensemble method that combines many decision trees. Instead of training and relying on a single decision tree, it trains multiple trees and combines their predictions.

## How a Random Forest Works

The process can be summarized as follows:

1. Select a random sample of data points from the training set, usually by sampling with replacement.
2. Build a decision tree using that sample. At each split, consider a random subset of the available features.
3. Choose how many trees the forest should contain.
4. Repeat the sampling and tree-building process until all the trees have been trained.
5. When classifying a new observation, ask every tree to predict its class.
6. Assign the observation to the class that receives the majority of the trees' votes.

Each tree is trained on a different random sample of the data and considers different subsets of features. Consequently, the trees make somewhat different errors. Although an individual tree may not be ideal, combining many trees generally produces a more accurate and stable model.

This is similar to using the wisdom of a crowd. Rather than trusting one tree, the forest considers the predictions of many trees and selects the class supported by the majority. Combining predictions in this way can reduce variance, limit overfitting, and make the final classifier more robust.

## Real-World Application: Microsoft Kinect

A notable application of decision forests appeared in the development of Microsoft Kinect. Kinect used depth images and machine learning to estimate the locations of body parts, such as the head, arms, and hands, in real time. This allowed users to interact with games through body movement instead of a handheld controller.

The research paper **“Real-Time Human Pose Recognition in Parts from Single Depth Images”** describes how randomized decision forests were used for body-part classification. Their speed and accuracy made real-time pose estimation practical while helping to reduce the computing power—and therefore the hardware cost—required by the system.

Random forests are therefore not only a useful example of ensemble learning; they are also practical models that can provide strong performance on real-world classification problems.

---

# Study Notes

## Key Definitions

- **Ensemble learning:** Combining several models to obtain a stronger final prediction.
- **Decision tree:** A model that repeatedly splits data according to feature-based rules.
- **Random Forest:** An ensemble of decision trees trained with randomness in both the observations and the features.
- **Bootstrap sample:** A training sample created by randomly drawing observations **with replacement** from the original training set.
- **Majority voting:** The class predicted by the greatest number of trees becomes the forest's final classification.

## Random Forest Classification Workflow

```text
Training data
    ↓
Create multiple bootstrap samples
    ↓
Train one decision tree on each sample
    ↓
Use a random subset of features at each split
    ↓
Collect every tree's prediction
    ↓
Return the class with the majority vote
```

## Why Introduce Randomness?

Random Forest uses two main sources of randomness:

1. **Random training observations:** Each tree receives a different bootstrap sample.
2. **Random feature subsets:** Each split considers only some of the available features.

These choices reduce the correlation between trees. If every tree were trained identically and always considered the same strongest features, the trees would tend to make similar mistakes. A collection of diverse trees is more useful because their errors can partially cancel out when their predictions are combined.

## Advantages

- Often performs well without extensive tuning.
- Handles nonlinear relationships and complex decision boundaries.
- Usually overfits less than a single deep decision tree.
- Works with many input features.
- Requires no feature scaling for ordinary tree-based splitting.
- Can estimate feature importance.
- Can evaluate generalization using out-of-bag observations.

## Limitations

- Less interpretable than a single decision tree.
- Requires more memory and computation as the number of trees increases.
- Feature-importance measures can be misleading when predictors differ greatly in scale or number of categories.
- A large forest may predict more slowly than one decision tree.
- Like other supervised models, it still depends on representative, correctly labelled training data.

## Important Hyperparameters

| Hyperparameter | Purpose |
|---|---|
| `n_estimators` | Number of decision trees in the forest |
| `criterion` | Function used to measure split quality, such as Gini impurity or entropy |
| `max_depth` | Maximum depth of each tree |
| `min_samples_split` | Minimum number of samples required to split an internal node |
| `min_samples_leaf` | Minimum number of samples allowed in a leaf |
| `max_features` | Number or proportion of features considered at each split |
| `bootstrap` | Whether trees are trained on bootstrap samples |
| `class_weight` | Adjusts the importance of classes, which can help with imbalanced data |
| `random_state` | Makes randomized training reproducible |

## Out-of-Bag Evaluation

Because bootstrap sampling is performed with replacement, some training observations are not selected for a particular tree. These are called **out-of-bag (OOB) observations** for that tree.

An observation can be predicted using only the trees for which it was out of bag. Combining these predictions produces an OOB score, which provides an internal estimate of model performance without requiring a separate validation set. A final test set should still be retained for an unbiased evaluation.

## Classification vs. Regression

- **Random Forest Classification:** Combines class predictions using majority voting or averaged class probabilities.
- **Random Forest Regression:** Combines numerical predictions, typically by taking their mean.

## Single Decision Tree vs. Random Forest

| Single Decision Tree | Random Forest |
|---|---|
| One tree | Many trees |
| Easy to visualize and explain | Harder to interpret |
| Can have high variance | Usually has lower variance |
| More likely to overfit when deep | More resistant to overfitting |
| Fast and lightweight | More computationally expensive |

## Exam-Style Summary

A Random Forest is a **bagging-based ensemble** of decision trees. Each tree is trained on a bootstrap sample of the training data, and only a random subset of features is considered at each split. For classification, the final prediction is normally determined by majority vote. The combination of many diverse trees reduces variance and usually generalizes better than a single decision tree.

## Quick Review Questions

1. What is ensemble learning?
2. What two sources of randomness are used by a Random Forest?
3. How does a Random Forest produce its final classification?
4. Why is a Random Forest generally less prone to overfitting than one deep decision tree?
5. What is a bootstrap sample?
6. What are out-of-bag observations, and how are they useful?
7. How does prediction differ between Random Forest classification and regression?
