# Training-Test Split and Feature Scaling
A **training-test split** is important in machine learning because it helps you check whether your model has actually learned useful patterns, or whether it has simply memorised the data it was trained on.

In machine learning, we usually divide the available dataset into at least two parts:

**Training set**: used to teach the model.

**Test set**: used to evaluate how well the model performs on new, unseen data.

For example, if you have 10,000 records, you might use 80% for training and 20% for testing.

```text
Dataset
 ├── Training data: 80%
 └── Test data: 20%
```

The main importance is that it gives a realistic estimate of model performance. A model may perform very well on the data it has already seen, but fail badly when given new data. The test set helps simulate real-world use, where the model will receive data it has never seen before.

It also helps detect **overfitting**. Overfitting happens when a model learns the training data too closely, including noise and random patterns, instead of learning general rules. Such a model may show high accuracy on the training data but poor accuracy on the test data.

For example:

```text
Training accuracy: 98%
Test accuracy: 65%
```

This suggests the model has probably overfitted.

Training-test splitting also supports fair comparison between models. If you train several models, such as logistic regression, decision tree, random forest, or neural network, you can test all of them on the same test set. This makes the comparison more reliable.

It also helps avoid misleading results. If you train and test using the same data, the model may appear highly accurate, but that accuracy is not trustworthy because the model has already seen the answers.

A typical split may be:

```text
70% training / 30% testing
80% training / 20% testing
90% training / 10% testing
```

For small datasets, cross-validation is often better because it allows the model to be trained and tested on different portions of the data multiple times.

In simple terms, the training-test split is important because it answers this question:

> Can the model perform well on data it has never seen before?

That is the real measure of a useful machine learning model.



# Feature Scaling
**Feature scaling** is the process of putting numerical features into a similar range so that one feature does not dominate others simply because it has larger values.

For example, suppose you are training a model using these features:

```text
Age:              20 to 70
Monthly income:   2,000 to 50,000
Years employed:   1 to 30
```

The income values are much larger than age or years employed. Some machine learning algorithms may treat income as more important just because the numbers are larger, even when that is not logically true. Feature scaling fixes this.

## Common types of feature scaling

### 1. Normalization

Normalization usually scales values between **0 and 1**.

Formula:

```text
X_scaled = (X - X_min) / (X_max - X_min)
```

Example:

```text
Original salary values: 2,000, 10,000, 50,000

After normalization:   0.00, 0.17, 1.00
```

This is useful when you want all values in a fixed range.

### 2. Standardization

Standardization transforms values so that the feature has a mean of **0** and a standard deviation of **1**.

Formula:

```text
X_scaled = (X - mean) / standard deviation
```

Example:

```text
Original age values: 20, 40, 60

After standardization: approximately -1.22, 0, 1.22
```

This is commonly used in many machine learning models.

## Where feature scaling applies

Feature scaling is especially important for algorithms that depend on **distance, gradients, or magnitude**.

It applies strongly to:

| Algorithm                                 | Why scaling matters                          |
| ----------------------------------------- | -------------------------------------------- |
| **K-Nearest Neighbours**                  | Uses distance between data points            |
| **K-Means Clustering**                    | Uses distance to group similar records       |
| **Support Vector Machines**               | Sensitive to feature magnitude               |
| **Linear Regression with regularization** | Helps regularization treat features fairly   |
| **Logistic Regression**                   | Improves convergence during training         |
| **Neural Networks**                       | Helps training become faster and more stable |
| **Principal Component Analysis**          | Large-scale features can dominate components |

Example: in KNN, if one feature is salary and another is age, salary may dominate the distance calculation unless scaling is applied.

```text
Without scaling:
Salary difference = 20,000
Age difference    = 5

The model focuses mostly on salary.

With scaling:
Salary difference = 0.4
Age difference    = 0.2

Both features are treated more fairly.
```

## Where feature scaling is usually not necessary

Feature scaling is usually **not very important** for tree-based models because they split data based on thresholds rather than distances.

Examples:

```text
Decision Tree
Random Forest
Gradient Boosting
XGBoost
LightGBM
CatBoost
```

For these models, scaling usually does not significantly change the result.

## Important rule

Always apply feature scaling **after the training-test split**, not before.

Correct approach:

```text
1. Split data into training and test sets
2. Fit the scaler on the training set only
3. Transform both training and test sets using that scaler
4. Train the model
5. Evaluate on the test set
```

This avoids **data leakage**, where information from the test set accidentally influences the training process.

In simple terms, feature scaling helps machine learning models learn fairly and efficiently when the size of the numbers matters.


# Replacing missing values in machine learning
**Replacing missing values** in machine learning is called **missing value imputation**. It means filling empty, null, or unavailable values in a dataset before training a model.

Missing values can appear as:

```text
NaN
NULL
None
Blank cells
?
0 where 0 actually means unknown
```

Example:

| Name  | Age | Salary | Department |
| ----- | --: | -----: | ---------- |
| John  |  32 |  8,000 | IT         |
| Mary  |     | 12,000 | Finance    |
| Peter |  41 |        | HR         |

Here, Mary’s age is missing and Peter’s salary is missing. Before training many machine learning models, we need to handle those missing values.

## Why replacing missing values is important

Many machine learning algorithms cannot work properly when the dataset contains missing values. If missing data is not handled, the model may fail during training or produce unreliable results.

It is important because it helps:

1. **Prevent model errors**

Some algorithms do not accept missing values at all. For example, linear regression, logistic regression, SVM, KNN, and neural networks usually require complete numerical input.

2. **Avoid losing useful data**

One option is to delete rows with missing values, but that can reduce the dataset too much.

Example:

```text
Dataset size: 1,000 records
Rows with missing values: 300 records
```

If you delete all missing rows, you lose 30% of your data. Replacing missing values allows you to keep more information.

3. **Improve model performance**

A good missing-value strategy can make the model more stable and accurate. Poor handling can introduce bias or distort patterns in the data.

## Common ways of replacing missing values

### 1. Replace with the mean

Use the average value of the column.

Example:

```text
Ages: 20, 30, 40, missing
Mean = 30

Missing value becomes 30
```

This works well for numerical data without extreme outliers.

Applies to:

```text
Age
Temperature
Exam scores
Income
Sensor readings
```

### 2. Replace with the median

Use the middle value of the column.

Example:

```text
Salaries: 3,000, 5,000, 100,000, missing
Median = 5,000

Missing value becomes 5,000
```

The median is better than the mean when the data has outliers.

For example, salary data often has outliers, so median is usually safer than mean.

### 3. Replace with the mode

Use the most frequent value.

Example:

```text
Department: IT, Finance, IT, HR, missing
Mode = IT

Missing value becomes IT
```

This is commonly used for categorical data.

Applies to:

```text
Gender
Department
Marital status
Product category
Education level
Vehicle type
```

### 4. Replace with a constant value

You can replace missing values with a fixed value such as:

```text
Unknown
Not Provided
0
-1
Missing
```

Example:

```text
Customer segment: missing

Replace with: Unknown
```

This is useful when the fact that a value is missing has meaning.

For example, if a customer did not provide employment status, replacing it with `"Not Provided"` may preserve useful information.

### 5. Forward fill or backward fill

This is common in time-series data.

Example:

| Date      | Temperature |
| --------- | ----------: |
| Monday    |          24 |
| Tuesday   |             |
| Wednesday |          26 |

Using forward fill:

```text
Tuesday = Monday’s value = 24
```

Applies to:

```text
Stock prices
Weather records
IoT sensor readings
Energy consumption
Network monitoring data
GPS tracking data
```

### 6. Predict missing values using a model

A more advanced method is to train another model to predict the missing values.

For example, if salary is missing, we can predict it using:

```text
Age
Job title
Education level
Years of experience
Department
```

This can be useful when missing values are many and the relationships between features are important.

## Where replacing missing values applies

Missing value replacement applies during **data preprocessing**, before training the machine learning model.

Typical machine learning workflow:

```text
1. Collect data
2. Clean data
3. Handle missing values
4. Encode categorical variables
5. Scale numerical features
6. Split into training and test sets
7. Train model
8. Evaluate model
```

It applies in many areas, such as:

| Area                       | Example missing values                               |
| -------------------------- | ---------------------------------------------------- |
| **Healthcare**             | Missing age, blood pressure, diagnosis history       |
| **Finance**                | Missing income, credit score, employment status      |
| **Education**              | Missing exam scores, attendance, school type         |
| **Telecoms**               | Missing usage records, location, customer details    |
| **IoT systems**            | Missing sensor readings due to device failure        |
| **Fleet management**       | Missing GPS coordinates, mileage, driver details     |
| **Human resource systems** | Missing qualifications, department, employment grade |

## Important rule

Handle missing values **after splitting the dataset** into training and test sets.

Correct approach:

```text
1. Split data into training and test sets
2. Calculate mean, median, or mode from the training set only
3. Use those values to fill missing data in the training set
4. Use the same values to fill missing data in the test set
```

This prevents **data leakage**.

Wrong approach:

```text
Fill missing values using the full dataset before splitting
```

That allows information from the test set to influence the training process, which can make the model performance look better than it really is.

## Simple example

Suppose you have this training data:

```text
Age: 20, 30, 40, missing
```

Mean age:

```text
(20 + 30 + 40) / 3 = 30
```

So the missing value is replaced with:

```text
30
```

Final data:

```text
Age: 20, 30, 40, 30
```

## In simple terms

Replacing missing values helps prepare incomplete data so that machine learning models can learn from it properly.

The best method depends on the type of data:

```text
Numerical data without outliers → Mean
Numerical data with outliers → Median
Categorical data → Mode or Unknown
Time-series data → Forward fill, backward fill, or interpolation
Complex relationships → Predict missing values using another model
```

So, missing value replacement applies mainly during **data cleaning and preprocessing**, before model training.
