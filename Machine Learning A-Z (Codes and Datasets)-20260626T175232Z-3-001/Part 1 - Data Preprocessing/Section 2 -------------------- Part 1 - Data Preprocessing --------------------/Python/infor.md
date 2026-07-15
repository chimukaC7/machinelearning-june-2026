# Study Notes: Data Preprocessing for Machine Learning

## 1. Purpose of Data Preprocessing

Data preprocessing converts raw data into a form that machine learning algorithms can use effectively.

Real-world datasets often contain:

* Missing values
* Categorical variables
* Features measured on different scales
* Irrelevant or inconsistent data
* Variables that must be separated into inputs and outputs

The quality of preprocessing directly influences model accuracy, stability, training speed, and generalisation.

A typical preprocessing workflow is:

```text
Import libraries
      ↓
Import dataset
      ↓
Separate features and target
      ↓
Handle missing values
      ↓
Encode categorical variables
      ↓
Split into training and test sets
      ↓
Apply feature scaling where necessary
      ↓
Train the machine learning model
```

---

## 2. Importing the Required Libraries

The commonly used libraries for introductory machine learning implementations include:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### NumPy

NumPy supports numerical computing and array operations.

```python
import numpy as np
```

Typical uses include:

* Mathematical operations
* Numerical arrays
* Matrix manipulation
* Statistical calculations

### Pandas

Pandas is used for loading, exploring, cleaning, and transforming tabular data.

```python
import pandas as pd
```

Its main data structures are:

* `Series`: a one-dimensional labelled collection
* `DataFrame`: a two-dimensional table with rows and columns

### Matplotlib

Matplotlib is used to visualise data.

```python
import matplotlib.pyplot as plt
```

It supports plots such as:

* Line charts
* Scatter plots
* Histograms
* Bar charts
* Box plots

### Insight

Libraries should be imported only when they are needed. However, keeping frequently used imports in a reusable preprocessing template can improve consistency during learning and experimentation.

---

## 3. Importing the Dataset

In Google Colab, a dataset can be uploaded manually and then loaded using Pandas.

```python
dataset = pd.read_csv("Data.csv")
```

A CSV file is normally loaded into a Pandas `DataFrame`.

### Inspecting the Dataset

Always inspect the data before preprocessing it.

```python
dataset.head()
```

```python
dataset.info()
```

```python
dataset.describe()
```

Useful checks include:

```python
dataset.shape
dataset.columns
dataset.dtypes
dataset.isnull().sum()
dataset.duplicated().sum()
```

### Questions to Ask During Inspection

* How many rows and columns are present?
* Which column is the prediction target?
* Which columns are numerical?
* Which columns are categorical?
* Are any values missing?
* Are there duplicate records?
* Are there suspicious or impossible values?
* Are the classes balanced?

### Important Insight

Do not begin transforming data before understanding its meaning. A value that appears incorrect may be valid within the business or research context.

---

## 4. Separating Independent and Dependent Variables

A supervised machine learning dataset is commonly divided into:

* **Independent variables**, represented by `X`
* **Dependent variable**, represented by `y`

### Independent Variables

Independent variables are the predictors or features used by the model.

Examples include:

* Country
* Age
* Salary

### Dependent Variable

The dependent variable is the outcome the model must predict.

Examples include:

* Purchased: Yes or No
* House price
* Customer churn
* Disease diagnosis

### Example

```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

Here:

* `X` contains all columns except the final column
* `y` contains the final column

A more readable production approach is to select columns explicitly:

```python
X = dataset[["Country", "Age", "Salary"]]
y = dataset["Purchased"]
```

### Insight

Explicit column selection is safer than position-based selection because the code remains understandable and is less likely to break when the dataset structure changes.

---

## 5. Handling Missing Data

Missing data occurs when some observations do not contain values for particular variables.

Examples from the discussed dataset include:

* A missing age
* A missing salary

### Why Missing Values Matter

Many machine learning algorithms cannot operate directly on missing values.

Ignoring missing data may also:

* Reduce the number of usable observations
* Introduce bias
* Distort relationships between variables
* Produce unreliable predictions

---

## 6. Common Missing-Data Strategies

### 6.1 Removing Rows

Rows containing missing values may be removed.

```python
dataset = dataset.dropna()
```

This may be appropriate when:

* Only a very small number of rows are affected
* The missing values appear random
* The dataset is sufficiently large

### Limitation

Removing rows may discard valuable information and introduce bias, particularly when missingness follows a systematic pattern.

---

### 6.2 Removing Columns

A column may be removed when most of its values are missing or when it offers little predictive value.

```python
dataset = dataset.drop(columns=["ColumnName"])
```

This should be based on both:

* Statistical evidence
* Domain knowledge

---

### 6.3 Mean Imputation

Missing numerical values can be replaced by the mean.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
```

The imputer can then be fitted and applied:

```python
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
```

Mean imputation works reasonably well for approximately symmetric numerical data.

### Limitation

The mean is sensitive to extreme values.

---

### 6.4 Median Imputation

```python
imputer = SimpleImputer(strategy="median")
```

Median imputation is often preferable when the data is skewed or contains outliers.

---

### 6.5 Most-Frequent Imputation

```python
imputer = SimpleImputer(strategy="most_frequent")
```

This is commonly used for categorical variables.

---

### 6.6 Constant Imputation

```python
imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
```

This may be useful when a missing category has its own meaning.

---

## 7. The Meaning of `fit` and `transform`

Scikit-learn preprocessing tools commonly use two stages.

### `fit()`

The preprocessing object learns information from the data.

For a mean imputer, it calculates the mean.

```python
imputer.fit(X)
```

### `transform()`

The learned information is applied to the data.

```python
X_transformed = imputer.transform(X)
```

### `fit_transform()`

Both operations are performed together.

```python
X_transformed = imputer.fit_transform(X)
```

### Critical Insight

Preprocessors should normally be fitted using the training set only. Fitting them on the complete dataset allows information from the test set to influence training, causing **data leakage**.

---

## 8. Understanding Categorical Data

Categorical variables represent labels or groups rather than continuous measurements.

Examples include:

* Country: France, Spain, Germany
* Purchased: Yes, No

Most machine learning algorithms require numerical input. Categorical values must therefore be encoded.

There are two main categories:

### Nominal Variables

Categories have no natural order.

Examples:

* Country
* Department
* Colour

### Ordinal Variables

Categories have a meaningful order.

Examples:

* Low, Medium, High
* Junior, Intermediate, Senior
* Poor, Fair, Good, Excellent

The encoding strategy should reflect whether the variable is nominal or ordinal.

---

## 9. Encoding the Dependent Variable

A binary target such as `Yes` and `No` can be converted into numerical labels.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
```

This may produce:

```text
No  → 0
Yes → 1
```

### Appropriate Use

`LabelEncoder` is generally suitable for encoding a one-dimensional target variable.

### Caution

It should not normally be applied directly to nominal input features because the resulting numbers may imply a false order.

For example:

```text
France  → 0
Germany → 1
Spain   → 2
```

A model might incorrectly interpret Spain as being numerically greater than France.

---

## 10. Encoding Categorical Independent Variables

For nominal input variables, one-hot encoding is usually more appropriate.

### One-Hot Encoding

Each category becomes a separate binary column.

Original column:

| Country |
| ------- |
| France  |
| Spain   |
| Germany |

Encoded representation:

| France | Spain | Germany |
| -----: | ----: | ------: |
|      1 |     0 |       0 |
|      0 |     1 |       0 |
|      0 |     0 |       1 |

### Scikit-Learn Implementation

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

column_transformer = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"), [0])
    ],
    remainder="passthrough"
)

X = column_transformer.fit_transform(X)
```

### Why `handle_unknown="ignore"` Is Useful

A future dataset may contain a category that was not present during training. Ignoring unknown categories prevents the transformer from failing immediately.

### Insight

One-hot encoding avoids introducing an artificial numerical relationship between nominal categories.

---

## 11. Encoding Ordinal Variables

Ordinal values should be encoded according to their true order.

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)
```

This produces an ordered mapping:

```text
Low    → 0
Medium → 1
High   → 2
```

Unlike nominal encoding, this order has a legitimate interpretation.

---

## 12. Splitting the Dataset

The dataset should be divided into:

* A training set
* A test set

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Training Set

The model learns patterns and relationships from the training set.

### Test Set

The test set estimates how well the trained model performs on unseen observations.

### Common Split Ratios

* 80% training and 20% testing
* 75% training and 25% testing
* 70% training and 30% testing

The appropriate ratio depends on dataset size and modelling requirements.

---

## 13. Why `random_state` Matters

The split is random by default.

```python
random_state=42
```

Setting a fixed random state makes the split reproducible.

This means that repeated executions produce the same training and test observations.

### Insight

The exact number is not important. Consistency is what matters.

---

## 14. Stratified Splitting

For classification problems, the target classes may be unevenly distributed.

A stratified split preserves approximately the same class distribution in both training and test sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This is particularly important when dealing with:

* Fraud detection
* Disease classification
* Customer churn
* Rare-event prediction

---

## 15. Overfitting and Generalisation

### Overfitting

Overfitting occurs when a model learns the training data too closely, including noise and accidental patterns.

An overfitted model usually:

* Performs very well on training data
* Performs poorly on unseen data

### Underfitting

Underfitting occurs when the model is too simple to learn the underlying patterns.

An underfitted model performs poorly on both:

* Training data
* Test data

### Generalisation

Generalisation is the model's ability to perform well on new observations.

The test set helps estimate generalisation performance.

### Important Clarification

A test set does not prevent overfitting by itself. It helps detect whether the model generalises. Model selection and tuning should preferably use a validation set or cross-validation, while the test set should remain untouched until final evaluation.

---

## 16. Training, Validation and Test Sets

A stronger evaluation structure uses three partitions.

### Training Set

Used to fit the model.

### Validation Set

Used to:

* Select algorithms
* Tune hyperparameters
* Compare model configurations
* Choose preprocessing decisions

### Test Set

Used once for final, unbiased evaluation.

For small and medium datasets, cross-validation is often used instead of maintaining a separate validation set.

---

## 17. Feature Scaling

Feature scaling transforms numerical variables so that they have comparable magnitudes.

Consider:

```text
Age:      20–60
Salary:   20,000–150,000
```

Without scaling, salary has a much larger numerical magnitude than age.

Some algorithms may therefore allow salary to dominate calculations even when it is not more important.

---

## 18. Standardisation

Standardisation transforms a feature using:

[
z = \frac{x-\mu}{\sigma}
]

Where:

* (x) is the original value
* (\mu) is the feature mean
* (\sigma) is the standard deviation

Scikit-learn implementation:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

After standardisation, features usually have:

* Mean close to 0
* Standard deviation close to 1

---

## 19. Normalisation

Min-max normalisation scales values into a fixed interval, commonly between 0 and 1.

[
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
]

Implementation:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Standardisation vs Normalisation

| Method                | Typical result               | Common use                         |
| --------------------- | ---------------------------- | ---------------------------------- |
| Standardisation       | Mean 0, standard deviation 1 | General-purpose ML                 |
| Min-max normalisation | Usually between 0 and 1      | Neural networks and bounded inputs |

---

## 20. When Feature Scaling Is Important

Scaling is usually important for algorithms that depend on:

* Distance
* Feature magnitude
* Gradient-based optimisation
* Variance

Examples include:

* K-nearest neighbours
* K-means clustering
* Support vector machines
* Logistic regression
* Linear regression with regularisation
* Principal component analysis
* Neural networks

---

## 21. When Scaling Is Often Less Important

Tree-based models are generally less sensitive to feature scale.

Examples include:

* Decision trees
* Random forests
* Gradient-boosted trees
* XGBoost-style tree models

These models split features using thresholds rather than calculating distances between observations.

However, scaling may still be necessary when tree models are part of a broader pipeline containing scale-sensitive transformations.

---

## 22. Avoiding Data Leakage During Scaling

Incorrect:

```python
scaler.fit(X)
X_train, X_test = train_test_split(X)
```

This allows the scaler to learn information from observations that later appear in the test set.

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```

### Core Rule

```text
Fit preprocessing tools on training data.
Transform both training and test data using the fitted tool.
```

---

## 23. Correct Order of Preprocessing

A simplified teaching workflow may demonstrate preprocessing before splitting. In practice, the safer order is:

```text
1. Load and inspect the data
2. Separate X and y
3. Split into training and test sets
4. Fit imputers on X_train
5. Transform X_train and X_test
6. Fit encoders on X_train
7. Transform X_train and X_test
8. Fit scalers on X_train
9. Transform X_train and X_test
10. Train the model
11. Evaluate it on X_test
```

### Why This Matters

Any operation that learns statistics or category mappings should be fitted without access to the test data.

Examples include:

* Mean and median imputation
* One-hot encoding category discovery
* Standardisation
* Min-max scaling
* Feature selection
* Principal component analysis

---

## 24. Recommended Pipeline Approach

Scikit-learn pipelines combine preprocessing and modelling into a single repeatable workflow.

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

dataset = pd.read_csv("Data.csv")

X = dataset[["Country", "Age", "Salary"]]
y = dataset["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_features = ["Age", "Salary"]
categorical_features = ["Country"]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.2%}")
```

### Advantages of Pipelines

Pipelines:

* Reduce data leakage
* Keep preprocessing consistent
* Make code easier to reproduce
* Simplify cross-validation
* Prevent accidental omission of preprocessing steps
* Make deployment more reliable

---

## 25. Google Colab Workflow

A shared Colab notebook may be read-only. To edit it:

```text
File → Save a copy in Drive
```

The copied notebook can then be modified.

### Colab Cell Types

#### Code Cells

Contain executable Python code.

#### Text Cells

Contain:

* Explanations
* Headings
* Mathematical notation
* Instructions
* Documentation

When rebuilding an implementation from scratch, delete only the code cells while preserving the explanatory text and structure.

### Learning Insight

Re-implementing each code cell manually is more effective than simply reading completed code. Active coding improves recall, debugging ability, and understanding of how the steps connect.

---

# Conceptual Summary

## Independent and Dependent Variables

```text
Independent variables → inputs used to make predictions
Dependent variable    → output the model attempts to predict
```

## Training and Test Sets

```text
Training set → used to learn model parameters
Test set     → used to evaluate unseen-data performance
```

## Fitting and Transforming

```text
fit       → learn from data
transform → apply learned transformation
```

## Categorical Encoding

```text
Nominal target variable → label encoding may be suitable
Nominal input variable  → one-hot encoding
Ordinal input variable  → ordinal encoding with an explicit order
```

## Feature Scaling

```text
StandardScaler → centres and standardises features
MinMaxScaler   → maps features into a defined range
```

---

# Critical Study Insights

### 1. Preprocessing Is Part of the Model

Preprocessing should not be treated as an unrelated preliminary task. The fitted imputer, encoder and scaler are part of the complete predictive system.

### 2. The Test Set Must Remain Unseen

The test set should not influence:

* Imputation values
* Scaling statistics
* Feature selection
* Model tuning
* Algorithm selection

### 3. Not Every Algorithm Requires Scaling

Scaling should be selected according to the algorithm rather than applied automatically.

### 4. Missing Data Requires Investigation

Imputation is not merely a technical repair. The reason why data is missing may carry useful information.

### 5. Encoding Must Preserve Meaning

Nominal categories should not be assigned arbitrary ordered numbers. Ordinal variables should be encoded using their genuine order.

### 6. Reproducibility Matters

Fixed random seeds, pipelines and explicit column names make experiments easier to verify and repeat.

### 7. Accuracy Alone May Be Misleading

For imbalanced classification problems, consider metrics such as:

* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix

---

# Revision Questions

1. What is the purpose of data preprocessing?
2. What is the difference between an independent and dependent variable?
3. Why can missing values cause problems for machine learning algorithms?
4. When is median imputation preferable to mean imputation?
5. What is the difference between `fit()`, `transform()` and `fit_transform()`?
6. Why should nominal variables usually use one-hot encoding?
7. Why is label encoding potentially inappropriate for a country feature?
8. Why must a dataset be divided into training and test sets?
9. What is overfitting?
10. Does a test set prevent overfitting, or does it help detect it?
11. Why is `random_state` used?
12. When should stratified splitting be applied?
13. What is the difference between standardisation and normalisation?
14. Which machine learning algorithms are sensitive to feature scale?
15. Why are tree-based models generally less sensitive to scaling?
16. What is data leakage?
17. Why must an imputer or scaler be fitted only on the training data?
18. What are the advantages of using a Scikit-learn pipeline?
19. Why should preprocessing code use explicit column names where possible?
20. Why is rebuilding the implementation more valuable than merely reading it?

---

# Memory Aid

Use the mnemonic **I-I-M-E-S-S**:

```text
I → Import libraries
I → Import and inspect data
M → Manage missing values
E → Encode categorical variables
S → Split the dataset
S → Scale features where necessary
```

The practical rule to remember is:

> **Split first, fit preprocessing on the training set, and only transform the test set.**
