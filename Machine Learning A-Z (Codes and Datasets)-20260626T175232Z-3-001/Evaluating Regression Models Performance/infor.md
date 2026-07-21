# Evaluating Regression Models with R²

## What is R²?

**R²**, also called the **coefficient of determination**, measures how well a regression model explains the variation in the dependent variable (`y`).

It compares the model's predictions with a simple baseline that always predicts the mean of the observed target values.

## Key notation

- `yᵢ`: the actual value for observation `i`
- `ŷᵢ`: the value predicted by the regression model
- `ȳ`: the mean of all actual `y` values

## Residual Sum of Squares (RSS)

The residual for an observation is the vertical difference between its actual value and the model's prediction:

$$
e_i = y_i - \hat{y}_i
$$

The **Residual Sum of Squares** is:

$$
RSS = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

RSS represents the variation that the regression model does **not** explain. Ordinary Least Squares (OLS) chooses the model parameters that minimize this quantity.

- A small RSS indicates that predictions are close to the actual values.
- An RSS of `0` means every prediction matches its actual value exactly.

## Total Sum of Squares (TSS)

The **Total Sum of Squares** measures the total variation of the actual values around their mean:

$$
TSS = \sum_{i=1}^{n}(y_i - \bar{y})^2
$$

This corresponds to a baseline model that ignores all input features and predicts `ȳ` for every observation.

## R² formula

$$
R^2 = 1 - \frac{RSS}{TSS}
$$

R² expresses how much better the regression model performs than the mean-only baseline.

An equivalent interpretation is:

$$
R^2 = \frac{\text{variation explained by the model}}{\text{total variation in } y}
$$

For example, `R² = 0.80` means that the model explains approximately **80% of the variation** in the target variable in the evaluated data. It does not mean that the model's predictions are 80% accurate.

## Interpreting R² values

| R² value | General interpretation |
| ---: | --- |
| `1.00` | Perfect fit: all evaluated values are predicted exactly |
| `0.90` | The model explains 90% of the observed variation |
| `0.70` | The model explains 70% of the observed variation |
| `0.00` | The model is no better than always predicting the mean |
| `< 0.00` | The model performs worse than the mean-only baseline |

Higher R² values generally indicate a better fit, but there is **no universal threshold** for a good model. Acceptable values depend on the field, the amount of noise in the data, and the purpose of the model. A modest R² may be useful in noisy domains, while a very high R² may be expected in controlled physical systems.

## Why can R² be negative?

R² is negative when:

$$
RSS > TSS
$$

In that case, the regression model makes larger squared errors than the simple mean prediction. This commonly occurs when a model generalizes poorly to unseen test data. It can also occur when fitting certain models without an intercept.

For standard OLS regression **with an intercept**, R² on the training data lies between `0` and `1`. On test data, however, it can be negative.

## Worked example

Suppose:

$$
RSS = 20 \quad \text{and} \quad TSS = 100
$$

Then:

$$
R^2 = 1 - \frac{20}{100} = 0.80
$$

The model therefore explains 80% of the variation in `y`, leaving 20% unexplained on the evaluated data.

## Important limitations

- **R² does not prove causation.** A strong fit does not show that the predictors cause the outcome.
- **A high R² does not guarantee useful predictions.** The model may be overfitting the training data.
- **R² does not measure error in the original units.** Use metrics such as MAE or RMSE to understand prediction-error magnitude.
- **R² never decreases when predictors are added to a standard training regression**, even when the new predictors add little real value. Adjusted R² is more useful when comparing models with different numbers of predictors.
- **Always evaluate on unseen data.** Test-set R² or cross-validation gives a more realistic estimate of generalization performance.

## R² versus Adjusted R²

Adjusted R² penalizes the inclusion of predictors that do not improve the model enough:

$$
\bar{R}^2 = 1 - (1-R^2)\frac{n-1}{n-p-1}
$$

where:

- `n` is the number of observations.
- `p` is the number of predictors.

Use **R²** to describe explained variation and **Adjusted R²** when comparing regression models containing different numbers of predictors.

## Key takeaway

R² compares a regression model with a mean-only baseline. A value closer to `1` usually indicates that the model explains more of the target's variation, but R² should be interpreted in context and considered alongside validation results and error metrics such as MAE and RMSE.
