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

---

# Adjusted R²

## Why ordinary R² can be misleading

Ordinary R² is a useful goodness-of-fit measure:

$$
R^2 = 1 - \frac{RSS}{TSS}
$$

where:

- `RSS` is the Residual Sum of Squares.
- `TSS` is the Total Sum of Squares.

A larger R² generally indicates a closer fit to the evaluated data. However, the meaning of a particular value depends on the domain and use case. For example, `R² = 0.40` may be weak in one field but valuable in a noisy field where outcomes are difficult to predict.

The main limitation of ordinary R² appears when predictors are added to a model.

Suppose the original multiple linear regression model is:

$$
\hat{y} = b_0 + b_1x_1 + b_2x_2
$$

and a third predictor is added:

$$
\hat{y} = b_0 + b_1x_1 + b_2x_2 + b_3x_3
$$

For OLS regression fitted to the same training observations and containing an intercept:

- `TSS` remains unchanged because it depends only on the actual `y` values and their mean.
- `RSS` can decrease or remain unchanged, but it cannot increase.
- Consequently, training R² can increase or remain unchanged, but it cannot decrease.

## Why RSS cannot increase

Ordinary Least Squares selects the coefficients that minimize RSS. After adding `x₃`, the model has two possibilities:

1. If `x₃` helps explain the target, OLS can choose a non-zero value for `b₃`, reduce RSS, and increase R².
2. If `x₃` does not help, the enlarged model can reproduce the old model by setting `b₃ = 0`. RSS then remains unchanged.

In practice, even an irrelevant predictor may slightly reduce training RSS because of random correlations in the sample. Ordinary R² rewards this small improvement without accounting for the added model complexity. As a result, repeatedly adding predictors can make a model appear better even when those predictors provide little genuine value.

## Adjusted R² formula

Adjusted R² corrects this weakness by penalizing the model for each additional predictor:

$$
R^2_{\text{adj}} = 1 - (1-R^2)\frac{n-1}{n-k-1}
$$

where:

- `R²` is the ordinary coefficient of determination.
- `n` is the number of observations in the sample.
- `k` is the number of independent variables, excluding the intercept.

The formula requires `n > k + 1`.

## How the penalty works

When another predictor is added, `k` increases and the denominator `n - k - 1` becomes smaller. This makes the penalty larger.

Therefore:

- Adjusted R² **increases** only when the new predictor improves ordinary R² enough to overcome the penalty.
- Adjusted R² **decreases** when the predictor adds too little explanatory value.
- Adjusted R² may **remain nearly unchanged** when the improvement and penalty are approximately balanced.

Unlike ordinary R², Adjusted R² can decrease when predictors are added. It can also be negative when the fitted model performs poorly relative to the mean-only baseline after accounting for its complexity.

## Worked example

Assume a model has:

- `n = 100` observations
- `k = 2` predictors
- `R² = 0.80`

Then:

$$
R^2_{\text{adj}}
= 1 - (1-0.80)\frac{100-1}{100-2-1}
$$

$$
R^2_{\text{adj}}
= 1 - 0.20\left(\frac{99}{97}\right)
\approx 0.796
$$

The Adjusted R² is slightly lower than the ordinary R² because it accounts for the two predictors used by the model.

Suppose a third predictor raises R² only slightly, from `0.800` to `0.801`:

$$
R^2_{\text{adj}}
= 1 - (1-0.801)\frac{99}{96}
\approx 0.795
$$

Although ordinary R² increased, Adjusted R² decreased. The new predictor did not improve the fit enough to justify its inclusion.

## R² versus Adjusted R²

| Measure | Effect of adding a predictor | Best use |
| --- | --- | --- |
| R² | Never decreases on the same OLS training data | Describing the proportion of variation explained by one fitted model |
| Adjusted R² | Can increase or decrease | Comparing models with different numbers of predictors |

When comparing Adjusted R² values, the models should use the **same dependent variable and the same set of observations**. Values from different datasets are not directly comparable.

## Important limitations

- Adjusted R² discourages unnecessary predictors, but it does not guarantee that the selected model will generalize well.
- It does not establish causation or prove that a predictor is statistically significant.
- It does not account for every type of model complexity, such as extensive feature selection or hyperparameter tuning.
- It should not replace evaluation on unseen data.
- Cross-validation and test-set metrics such as R², MAE, and RMSE are better indicators of predictive performance.

## Key takeaway

Ordinary R² rewards every reduction in training error, even a tiny reduction caused by an irrelevant predictor. Adjusted R² introduces a penalty for additional predictors, so a new variable is worthwhile only when it improves the model enough to justify the added complexity.
