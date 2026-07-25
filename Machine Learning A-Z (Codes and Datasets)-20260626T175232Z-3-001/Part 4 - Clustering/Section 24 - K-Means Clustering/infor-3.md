# The Elbow Method

The **Elbow Method** is a visual technique used to help select a suitable number of clusters, \(K\), for K-Means clustering.

Although clustering examples are often illustrated in two dimensions, K-Means is not limited to two-dimensional data. It can work with datasets containing many features. Two-dimensional examples are simply easier to visualize.

## Why Is the Elbow Method Needed?

K-Means requires the number of clusters to be specified before the algorithm is fitted. In some problems, domain knowledge may suggest an appropriate value of \(K\). When this information is unavailable, the Elbow Method provides a useful starting point.

The method compares several K-Means solutions using the **Within-Cluster Sum of Squares (WCSS)**.

## Within-Cluster Sum of Squares

WCSS measures how tightly the observations are grouped around their assigned centroids. It is calculated by finding the squared distance between every observation and the centroid of its cluster, then adding all those squared distances:

\[
\text{WCSS} =
\sum_{j=1}^{K}
\sum_{\mathbf{x}_i \in C_j}
\left\lVert \mathbf{x}_i-\boldsymbol{\mu}_j \right\rVert^2
\]

where:

- \(K\) is the number of clusters;
- \(C_j\) is cluster \(j\);
- \(\mathbf{x}_i\) is an observation assigned to that cluster; and
- \(\boldsymbol{\mu}_j\) is the centroid of cluster \(j\).

WCSS is also commonly called **inertia**.

### Example

If the model contains one cluster, calculate the squared distance from every observation to the single centroid and add the results.

If the model contains two clusters:

1. calculate and sum the squared distances from the observations in the first cluster to its centroid;
2. repeat the calculation for the second cluster; and
3. add the two cluster totals.

The same process applies to any number of clusters.

## Applying the Elbow Method

The Elbow Method does not determine \(K\) before K-Means is run. Instead, K-Means must be fitted repeatedly using several candidate values of \(K\).

For example:

1. fit K-Means with \(K=1\) and record its WCSS;
2. repeat for \(K=2,3,4,5,\ldots\);
3. plot the number of clusters on the x-axis;
4. plot WCSS on the y-axis; and
5. inspect the curve for an elbow-shaped bend.

As \(K\) increases, WCSS can only decrease or remain unchanged because more centroids allow observations to be represented more closely. In the extreme case where every observation forms its own cluster, each observation is also its cluster centroid, so WCSS equals zero.

Choosing the largest possible \(K\) is not useful, however. It would reproduce the individual observations instead of providing a meaningful summary of the dataset.

## Identifying the Elbow

The elbow is the point where adding another cluster no longer produces a large reduction in WCSS. It represents a practical compromise between:

- creating compact clusters with a low WCSS; and
- keeping the model simple with relatively few clusters.

The value of \(K\) at this bend is treated as a suitable candidate for the number of clusters.

The Elbow Method is a heuristic rather than an exact rule. Some curves have a clear bend, while others decline smoothly or contain several plausible elbows. When the answer is ambiguous, the final choice should also consider domain knowledge, interpretability, cluster stability, and other evaluation measures.

---

# Study Notes

## Core idea

> Run K-Means for several values of \(K\), record the WCSS for each model, plot WCSS against \(K\), and select the point after which additional clusters provide diminishing returns.

## Elbow Method workflow

\[
\text{Choose candidate values of }K
\rightarrow
\text{fit K-Means for each }K
\rightarrow
\text{record WCSS}
\rightarrow
\text{plot the curve}
\rightarrow
\text{inspect the elbow}
\]

## How to interpret the graph

| Graph element | Meaning |
|---|---|
| x-axis | Number of clusters, \(K\) |
| y-axis | WCSS or inertia |
| Steep decline | Adding clusters substantially improves compactness |
| Flattening curve | Additional clusters provide smaller improvements |
| Elbow | Candidate balance between compactness and simplicity |

## Why squared distances are used

Squaring the distances:

- prevents positive and negative differences from cancelling;
- gives greater weight to observations far from their centroids; and
- matches the objective optimized by standard K-Means.

## Important properties

- WCSS is never negative.
- WCSS generally decreases as \(K\) increases.
- WCSS reaches zero when each distinct observation is assigned its own cluster.
- A lower WCSS alone does not prove that a clustering is useful.
- Values of WCSS should be compared on the same preprocessed dataset.
- Because K-Means can converge to a local optimum, each candidate \(K\) should normally use multiple centroid initializations.

## Practical considerations

### Scale the features

K-Means and WCSS are distance-based. A feature with a much larger numerical scale can dominate the distances, centroid positions, and resulting elbow curve. Standardize or otherwise scale features when their units or ranges differ substantially.

### Use a sensible range of K values

The range should be large enough to reveal where the curve begins to flatten, but it should remain meaningful for the problem. Testing every value up to the number of observations is rarely useful.

### Use reproducible fitting

Set a random seed when reproducibility is required. Use multiple initializations for every candidate \(K\), and compare the best resulting inertia for each value.

### Validate the selected K

The elbow should be treated as evidence rather than a guaranteed answer. A candidate value can also be evaluated using:

- the **silhouette score**;
- cluster stability across repeated samples or runs;
- domain knowledge;
- business usefulness; and
- the interpretability of the resulting cluster profiles.

## Limitations of the Elbow Method

- The elbow may be weak or absent.
- Several bends may appear plausible.
- Visual selection is subjective.
- Outliers can distort WCSS because their distances are squared.
- The result can change after feature scaling or feature selection.
- An apparent elbow does not guarantee that the dataset contains natural clusters.
- The method inherits the assumptions and limitations of K-Means, including its preference for compact, roughly spherical clusters.

## Elbow Method versus K-Means

The two concepts serve different purposes:

- **K-Means** creates the cluster assignments and centroids for a specified \(K\).
- **The Elbow Method** compares K-Means models fitted with different values of \(K\).

Therefore, K-Means must be run multiple times before the elbow curve can be evaluated.

## Quick revision example

Suppose the WCSS values are:

| \(K\) | WCSS |
|---:|---:|
| 1 | 1,000 |
| 2 | 560 |
| 3 | 300 |
| 4 | 230 |
| 5 | 195 |

The reductions are 440, 260, 70, and 35. The improvement becomes much smaller after \(K=3\), so \(K=3\) is a reasonable elbow candidate. This conclusion should still be checked against the meaning and quality of the resulting clusters.

## Questions for self-testing

1. What does WCSS measure?
2. Why must K-Means be fitted several times when applying the Elbow Method?
3. Why does WCSS decrease as \(K\) increases?
4. What happens to WCSS when every observation has its own cluster?
5. What trade-off does the elbow represent?
6. Why can identifying the elbow be subjective?
7. How can feature scaling affect the elbow curve?
8. Which other evidence could support or challenge the value of \(K\) suggested by the elbow?
