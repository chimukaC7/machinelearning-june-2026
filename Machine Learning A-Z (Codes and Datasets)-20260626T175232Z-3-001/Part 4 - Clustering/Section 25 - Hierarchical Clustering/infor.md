# Hierarchical Clustering

Hierarchical clustering is an **unsupervised learning** technique that groups similar observations into a hierarchy of nested clusters. Like K-Means, it can reveal structure in unlabeled data, and the two methods may sometimes produce similar final groups. Their procedures, however, are fundamentally different.

K-Means directly partitions observations into a predetermined number of clusters. Hierarchical clustering records a sequence of cluster merges or splits, allowing the clustering structure to be examined at multiple levels.

## Types of Hierarchical Clustering

There are two main approaches:

### Agglomerative clustering

Agglomerative hierarchical clustering uses a **bottom-up** strategy:

1. Begin with every observation in its own cluster.
2. Find the two closest clusters.
3. Merge them into one larger cluster.
4. Repeat until all observations belong to one cluster.

This is the approach used most commonly in introductory examples and in this course.

### Divisive clustering

Divisive hierarchical clustering uses the reverse, **top-down** strategy:

1. Begin with all observations in one cluster.
2. Split a cluster into smaller clusters.
3. Continue splitting until every observation forms its own cluster, or until another stopping condition is met.

Divisive clustering follows the same hierarchical idea in the opposite direction, although deciding how to split a cluster requires its own procedure.

## Agglomerative Hierarchical Clustering Step by Step

Suppose the dataset contains \(n\) observations.

### Step 1: Create singleton clusters

Treat every observation as a separate cluster. The algorithm therefore begins with \(n\) clusters.

### Step 2: Merge the closest pair

Find the two closest clusters and merge them. After the first merge, \(n-1\) clusters remain.

### Step 3: Recalculate cluster distances

After the merge, update the distances between the new cluster and every remaining cluster. The rule used for this calculation is called the **linkage criterion**.

### Step 4: Repeat

Continue selecting and merging the closest pair:

\[
n \rightarrow n-1 \rightarrow n-2 \rightarrow \cdots \rightarrow 2 \rightarrow 1
\]

After \(n-1\) merges, all observations belong to one cluster.

The purpose is not normally to use this final all-inclusive cluster. Instead, the algorithm preserves the entire merge history in a **dendrogram**. Cutting the dendrogram at a chosen height produces the desired number of clusters.

## Distance Between Individual Observations

For two points in a two-dimensional space,

\[
\mathbf{p}_1=(x_1,y_1)
\qquad\text{and}\qquad
\mathbf{p}_2=(x_2,y_2),
\]

their Euclidean distance is:

\[
d(\mathbf{p}_1,\mathbf{p}_2)
=
\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
\]

This follows from the Pythagorean theorem: the horizontal and vertical differences form the two legs of a right-angled triangle, while the Euclidean distance is its hypotenuse.

In \(m\) dimensions, the formula becomes:

\[
d(\mathbf{x},\mathbf{z})
=
\sqrt{\sum_{r=1}^{m}(x_r-z_r)^2}
\]

Euclidean distance is common, but other distance measures may be more suitable for certain feature types and business problems.

## Distance Between Clusters: Linkage Criteria

Once a cluster contains several observations, there is no single obvious definition of its distance from another cluster. The chosen definition is the **linkage criterion**, and it can significantly change the resulting hierarchy.

Let \(A\) and \(B\) be two clusters.

### Single linkage

Single linkage uses the shortest distance between any observation in \(A\) and any observation in \(B\):

\[
d_{\text{single}}(A,B)
=
\min_{\mathbf{x}\in A,\mathbf{z}\in B} d(\mathbf{x},\mathbf{z})
\]

It can identify elongated or irregularly shaped clusters, but it is susceptible to the **chaining effect**, where a sequence of nearby observations joins otherwise distinct groups.

### Complete linkage

Complete linkage uses the greatest pairwise distance:

\[
d_{\text{complete}}(A,B)
=
\max_{\mathbf{x}\in A,\mathbf{z}\in B} d(\mathbf{x},\mathbf{z})
\]

It generally favors compact clusters and is less prone to chaining, but it can be sensitive to outliers.

### Average linkage

Average linkage uses the mean of all pairwise distances between observations in the two clusters:

\[
d_{\text{average}}(A,B)
=
\frac{1}{|A||B|}
\sum_{\mathbf{x}\in A}
\sum_{\mathbf{z}\in B}
d(\mathbf{x},\mathbf{z})
\]

It provides a compromise between single and complete linkage by considering the overall relationship between the two clusters.

### Centroid linkage

Centroid linkage measures the distance between the cluster centroids:

\[
d_{\text{centroid}}(A,B)
=
d(\boldsymbol{\mu}_A,\boldsymbol{\mu}_B)
\]

where \(\boldsymbol{\mu}_A\) and \(\boldsymbol{\mu}_B\) are the mean vectors of the two clusters.

### Ward linkage

Ward linkage merges the pair of clusters that causes the smallest increase in total within-cluster variance, often expressed as the increase in within-cluster sum of squares. It tends to create compact, similarly sized clusters and is commonly used with Euclidean distance.

Ward linkage is not simply another pairwise minimum or maximum rule. Its merge criterion is the loss of cluster compactness caused by combining two groups.

## A Small Conceptual Example

Assume the dataset contains six observations:

1. Start with six singleton clusters.
2. Merge the closest pair, leaving five clusters.
3. Recalculate distances using the selected linkage criterion.
4. Merge the closest remaining pair, leaving four clusters.
5. Continue until three, two, and finally one cluster remain.

At every step, “closest” must be interpreted according to the selected linkage rule. With single linkage, for example, the distance between two clusters is the distance between their closest members. A different linkage criterion could produce a different merge order.

## The Role of the Dendrogram

A dendrogram is a tree-like visualization of the merge history:

- each leaf represents an observation;
- each horizontal join represents a merge;
- the height of a join represents the dissimilarity or merge cost; and
- a horizontal cut through the tree determines the final clusters.

Because the hierarchy is retained, the algorithm does not need the final number of clusters before constructing the full dendrogram. A suitable partition can be selected afterward, although that choice still requires judgment and validation.

---

# Study Notes

## Core idea

> Agglomerative hierarchical clustering starts with one cluster per observation and repeatedly merges the closest pair. The definition of “closest” is controlled by the distance metric and linkage criterion, while the merge history is stored in a dendrogram.

## Quick algorithm summary

\[
\text{Singleton clusters}
\rightarrow
\text{find closest pair}
\rightarrow
\text{merge}
\rightarrow
\text{update distances}
\rightarrow
\text{repeat}
\rightarrow
\text{dendrogram}
\rightarrow
\text{cut the tree}
\]

## Important terminology

| Term | Meaning |
|---|---|
| Singleton cluster | A cluster containing one observation |
| Agglomerative | Bottom-up merging approach |
| Divisive | Top-down splitting approach |
| Distance metric | Rule for measuring dissimilarity between observations |
| Linkage criterion | Rule for measuring the separation or merge cost between clusters |
| Dendrogram | Tree that records the hierarchical merge or split history |
| Dendrogram cut | A selected height used to obtain a final partition |

## Linkage comparison

| Linkage | Cluster distance or merge rule | Typical behavior |
|---|---|---|
| Single | Closest pair of observations | Can find elongated groups; may chain |
| Complete | Farthest pair of observations | Produces compact groups; outliers can matter |
| Average | Mean of all cross-cluster pairwise distances | Balanced compromise |
| Centroid | Distance between cluster means | Summarizes clusters by their centers |
| Ward | Smallest increase in within-cluster variance | Often creates compact, similarly sized groups |

## Two separate choices

Do not confuse the following:

1. The **distance metric** defines dissimilarity between observations—for example, Euclidean, Manhattan, cosine, or a metric designed for mixed data.
2. The **linkage criterion** defines how observation-level distances or cluster statistics are used to compare clusters.

Both choices can change the dendrogram and final cluster assignments.

## Hierarchical Clustering versus K-Means

| Aspect | Hierarchical clustering | K-Means |
|---|---|---|
| Main process | Repeated merges or splits | Repeated assignment and centroid updates |
| Number of clusters required before fitting | Not necessarily for the full hierarchy | Yes |
| Main representation | Dendrogram | Centroids and labels |
| Cluster center required | Not for every linkage | Yes |
| Typical scalability | Can be expensive for large datasets | Usually more scalable |
| Reversibility during fitting | Agglomerative merges are not undone | Assignments can change across iterations |
| Sensitivity | Metric, linkage, scale, and outliers | Initialization, scale, outliers, and \(K\) |

## Practical workflow

1. Select meaningful features.
2. Handle missing values and unsuitable data types.
3. Scale features when their units or ranges differ.
4. Choose a distance metric that matches the data.
5. Choose and document a linkage criterion.
6. Fit the hierarchy and inspect the dendrogram.
7. Select a plausible cut height or number of clusters.
8. Validate the result using stability, silhouette measures, domain knowledge, and interpretability.
9. Profile and name the clusters only after examining their characteristics.

## Important cautions

- Features with large scales can dominate Euclidean distance.
- Outliers can cause early singleton behavior or distorted merges.
- A greedy agglomerative merge cannot be reversed later.
- Different linkage rules can yield very different solutions.
- A dendrogram may be difficult to read when the dataset contains many observations.
- Computing and storing pairwise relationships can make hierarchical clustering expensive for large datasets.
- A visible hierarchy does not prove that natural clusters exist.
- The selected dendrogram cut should be validated rather than chosen from appearance alone.

## Common misconceptions

### “Hierarchical clustering always returns one useful cluster”

The full agglomerative process ends with one cluster, but the useful result normally comes from cutting the dendrogram earlier.

### “The closest clusters are always the ones with the closest centroids”

That is true only when centroid linkage is selected. Other linkage criteria define closeness differently.

### “Hierarchical clustering works only in two dimensions”

Scatter plots are convenient illustrations, but the method can operate in higher-dimensional feature spaces.

### “The dendrogram height always represents ordinary Euclidean distance”

The meaning of height depends on the metric and linkage criterion. With Ward linkage, for example, it reflects the merge cost associated with increased within-cluster variation.

## Questions for self-testing

1. What is the difference between agglomerative and divisive clustering?
2. How many clusters exist at the beginning of agglomerative clustering with \(n\) observations?
3. How many merges are required to reach one cluster?
4. What is the Euclidean distance formula in two dimensions?
5. What is the difference between a distance metric and a linkage criterion?
6. How do single, complete, and average linkage define cluster distance?
7. What does Ward linkage attempt to minimize at each merge?
8. Why can different linkage criteria produce different dendrograms?
9. How is a final clustering obtained from a dendrogram?
10. Why is feature scaling often important?
11. What is the chaining effect?
12. Why can hierarchical clustering be difficult to apply to very large datasets?
