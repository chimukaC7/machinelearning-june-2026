# K-Means Clustering

K-Means is a simple but effective **unsupervised learning** algorithm used to divide data points into clusters. Unlike supervised learning, clustering begins without predefined classes, labels, or training examples. We are given only the data and must discover meaningful groups within it.

## How K-Means Works

### 1. Choose the number of clusters

First, choose the number of clusters, denoted by **K**. The method used to select a suitable value of K will be discussed separately. For this example, assume that:

> K = 2

### 2. Initialize the centroids

Place one **centroid** for each cluster at an initial position. These centroids are commonly initialized randomly, and they do not have to coincide with existing data points.

### 3. Assign each point to its nearest centroid

Calculate the distance between every data point and each centroid. Assign each point to the cluster whose centroid is closest.

For two centroids, the assignment boundary can be visualized as a line containing all positions that are equally distant from both centroids. Points on one side of the boundary belong to one cluster, while points on the other side belong to the second cluster.

### 4. Recalculate the centroids

Find the mean position of all the points assigned to each cluster. In two dimensions, this involves:

- calculating the mean of their x-coordinates; and
- calculating the mean of their y-coordinates.

These two means give the new position of the cluster's centroid. The previous centroid is not treated as a data point and is therefore not included in this calculation.

For a cluster containing \(n\) points, its centroid is:

\[
\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i
\]

### 5. Repeat until convergence

After moving the centroids to their newly calculated positions:

1. reassign every point to its nearest centroid;
2. recalculate the mean of each cluster; and
3. move each centroid to its new mean position.

This process continues until the assignments and centroid positions stop changing, or until their changes become smaller than a chosen tolerance. At this point, the algorithm has **converged**, and the resulting centroid positions and cluster assignments are final.

The clusters can then be interpreted using business context and domain knowledge. K-Means identifies groups mathematically, but it does not explain what those groups mean.

---

# Study Notes

## Core idea

K-Means repeatedly alternates between two operations:

1. **Assignment step:** assign each observation to its nearest centroid.
2. **Update step:** replace each centroid with the mean of the observations assigned to it.

This is why the algorithm is called **K-Means**:

- **K** is the chosen number of clusters.
- **Means** refers to the mean position used as the center of each cluster.

## Objective function

K-Means attempts to minimize the **within-cluster sum of squares (WCSS)**, also called **inertia**:

\[
\text{WCSS} = \sum_{j=1}^{K}\sum_{\mathbf{x}_i \in C_j}
\left\lVert \mathbf{x}_i-\boldsymbol{\mu}_j \right\rVert^2
\]

where:

- \(K\) is the number of clusters;
- \(C_j\) is the set of points assigned to cluster \(j\);
- \(\mathbf{x}_i\) is a data point; and
- \(\boldsymbol{\mu}_j\) is the centroid of cluster \(j\).

A smaller WCSS means that observations are, on average, closer to the centers of their assigned clusters.

## Important terminology

| Term | Meaning |
|---|---|
| Cluster | A group of observations considered similar according to their distance |
| Centroid | The mean position of all observations assigned to a cluster |
| K | The number of clusters specified before training |
| Convergence | The point at which assignments or centroids no longer change significantly |
| Inertia/WCSS | The total squared distance from each point to its assigned centroid |

## Distance measure

Standard K-Means normally uses **Euclidean distance**:

\[
d(\mathbf{x},\boldsymbol{\mu}) =
\sqrt{\sum_{r=1}^{m}(x_r-\mu_r)^2}
\]

Because distance determines cluster membership, features with larger numeric scales can dominate the result. Feature scaling, such as standardization, is therefore often important before applying K-Means.

## Key considerations

- The value of K must be selected in advance.
- Different initial centroid positions can produce different final clusters.
- **K-Means++** initialization generally gives more reliable starting centroids than purely random initialization.
- It is common to run K-Means several times with different initializations and retain the solution with the lowest inertia.
- K-Means works best when clusters are compact, roughly spherical, similarly sized, and well separated.
- It can perform poorly with irregularly shaped clusters, clusters of very different densities or sizes, and significant outliers.
- Standard K-Means is designed for numerical features because it relies on means and distances.
- The numerical cluster labels have no ranking or intrinsic meaning; “Cluster 0” is not inherently better or lower than “Cluster 1.”

## Convergence does not guarantee the global optimum

K-Means improves or preserves its objective value during each iteration, so it eventually converges. However, it may converge to a **local optimum** rather than the best possible clustering. This is why initialization strategy and multiple runs are important.

## Quick revision summary

> Choose K → initialize K centroids → assign points to the nearest centroid → update each centroid to the cluster mean → repeat until convergence → interpret the resulting clusters.

## Questions for self-testing

1. Why is K-Means described as an unsupervised learning algorithm?
2. What does the value K represent?
3. How is a centroid recalculated?
4. What is minimized by the K-Means algorithm?
5. Why should features often be scaled before clustering?
6. Why can two runs of K-Means produce different results?
7. What types of cluster shapes are difficult for K-Means to identify?
8. Why is domain knowledge needed after the clusters have been formed?
