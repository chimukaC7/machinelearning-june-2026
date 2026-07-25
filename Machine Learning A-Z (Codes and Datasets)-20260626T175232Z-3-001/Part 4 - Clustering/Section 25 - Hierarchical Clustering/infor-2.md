# Understanding Dendrograms

Agglomerative hierarchical clustering begins with every observation in its own cluster and repeatedly merges the closest pair until only one cluster remains.

At first, ending with a single cluster may seem unhelpful. The real value of the procedure is that it preserves the complete merge history. This history is visualized using a **dendrogram**, which allows us to inspect the hierarchy and later select a useful clustering with two, three, or more groups.

## What Is a Dendrogram?

A dendrogram is a tree-like diagram that records how hierarchical clusters were formed.

- The **leaves** along the bottom represent the original observations.
- A **horizontal joining line** shows that two observations or clusters were merged.
- The **height of the join** represents the dissimilarity or linkage value at which the merge occurred.
- Lower joins represent more similar clusters.
- Higher joins represent less similar clusters.

The dendrogram is therefore a memory of the agglomerative clustering algorithm: reading it from the bottom upward reveals the order in which clusters were combined.

> The horizontal positions of the leaves mainly arrange the tree for display. The meaningful quantity is normally the height at which branches join, not the visual spacing between neighboring leaf labels.

## A Six-Point Example

Consider six observations labeled:

\[
P_1,\ P_2,\ P_3,\ P_4,\ P_5,\ P_6
\]

At the start, every point is a singleton cluster:

\[
\{P_1\},\{P_2\},\{P_3\},\{P_4\},\{P_5\},\{P_6\}
\]

### First merge: \(P_2\) and \(P_3\)

Suppose \(P_2\) and \(P_3\) are the closest pair. They are merged into:

\[
\{P_2,P_3\}
\]

On the dendrogram, vertical branches rise from \(P_2\) and \(P_3\) and are connected by a horizontal line. The height of this line is their merge dissimilarity.

Because this is the first merge, it occurs at the lowest linkage value in the example.

### Second merge: \(P_5\) and \(P_6\)

Suppose \(P_5\) and \(P_6\) are the next closest pair:

\[
\{P_5,P_6\}
\]

Their horizontal join appears higher than the \(P_2\)-\(P_3\) join because their computed dissimilarity is greater.

The current clusters are now:

\[
\{P_1\},\{P_2,P_3\},\{P_4\},\{P_5,P_6\}
\]

### Third merge: \(P_1\) joins \(\{P_2,P_3\}\)

The algorithm compares all remaining clusters using the chosen distance metric and linkage criterion. Suppose \(P_1\) is closest to the cluster containing \(P_2\) and \(P_3\). The new cluster becomes:

\[
\{P_1,P_2,P_3\}
\]

The dendrogram joins the \(P_1\) branch to the existing \(P_2\)-\(P_3\) branch at the appropriate linkage height.

### Fourth merge: \(P_4\) joins \(\{P_5,P_6\}\)

Suppose \(P_4\) is closest to the cluster containing \(P_5\) and \(P_6\):

\[
\{P_4,P_5,P_6\}
\]

This merge is added to the dendrogram at its computed height.

### Final merge

Two clusters remain:

\[
\{P_1,P_2,P_3\}
\qquad\text{and}\qquad
\{P_4,P_5,P_6\}
\]

They are merged at a relatively large height because the two groups are considerably less similar than the clusters merged during the earlier steps.

The final tree contains all six observations, but it also retains every intermediate grouping.

## How to Read the Merge Order

To reconstruct the clustering sequence:

1. begin at the lowest horizontal join;
2. identify the leaves or branches it connects;
3. move upward to the next join;
4. repeat until reaching the final merge.

For the illustrative example, the sequence is:

\[
\{P_2\}+\{P_3\}
\rightarrow
\{P_5\}+\{P_6\}
\rightarrow
\{P_1\}+\{P_2,P_3\}
\rightarrow
\{P_4\}+\{P_5,P_6\}
\rightarrow
\{P_1,P_2,P_3\}+\{P_4,P_5,P_6\}
\]

Merges that occur at similar heights have similar linkage values. The precise ordering of equal or nearly equal merges may depend on exact calculations and tie-breaking rules.

## What Does Dendrogram Height Measure?

The vertical axis represents the **linkage dissimilarity** or **merge cost**.

In a simple example using Euclidean distance and single linkage, the height of a merge is the shortest Euclidean distance between members of the two clusters:

\[
d_{\text{single}}(A,B)
=
\min_{\mathbf{x}\in A,\mathbf{z}\in B}
\left\lVert\mathbf{x}-\mathbf{z}\right\rVert_2
\]

For two individual observations, this reduces to their ordinary Euclidean distance.

However, the vertical axis should not always be described as raw Euclidean distance. Its exact meaning depends on the linkage criterion:

| Linkage | Meaning of merge height |
|---|---|
| Single | Smallest cross-cluster pairwise distance |
| Complete | Largest cross-cluster pairwise distance |
| Average | Mean cross-cluster pairwise distance |
| Centroid | Distance between cluster centroids |
| Ward | Increase in within-cluster variation caused by the merge |

Therefore, a dendrogram must be interpreted together with the metric and linkage method used to construct it.

## From a Dendrogram to Final Clusters

The hierarchy becomes a final partition by drawing an imaginary horizontal line across the dendrogram:

- every connected subtree below the cut becomes one cluster; and
- the number of vertical branches intersected by the cut equals the number of selected clusters.

A cut below the final high merge in the six-point example would produce two clusters:

\[
\{P_1,P_2,P_3\}
\qquad\text{and}\qquad
\{P_4,P_5,P_6\}
\]

Moving the cut lower produces more clusters; moving it higher produces fewer.

A large vertical gap with no intervening merges is often considered a possible place to cut, because it suggests that the next merge combines relatively dissimilar groups. This remains a heuristic and should be supported by validation and domain knowledge.

---

# Study Notes

## Core idea

> A dendrogram records who merged with whom, the order of the merges, and the linkage dissimilarity at which each merge occurred.

## Dendrogram anatomy

| Component | Interpretation |
|---|---|
| Leaf | Original observation |
| Branch | Observation or cluster lineage |
| Horizontal join | Merger of two clusters |
| Join height | Linkage dissimilarity or merge cost |
| Root/top join | Final merge into one cluster |
| Horizontal cut | Rule for extracting a final partition |

## Reading rules

- Read from the bottom upward to follow agglomerative clustering.
- A lower merge indicates greater similarity under the selected metric and linkage.
- A higher merge indicates greater dissimilarity.
- Trace downward from a join to find all observations contained in that cluster.
- Do not interpret the left-to-right spacing of leaves as feature-space distance.
- Leaf order can sometimes be rearranged without changing the underlying hierarchy.

## Why the final one-cluster result is useful

The final cluster is not usually the intended business result. It is the root of a tree containing every nested partition created along the way. Because the whole hierarchy is available, analysts can compare alternative cluster counts without rerunning the complete algorithm for each possible number.

## Dissimilarity is contextual

In a customer dataset, two points might represent people described by age and salary. A short distance means the customers are similar only with respect to those selected, preprocessed features. It does not mean they are similar in every business or personal characteristic.

The interpretation also depends on:

- which features were included;
- how those features were scaled;
- which distance metric was chosen; and
- which linkage criterion was used.

## Common mistakes

### Treating leaf spacing as distance

The horizontal layout helps display the tree. It does not generally preserve the distances seen in the original scatter plot.

### Calling every merge height Euclidean distance

This is accurate only in specific metric-linkage configurations. Ward linkage, for example, represents a variance-based merge cost.

### Assuming that a dendrogram determines one objectively correct cluster count

The tree displays alternatives. Choosing a cut remains an analytical decision that should consider separation, stability, interpretability, and domain value.

### Ignoring feature scaling

If one feature has a much larger numerical scale, it can dominate pairwise distances and change the entire merge order.

### Comparing heights from differently configured dendrograms directly

Heights generated with different metrics, linkages, or preprocessing steps may not have the same meaning or scale.

## Practical checklist

1. Confirm that the input features are meaningful and properly scaled.
2. Record the distance metric and linkage criterion.
3. Inspect low merges to identify the tightest local groups.
4. Look for large increases in merge height.
5. Evaluate several plausible horizontal cuts.
6. Check cluster sizes, profiles, stability, and silhouette values.
7. Use domain knowledge before assigning names or actions to clusters.
8. Document the selected cut and the reasoning behind it.

## Questions for self-testing

1. Why is a dendrogram described as the memory of hierarchical clustering?
2. What does each leaf represent?
3. What does a horizontal joining line represent?
4. What information is encoded by the height of a merge?
5. Why do \(P_2\) and \(P_3\) merge before \(P_5\) and \(P_6\) in the example?
6. How can the merge order be reconstructed from the diagram?
7. Why does the final merge usually occur at a relatively large height?
8. Why is dendrogram height not always ordinary Euclidean distance?
9. How does a horizontal cut produce final clusters?
10. What happens to the number of clusters when the cut is moved downward?
11. Why should a large vertical gap be treated as a heuristic rather than proof?
12. Why can leaf order change while the hierarchy remains equivalent?
