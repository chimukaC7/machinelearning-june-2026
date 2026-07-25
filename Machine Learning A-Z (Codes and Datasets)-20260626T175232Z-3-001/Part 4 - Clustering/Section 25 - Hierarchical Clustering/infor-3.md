# Selecting Clusters from a Dendrogram

A dendrogram records the complete merge history of agglomerative hierarchical clustering. Reading from the bottom upward shows which observations or clusters merged first and the linkage dissimilarity at which each merge occurred.

This structure allows us to move beyond the final all-inclusive cluster and extract a useful partition containing two, three, four, or more clusters.

## Revisiting the Six-Point Hierarchy

Consider the observations:

\[
P_1,\ P_2,\ P_3,\ P_4,\ P_5,\ P_6
\]

The dendrogram records the following sequence:

1. \(P_2\) merges with \(P_3\).
2. \(P_5\) merges with \(P_6\).
3. \(P_1\) joins the cluster \(\{P_2,P_3\}\).
4. \(P_4\) joins the cluster \(\{P_5,P_6\}\).
5. The two remaining clusters merge into one:

\[
\{P_1,P_2,P_3\}
\qquad\text{and}\qquad
\{P_4,P_5,P_6\}
\]

Lower horizontal joins represent merges made at lower linkage dissimilarities. The final join appears much higher because the two large groups are considerably less similar.

## Cutting the Dendrogram

To extract final clusters, draw an imaginary horizontal line across the dendrogram at a selected **distance threshold** or **dissimilarity threshold**, denoted by \(t\).

The rule is:

> Do not perform any merge whose dendrogram height is above the selected threshold.

Every connected subtree below the cut becomes a separate cluster.

### Counting the resulting clusters

If the cut does not pass exactly through a merge:

\[
\text{Number of clusters}
=
\text{number of vertical branches crossed by the horizontal cut}
\]

Each crossed branch represents one connected component below the threshold.

## Example: A Threshold Producing Two Clusters

Place the cut below the final large merge but above the merges that created the two three-point groups.

The cut intersects two vertical branches, so the result is:

\[
C_1=\{P_1,P_2,P_3\}
\]

\[
C_2=\{P_4,P_5,P_6\}
\]

The final merge between these two groups is excluded because its linkage height exceeds the threshold.

## Example: A Threshold Producing Four Clusters

Now lower the threshold to a position below the joins that attach \(P_1\) and \(P_4\), but above the first two pairwise merges.

The horizontal cut crosses four branches, producing:

\[
\{P_1\},\quad
\{P_2,P_3\},\quad
\{P_4\},\quad
\{P_5,P_6\}
\]

The early \(P_2\)-\(P_3\) and \(P_5\)-\(P_6\) merges remain, while the later merges are excluded.

## Example: A Threshold Producing Six Clusters

If the threshold is placed below the first merge, no observations are allowed to join.

The cut crosses all six leaf branches:

\[
\{P_1\},\{P_2\},\{P_3\},\{P_4\},\{P_5\},\{P_6\}
\]

Thus:

- moving the cut **downward** generally increases the number of clusters; and
- moving the cut **upward** generally decreases the number of clusters.

## What the Threshold Really Guarantees

Suppose the threshold is \(t\). Cutting at this height guarantees that every merge retained in the extracted hierarchy occurred at a linkage value no greater than \(t\).

It does **not** always guarantee:

\[
d(\mathbf{x},\mathbf{z})\leq t
\quad
\text{for every pair }
\mathbf{x},\mathbf{z}
\text{ in the same cluster}.
\]

For example, single linkage can form a chain of nearby observations whose endpoints are much farther apart than the threshold. The interpretation of a cut therefore depends on the linkage criterion:

- with **single linkage**, retained merges satisfy a nearest-neighbor rule;
- with **complete linkage**, the threshold more directly limits the maximum cross-cluster distance at each merge;
- with **average linkage**, it limits average cross-cluster dissimilarity at each merge; and
- with **Ward linkage**, it limits a variance-based merge cost rather than ordinary pairwise distance.

## The Largest Vertical-Gap Heuristic

A common visual method for choosing a threshold is to look for a large jump in merge height.

Conceptually:

1. examine the horizontal merge levels;
2. identify the largest vertical interval containing no merge;
3. place a horizontal cut somewhere inside that interval; and
4. count the vertical branches crossed.

The logic is that merges below the gap combine relatively similar groups, while the next merge above it requires a much larger increase in dissimilarity.

In the six-point example, the largest gap appears before the final merge. Cutting through this interval gives two clusters:

\[
\{P_1,P_2,P_3\}
\qquad\text{and}\qquad
\{P_4,P_5,P_6\}
\]

For another dendrogram, the largest gap might produce three clusters if the cut crosses three branches.

## Is This the Optimal Number of Clusters?

The largest-gap rule is a **heuristic**, not a mathematical guarantee of the one correct answer.

A different threshold may be more useful when:

- domain knowledge requires a particular level of detail;
- one suggested cluster is too small or unstable;
- a different solution is easier to interpret or act upon;
- the metric or linkage choice creates misleading structure; or
- external validation favors another partition.

The dendrogram provides evidence and candidate solutions. The final decision should combine that evidence with quantitative validation and the goals of the analysis.

## Relating the Cut to the Scatter Plot

The dendrogram and scatter plot provide different views:

- the scatter plot shows geometric positions in the selected two-dimensional feature space; and
- the dendrogram shows the nested merge structure produced by the chosen metric and linkage.

When a cut suggests two or three clusters, coloring the scatter plot by those assignments helps determine whether the groups are visually coherent. However, scatter plots can be misleading when the model uses more than two features because the displayed projection may omit important structure.

---

# Study Notes

## Core idea

> Choose a horizontal dendrogram cut, count the vertical branches it crosses, and treat the connected subtrees below the cut as the final clusters.

## Threshold behavior

| Threshold position | Effect |
|---|---|
| Below every merge | Every observation remains a singleton |
| Between early and later merges | Some small clusters are preserved |
| Below the final merge | Usually produces a small number of large clusters |
| Above the final merge | All observations belong to one cluster |

## Six-point example summary

| Cut position | Number of clusters | Partition |
|---|---:|---|
| Below the first merge | 6 | Every point is separate |
| Above the first two merges but below later joins | 4 | \(\{P_1\},\{P_2,P_3\},\{P_4\},\{P_5,P_6\}\) |
| Below the final merge | 2 | \(\{P_1,P_2,P_3\}\), \(\{P_4,P_5,P_6\}\) |
| Above the final merge | 1 | All six observations together |

## Equivalent ways to specify the result

A hierarchical clustering result can often be requested by:

- selecting a distance threshold \(t\); or
- selecting a target number of clusters \(K\).

These are two ways of choosing a level in the same hierarchy. If merge heights are tied, the exact handling of a cut at that height can depend on the software's convention, so it is safer to place a conceptual cut between merge levels.

## Largest-gap procedure

\[
\text{Find a large jump in merge height}
\rightarrow
\text{cut inside the gap}
\rightarrow
\text{count crossed branches}
\rightarrow
\text{validate the partition}
\]

Large jumps suggest that the algorithm is beginning to merge substantially less similar groups.

## Validation checklist

After choosing a candidate cut, examine:

- the size of every cluster;
- cluster profiles and interpretability;
- silhouette score or related internal measures;
- stability across samples and time periods;
- sensitivity to feature scaling;
- sensitivity to distance metric and linkage criterion;
- the influence of outliers;
- agreement with relevant domain knowledge; and
- whether the resulting segments are useful for the intended decision.

## Common mistakes

### Calling the largest gap the guaranteed optimum

It is only one selection heuristic. Real datasets may contain several plausible gaps or no clear gap.

### Saying every pair within a cluster is closer than the threshold

The cut constrains retained **linkage merges**, not necessarily every pairwise observation distance. The distinction is especially important with single linkage.

### Counting horizontal bars instead of vertical intersections

Count the vertical branches crossed by the horizontal threshold line.

### Cutting exactly through a merge without defining a convention

A cut at a tied merge height can be ambiguous. Choose a threshold between merge levels or use an explicit software rule.

### Ignoring preprocessing

The hierarchy and its apparent largest gap can change after scaling, transforming, or selecting features.

### Reading a high-dimensional model only from a two-dimensional plot

A plotted projection may hide separation or overlap present in the full feature space.

## Hierarchical selection versus the Elbow Method

| Hierarchical clustering | K-Means Elbow Method |
|---|---|
| Uses merge heights in a dendrogram | Uses WCSS across several values of \(K\) |
| One fitted hierarchy exposes many possible cuts | K-Means is fitted repeatedly for different \(K\) values |
| Large height gaps suggest candidate cuts | A bend in the WCSS curve suggests a candidate \(K\) |
| Depends strongly on metric and linkage | Depends on initialization, scaling, and K-Means assumptions |

Neither visual method guarantees the correct number of clusters.

## Questions for self-testing

1. How is a final partition extracted from a dendrogram?
2. What does each vertical branch crossed by a cut represent?
3. What happens to the number of clusters when the cut moves downward?
4. Which four clusters appear in the intermediate six-point example?
5. Why does a cut below the first merge produce six clusters?
6. What does a threshold guarantee about retained merges?
7. Why does it not necessarily bound every pairwise distance within a cluster?
8. How is the largest vertical-gap heuristic applied?
9. Why is the largest gap not guaranteed to give the optimal number of clusters?
10. What should be checked after selecting a candidate threshold?
11. Why might tied merge heights require special care?
12. How does dendrogram-based selection differ from the K-Means Elbow Method?
