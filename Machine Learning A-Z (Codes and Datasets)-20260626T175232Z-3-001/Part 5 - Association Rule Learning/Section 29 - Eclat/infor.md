# Eclat: Intuition and Study Notes

## Overview

Eclat is an algorithm for discovering **frequent itemsets**: groups of items that repeatedly occur together in a transactional dataset. Its name is commonly expanded as **Equivalence Class Clustering and Bottom-Up Lattice Traversal**.

Like Apriori, Eclat can support applications such as:

- market-basket analysis;
- product bundling and cross-selling;
- movie or content discovery;
- patterns of services used together;
- exploratory work for recommendation systems.

For example, a retailer may discover that burgers and french fries frequently appear in the same basket. A streaming service may find that two movies frequently appear in the same users' watch histories.

Eclat and Apriori address the same fundamental frequent-itemset problem, but they search the data differently. Eclat is therefore better described as an **alternative to Apriori**, not merely a simplified form of it.

## Itemsets rather than directional rules

An itemset is an unordered collection of one or more items:

$$
X = \{\text{burger},\ \text{french fries}\}
$$

The itemset says that the two products occur together. It does not assign a direction such as:

$$
\text{burger} \rightarrow \text{french fries}
$$

This distinction matters:

- `{burger, french fries}` is a **frequent itemset**;
- `burger → french fries` is a **directional association rule**.

The core output of Eclat is a collection of frequent itemsets and their support values. If directional recommendations are required, association rules can subsequently be generated from those itemsets and evaluated with metrics such as confidence and lift.

## Support

In the course's Eclat workflow, the primary metric is **support**. For an itemset $X$:

$$
\operatorname{support}(X)
=
\frac{\text{number of transactions containing every item in }X}
{\text{total number of transactions}}
$$

Suppose 750 of 1,000 baskets contain both burgers and french fries:

$$
\operatorname{support}(\{\text{burger},\text{fries}\})
=
\frac{750}{1000}
=
0.75
$$

The pair therefore has 75% support.

### Important probability distinction

A support of 75% means that **75% of all transactions contain both items**. It does not automatically mean:

> Among customers who buy burgers, 75% also buy french fries.

That second statement describes confidence:

$$
\operatorname{confidence}(\text{burger}\rightarrow\text{fries})
=
\frac{\operatorname{support}(\{\text{burger},\text{fries}\})}
{\operatorname{support}(\{\text{burger}\})}
$$

For example, if burger support is 80% and joint support is 75%, then:

$$
\operatorname{confidence}(\text{burger}\rightarrow\text{fries})
=
\frac{0.75}{0.80}
=
0.9375
$$

The confidence is therefore 93.75%, not 75%.

## How Eclat works

### Horizontal transaction format

Transactional data are commonly introduced in a horizontal layout:

| Transaction | Items |
|---|---|
| T1 | burger, fries, drink |
| T2 | burger, fries |
| T3 | vegetables, fruit |
| T4 | burger, drink |

Each row identifies a transaction and lists its items.

### Vertical transaction-ID format

Eclat converts the data into a vertical representation. Each item is associated with the set of transaction IDs in which it occurs:

| Item | Transaction-ID set |
|---|---|
| burger | {T1, T2, T4} |
| fries | {T1, T2} |
| drink | {T1, T4} |
| vegetables | {T3} |
| fruit | {T3} |

To find the transactions containing both burgers and fries, Eclat intersects their transaction-ID sets:

$$
\{T1,T2,T4\}\cap\{T1,T2\}=\{T1,T2\}
$$

The support count of `{burger, fries}` is therefore 2:

$$
\operatorname{support}(\{\text{burger},\text{fries}\})=\frac{2}{4}=0.5
$$

Larger itemsets are evaluated using further intersections. For example:

$$
\operatorname{TID}(\{\text{burger},\text{fries},\text{drink}\})
=
\operatorname{TID}(\{\text{burger},\text{fries}\})
\cap
\operatorname{TID}(\{\text{drink}\})
$$

This vertical structure allows Eclat to calculate support without repeatedly scanning every original transaction for every candidate itemset.

## Eclat procedure

At a high level, Eclat performs the following steps:

1. Choose a minimum support threshold.
2. Build a transaction-ID set for each individual item.
3. Remove items whose support is below the threshold.
4. Combine remaining items and intersect their transaction-ID sets.
5. Retain each combined itemset whose support meets the threshold.
6. Recursively extend frequent itemsets, usually through a depth-first search.
7. Return the frequent itemsets and their support values.
8. Optionally sort the results by decreasing support or generate directional rules afterward.

Eclat uses the same downward-closure principle that supports Apriori pruning:

> If an itemset is infrequent, every larger itemset containing it must also be infrequent.

For example, if `{burger, fries}` is below the minimum support, `{burger, fries, drink}` cannot be frequent.

## Eclat compared with Apriori

| Aspect | Apriori | Eclat |
|---|---|---|
| Main task | Mine frequent itemsets | Mine frequent itemsets |
| Typical data layout | Horizontal transactions | Vertical transaction-ID sets |
| Search style | Usually breadth-first, level by level | Usually depth-first |
| Support calculation | Candidate counting through transaction scans | Transaction-ID set intersections |
| Core output | Frequent itemsets | Frequent itemsets |
| Association rules | Generated from frequent itemsets | Generated from frequent itemsets |

The course often presents Apriori together with support, confidence, and lift because its implementation directly exposes association rules. Algorithmically, however, both Apriori and Eclat primarily mine frequent itemsets; rule generation is a related stage.

Eclat can be faster when transaction-ID intersections are efficient, particularly for some dense datasets. It is not universally faster: large or dense transaction-ID sets can consume substantial memory, and performance depends on the dataset and implementation.

## Choosing the minimum support

A high threshold:

- produces fewer itemsets;
- focuses on common combinations;
- reduces computation;
- may discard niche but valuable patterns.

A low threshold:

- finds more itemsets;
- increases runtime and memory use;
- may produce many weak or accidental combinations;
- requires more careful business validation.

The threshold should reflect the number of transactions, the expected frequency of meaningful combinations, and the intended business action.

## Interpreting frequent itemsets responsibly

Sorting itemsets by decreasing support places the most common combinations first, but the most common pair is not necessarily the most useful recommendation.

Support alone does not reveal:

- direction: whether $A$ predicts $B$ or $B$ predicts $A$;
- baseline popularity: whether the items co-occur simply because both are common;
- incremental value: whether a recommendation changes customer behavior;
- causation: whether purchasing one item causes the other purchase;
- profitability: whether a bundle or promotion has positive commercial value.

For recommendation or promotion decisions, supplement support with confidence, lift, item margins, customer segmentation, and experimental validation.

## Quick review

1. **What does Eclat discover?**

   Frequent itemsets and their support values.

2. **What is Eclat's key data representation?**

   A vertical mapping from each item or itemset to the transaction IDs containing it.

3. **How is support calculated efficiently?**

   By intersecting transaction-ID sets and counting the IDs in the intersection.

4. **Is `{A, B}` the same as `A → B`?**

   No. The first is an unordered itemset; the second is a directional rule.

5. **Does 75% support for `{A, B}` imply 75% confidence for `A → B`?**

   No. Confidence also depends on the support of $A$.

6. **Why can Eclat prune an infrequent itemset?**

   None of its supersets can have greater support.

7. **Is Eclat always faster than Apriori?**

   No. Performance depends on data density, transaction-ID set sizes, memory, and implementation.

## Key takeaway

Eclat mines frequent combinations by representing the database vertically and intersecting transaction-ID sets. Its core measure is support. High-support itemsets are useful candidates for further analysis, but directional recommendations require additional rule metrics and business validation.
