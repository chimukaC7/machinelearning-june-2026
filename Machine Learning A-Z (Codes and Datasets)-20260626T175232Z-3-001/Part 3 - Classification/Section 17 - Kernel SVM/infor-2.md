# Mapping Data to a Higher-Dimensional Space

In this lesson, we will examine how a dataset that is not linearly separable in its original feature space can sometimes become linearly separable after being mapped to a higher-dimensional space.

The general process is:

1. Begin with data that cannot be separated effectively by a linear boundary.
2. apply a feature-mapping function;
3. represent the observations in a higher-dimensional feature space;
4. use an SVM to find a linear maximum-margin separator in that space; and
5. interpret that separator as a non-linear decision boundary in the original space.

We will begin with a one-dimensional example before extending the idea to two dimensions.

## A One-Dimensional Example

Suppose a dataset contains nine observations positioned along a single number line. The observations near the center belong to one class, while those farther to the left and right belong to another.

In one dimension, a linear decision boundary is a single threshold point. One threshold can divide the number line into only two regions:

\[
x < c
\qquad\text{and}\qquad
x > c
\]

It cannot isolate a class in the center while assigning observations on both outer sides to the other class. The dataset is therefore not linearly separable in its original one-dimensional space.

## Constructing a Feature Mapping

Assume that the center of the data occurs around \(x=5\). We can define a new feature:

\[
z=\phi(x)=(x-5)^2
\]

This transformation has two conceptual steps.

### Step 1: Center the Data

\[
u=x-5
\]

Subtracting \(5\) shifts the reference point from \(x=5\) to \(u=0\):

- observations to the left of \(5\) receive negative values;
- the observation at \(5\) becomes \(0\); and
- observations to the right of \(5\) receive positive values.

This translation alone does not make the classes linearly separable.

### Step 2: Square the Centered Values

\[
z=u^2=(x-5)^2
\]

Squaring removes the sign and measures squared distance from the center:

- observations close to \(x=5\) receive small \(z\)-values;
- observations far from \(x=5\), on either side, receive large \(z\)-values.

For example:

| Original value \(x\) | Centered value \(x-5\) | New feature \(z=(x-5)^2\) |
|---:|---:|---:|
| 2 | -3 | 9 |
| 4 | -1 | 1 |
| 5 | 0 | 0 |
| 6 | 1 | 1 |
| 8 | 3 | 9 |

Values that were on opposite sides of the center can now occupy the same position on the squared-distance axis. This is useful when class membership depends on distance from the center rather than on whether an observation lies to its left or right.

## Separation in the Transformed Space

The mapping can be visualized by representing each observation with the coordinates:

\[
(x,z)=(x,(x-5)^2)
\]

All transformed observations lie on a parabola. A horizontal line,

\[
z=c,
\]

can separate observations near the center from observations farther away. This is a linear boundary in the two-dimensional \((x,z)\) feature space.

If only the new feature \(z\) is needed, the same classifier can be viewed as a single threshold on the transformed one-dimensional axis:

\[
z<c
\qquad\text{or}\qquad
z>c.
\]

The key point is that the transformed representation makes a linear rule possible.

## Returning to the Original Space

It is common to describe the final step as “projecting” the separator back into the original space. More precisely, we substitute the feature mapping into the separator's equation.

Starting with:

\[
z=c,
\]

and substituting \(z=(x-5)^2\), we obtain:

\[
(x-5)^2=c.
\]

Solving for \(x\) gives:

\[
x=5-\sqrt{c}
\qquad\text{or}\qquad
x=5+\sqrt{c}.
\]

Therefore, one linear threshold in the transformed representation corresponds to two boundary points in the original one-dimensional space. These two points can enclose the center class:

\[
5-\sqrt{c}<x<5+\sqrt{c}.
\]

The decision rule is linear in the transformed feature but non-linear in the original feature.

## Extending the Idea from Two to Three Dimensions

Now consider a two-dimensional dataset with features \(x_1\) and \(x_2\). Suppose one class forms a central cluster and the second class surrounds it. A straight line cannot separate the classes effectively.

We can add a third feature based on distance from a selected center:

\[
z=(x_1-a)^2+(x_2-b)^2.
\]

Here, \((a,b)\) represents the center used by the transformation.

After mapping each observation from:

\[
(x_1,x_2)
\]

to:

\[
(x_1,x_2,z),
\]

the observations lie in a three-dimensional feature space. A plane such as:

\[
z=c
\]

may separate the two classes in that space.

In three dimensions, this plane is the SVM's linear hyperplane. Substituting the mapping into the plane equation gives:

\[
(x_1-a)^2+(x_2-b)^2=c.
\]

This is a circle in the original two-dimensional space. The linear separator in the transformed space therefore corresponds to a circular, non-linear decision boundary in the original space.

More generally:

> A linear hyperplane in a transformed feature space can correspond to a curved decision boundary in the original input space.

## The General Feature-Mapping Idea

A feature map can be written as:

\[
\phi:\mathbb{R}^{d}\rightarrow\mathbb{R}^{D},
\]

where \(D\) is usually larger than \(d\).

Instead of learning a classifier directly from \(x\), the SVM works with:

\[
\phi(x).
\]

Its decision function in the transformed space can be expressed as:

\[
f(x)=w^T\phi(x)+b.
\]

Although this expression is linear in \(\phi(x)\), it may be non-linear with respect to the original features \(x\).

Not every arbitrary mapping will make a dataset separable or produce a useful model. The transformation must expose structure that helps distinguish the classes, and the model must still be regularized to generalize beyond the training data.

## The Computational Problem

Explicitly constructing transformed features can be expensive:

- the number of generated features may be very large;
- storing the transformed dataset may require substantial memory;
- computing dot products in the expanded space may be slow; and
- some useful feature spaces are infinite-dimensional.

For example, polynomial transformations of many original features can generate a rapidly growing number of combinations. The computational cost becomes increasingly important as the number of observations and features grows.

## Motivation for the Kernel Trick

SVM training relies on inner products between feature vectors. In a transformed space, the required inner product is:

\[
\phi(x_i)^T\phi(x_j).
\]

A kernel function computes this value directly:

\[
K(x_i,x_j)=\phi(x_i)^T\phi(x_j),
\]

without explicitly constructing \(\phi(x_i)\) or \(\phi(x_j)\).

The kernel trick does not eliminate the higher-dimensional feature-space concept. Instead, it allows the algorithm to perform the calculations associated with that space implicitly and often much more efficiently.

This is the foundation of Kernel SVMs.

---

# Study Notes

## Key Terms

| Term | Meaning |
|---|---|
| **Linear separator** | A point in one dimension, a line in two dimensions, a plane in three dimensions, or a hyperplane in general. |
| **Feature mapping** | A function \(\phi(x)\) that creates a new representation of an observation. |
| **Input space** | The space defined by the original features. |
| **Feature space** | The space defined by the transformed features. |
| **Hyperplane** | A linear decision boundary in a feature space of arbitrary dimension. |
| **Non-linear boundary** | A boundary that is curved or otherwise not representable by one linear hyperplane in the original space. |
| **Kernel trick** | Computing transformed-space inner products through a kernel without explicitly generating the transformed features. |

## Separators by Dimension

| Feature-space dimension | Linear separator |
|---:|---|
| 1 | A point or threshold |
| 2 | A line |
| 3 | A plane |
| \(D\) | A \((D-1)\)-dimensional hyperplane |

The word **hyperplane** is the general term and can also be used for lines and planes in their respective spaces.

## Worked One-Dimensional Example

Feature mapping:

\[
\phi(x)=(x-5)^2.
\]

Linear boundary in transformed space:

\[
\phi(x)=c.
\]

Equivalent boundary in original space:

\[
(x-5)^2=c.
\]

Boundary points:

\[
x=5\pm\sqrt{c}.
\]

Interpretation:

- points close to \(5\) have small transformed values;
- points far from \(5\) have large transformed values; and
- a single transformed-space threshold becomes two original-space boundary points.

## Worked Two-Dimensional Example

Feature mapping:

\[
\phi(x_1,x_2)=(x_1,x_2,(x_1-a)^2+(x_2-b)^2).
\]

Linear plane in transformed space:

\[
z=c.
\]

Equivalent boundary in the original space:

\[
(x_1-a)^2+(x_2-b)^2=c.
\]

This equation describes a circle with:

- center \((a,b)\); and
- radius \(\sqrt{c}\).

## Important Distinctions

### Mapping vs. Adding Arbitrary Dimensions

Simply adding a dimension does not guarantee separation. The new feature must encode a useful relationship, such as squared distance from a center.

### Linear in Feature Space vs. Linear in Input Space

The SVM separator remains linear in the transformed feature space:

\[
w^T\phi(x)+b=0.
\]

It becomes non-linear only when expressed using the original variables.

### Explicit Mapping vs. Kernel Evaluation

- **Explicit mapping:** calculate and store every transformed feature.
- **Kernel evaluation:** calculate only the inner products required by the SVM.

Both approaches refer to a transformed feature space, but the kernel approach may avoid constructing it directly.

## Why the Kernel Trick Helps

Assume an explicit mapping produces \(D\) transformed features. Calculating an ordinary dot product in that representation typically requires work proportional to \(D\). If a kernel provides the same result directly from the original features, the calculation may be much cheaper.

The kernel identity is:

\[
K(x_i,x_j)=\phi(x_i)^T\phi(x_j).
\]

This identity is the central fact to remember.

## Caveats

- A higher-dimensional mapping does not automatically guarantee good generalization.
- A highly flexible feature space may make overfitting easier.
- The kernel and its parameters must be chosen carefully.
- Soft-margin regularization remains important even when a kernel is used.
- Not every similarity function is a valid kernel for standard SVM optimization.

## Quick Review Questions

1. Why can one threshold not isolate a class in the middle of a number line?
2. What information does \((x-5)^2\) capture?
3. How does a transformed-space threshold become two boundary points in the original space?
4. What type of separator is used in one, two, and three dimensions?
5. How can a plane in three dimensions correspond to a circle in two dimensions?
6. Is the SVM separator curved in the transformed feature space?
7. Why does merely adding an arbitrary feature not guarantee linear separability?
8. Why can explicit higher-dimensional mappings be computationally expensive?
9. What quantity does a kernel function compute?
10. Does the kernel trick remove the idea of a higher-dimensional feature space?

## One-Sentence Summary

A suitable feature mapping can turn a non-linear classification problem into a linear one in a higher-dimensional space, while the kernel trick lets an SVM use that space without explicitly constructing all of its transformed features.
