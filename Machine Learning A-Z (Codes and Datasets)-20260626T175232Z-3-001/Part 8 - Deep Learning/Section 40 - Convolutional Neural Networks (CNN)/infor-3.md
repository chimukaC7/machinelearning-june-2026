# Step 1B: The ReLU Activation Function

After convolution produces a feature map, a convolutional neural network usually applies a nonlinear activation function. A common choice is the **rectified linear unit**, or **ReLU**.

This can be treated as an additional part of the convolution stage:

```text
Input → convolution → bias → ReLU → activation map
```

Some explanations present convolution and activation as separate layers. Others describe them together as a convolutional block. Either convention is acceptable as long as the two operations are understood clearly.

## Definition of ReLU

ReLU is defined as:

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

It operates element by element:

- If $x>0$, the output is $x$.
- If $x<0$, the output is $0$.
- At $x=0$, the output is $0$ by definition.

For example:

$$
\begin{bmatrix}
-3 & 2 & -1\\
4 & 0 & 5
\end{bmatrix}
\xrightarrow{\text{ReLU}}
\begin{bmatrix}
0 & 2 & 0\\
4 & 0 & 5
\end{bmatrix}
$$

When ReLU is applied to a convolution result, all negative activations become zero while positive activations remain unchanged.

## Why CNNs Need Nonlinearity

Convolution is a linear operation. Without a nonlinear activation, stacking several convolutional layers would still be equivalent to applying one larger linear transformation.

For two linear transformations:

$$
f(x)=W_1x+b_1
$$

and


$$
g(f(x))=W_2f(x)+b_2,
$$

their composition is:

$$
g(f(x))=W_2W_1x+W_2b_1+b_2
$$

This is still linear—or, more precisely, affine—in $x$. Adding more linear layers does not change that fundamental limitation.

ReLU breaks this collapse because it is nonlinear. Once it is inserted between convolutions, the network can build piecewise-linear decision functions and represent complex relationships between visual patterns.

```text
Convolution → ReLU → convolution → ReLU → ...
```

This allows early features such as edges and colour contrasts to be combined into increasingly complex representations such as textures, shapes, object parts, and class-specific patterns.

## A More Accurate Intuition

ReLU’s role is sometimes described as “breaking linear progressions” between neighbouring image pixels. That can be a misleading visual explanation. ReLU does not introduce nonlinearity merely by removing dark regions or disrupting smooth brightness transitions.

Instead, it changes the mathematical mapping performed by the network. Each unit has two operating regions:

- An **inactive region**, where negative inputs become zero
- An **active region**, where positive inputs pass through unchanged

Which region is active depends on the input. Across many ReLU units, these input-dependent switches allow the network to represent complicated functions that a purely linear stack cannot express.

## ReLU on Feature Maps

A convolutional filter can have positive and negative weights, so its feature map may contain both positive and negative values. The sign of a response depends on the learned filter and the input pattern.

After ReLU:

- Positive responses are preserved.
- Negative responses are set to zero.
- Zero-valued activations indicate that the unit is inactive for that input.

The resulting output is commonly called an **activation map**. In practice, visualizations may display negative values as dark and positive values as bright, but those colours are only a plotting convention—not the original colours of the image.

Importantly, setting a negative response to zero does not mean that the corresponding image region is useless in every context. Another filter may respond positively to the same region, and the network learns many filters simultaneously.

## Why ReLU Became Popular

ReLU has several practical advantages:

- **Simple computation:** it requires only a comparison with zero.
- **Useful gradients for positive inputs:** the gradient is 1 in the positive region.
- **Sparse activations:** negative inputs become zero, so many units may be inactive for a given example.
- **Effective deep-network training:** it often avoids the severe positive-region saturation associated with sigmoid and tanh activations.

ReLU does not eliminate every optimization problem, but it has historically worked well across many CNN architectures.

## The Derivative of ReLU

Backpropagation uses the derivative of the activation function:

$$
\operatorname{ReLU}'(x)=
\begin{cases}
0, & x<0\\
1, & x>0
\end{cases}
$$

The derivative is not mathematically defined at exactly $x=0$ because the graph has a sharp corner there. Software frameworks choose a subgradient at that point, commonly 0. This convention works well in practice because landing at exactly zero is rarely a significant obstacle for gradient-based learning.

During backpropagation:

- A positive pre-activation passes its upstream gradient through.
- A negative pre-activation blocks its upstream gradient.

## The Dying-ReLU Problem

ReLU has an important limitation. If a unit produces negative pre-activations for every training example, its ReLU output is always zero. Its local gradient is also zero, so the unit may stop updating and never become active again. This is called the **dying-ReLU problem**.

It can be made more likely by:

- An excessively high learning rate
- Unfavourable weight initialization
- Updates that push the unit permanently into the negative region

Common mitigations include careful initialization, suitable learning rates, normalization, and activation variants that retain a small gradient for negative inputs.

## ReLU Variants

### Leaky ReLU

Leaky ReLU keeps a small fixed slope $\alpha$ for negative inputs:

$$
f(x)=
\begin{cases}
x, & x\geq0\\
\alpha x, & x<0
\end{cases}
$$

where $\alpha$ might be a small value such as 0.01. Negative inputs therefore retain a small gradient rather than becoming completely inactive.

### Parametric ReLU (PReLU)

PReLU has the same general form as Leaky ReLU, but the negative slope is learned during training:

$$
f(x)=
\begin{cases}
x, & x\geq0\\
a x, & x<0
\end{cases}
$$

where $a$ is a trainable parameter. PReLU was introduced by Kaiming He and colleagues in *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*.

### Other Alternatives

Modern neural networks may also use activations such as:

- ELU
- GELU
- SiLU, also called Swish

These alternatives have different smoothness, gradient, and computational properties. ReLU remains an excellent starting point and is still widely used, but it is not mandatory for every architecture.

## Weight Initialization and ReLU

The choice of weight initialization interacts with the activation function. **He initialization**, introduced in the same line of work on rectifiers, scales initial weights to help maintain signal variance through networks that use ReLU-like activations.

For a layer with $\text{fan}_{in}$ input connections, a common form draws weights from a zero-mean distribution with variance approximately:

$$
\operatorname{Var}(W)=\frac{2}{\text{fan}_{in}}
$$

Modern frameworks often expose this as Kaiming or He initialization.

## Suggested Reading

- C.-C. Jay Kuo, *Understanding Convolutional Neural Networks with a Mathematical Model* — discusses the mathematical role of nonlinear activation functions in CNNs.
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* — introduces PReLU and an initialization method designed for rectifier networks.

## Key Takeaway

Convolution extracts local patterns through a linear weighted-sum operation. ReLU then introduces nonlinearity by preserving positive activations and replacing negative ones with zero. This simple input-dependent switch enables stacked CNN layers to learn far more complex functions than a sequence of convolutions alone.

---

# Study Notes

## Essential Vocabulary

| Term | Meaning |
|---|---|
| Activation function | A function applied to a neuron’s pre-activation, usually to introduce nonlinearity |
| ReLU | The function $\max(0,x)$ |
| Pre-activation | The convolution or weighted sum plus bias, before applying the activation function |
| Activation map | A feature map after an activation function has been applied |
| Piecewise linear | Linear within separate regions but not necessarily linear across the whole domain |
| Sparse activation | An output representation containing many zeros |
| Dying ReLU | A ReLU unit that remains inactive and receives zero local gradient |
| PReLU | A ReLU variant with a learned slope for negative inputs |
| He initialization | A weight-initialization method designed for ReLU-like activations |

## ReLU Cheat Sheet

| Input $x$ | Output | Local derivative |
|---:|---:|---:|
| $x<0$ | $0$ | $0$ |
| $x=0$ | $0$ | Chosen by convention, commonly $0$ |
| $x>0$ | $x$ | $1$ |

## ReLU Versus Common Alternatives

| Activation | Negative region | Main characteristic |
|---|---|---|
| ReLU | Zero | Simple and computationally efficient |
| Leaky ReLU | Small fixed slope | Reduces the risk of permanently inactive units |
| PReLU | Learned slope | Learns how much negative signal to retain |
| Sigmoid | Smooth, bounded | Useful for binary output probabilities but can saturate in hidden layers |
| Tanh | Smooth, bounded, zero-centred | Can also saturate at large magnitudes |
| GELU/SiLU | Smooth nonlinear weighting | Common in several modern architectures |

## Why the Activation Must Be Nonlinear

```text
Linear layer → linear layer → linear layer
                 ↓
       Equivalent to one linear mapping

Linear layer → ReLU → linear layer → ReLU
                 ↓
       Can represent complex piecewise functions
```

The critical point is not that images themselves contain abrupt brightness changes. The network needs a nonlinear function because composing only linear transformations cannot model nonlinear relationships, regardless of how many linear layers are added.

## Common Misconceptions

**“ReLU removes black areas from the original image.”**  
ReLU acts on numerical activations, not directly on perceived colours. A black region in a visualization may simply represent a negative or zero value.

**“Negative feature-map values are always unwanted information.”**  
Their meaning depends on the filter. ReLU chooses to suppress them in that channel, while other filters may capture complementary patterns.

**“ReLU makes the model nonlinear because it creates abrupt visual edges.”**  
The nonlinearity comes from the mathematical function $\max(0,x)$, whose behaviour changes depending on the sign of its input.

**“ReLU solves the vanishing-gradient problem completely.”**  
It provides a gradient of 1 for positive inputs, but negative units have a zero gradient and can die. Deep-network optimization can still face other gradient problems.

**“ReLU is differentiable everywhere.”**  
It is not differentiable at exactly zero; frameworks use a chosen subgradient there.

## Self-Check Questions

1. What does ReLU stand for?
2. Write the ReLU function mathematically.
3. What happens to negative, zero, and positive inputs?
4. Why can several linear layers be collapsed into one linear transformation?
5. What does ReLU add between convolutional layers?
6. What is the derivative of ReLU for a positive input? For a negative input?
7. Why can ReLU create sparse activation maps?
8. What is the dying-ReLU problem?
9. How does Leaky ReLU differ from standard ReLU?
10. What parameter does PReLU learn?

## Short Answers

1. Rectified Linear Unit.
2. $\operatorname{ReLU}(x)=\max(0,x)$.
3. Negative values become zero, zero remains zero, and positive values pass through unchanged.
4. The composition of linear or affine transformations is still a linear or affine transformation.
5. A nonlinear, input-dependent transformation that increases the network’s expressive power.
6. The derivative is 1 for positive inputs and 0 for negative inputs.
7. Every negative pre-activation is replaced with zero.
8. A unit can remain in the negative region for all inputs, producing zero output and zero local gradient.
9. It retains a small fixed slope for negative inputs.
10. The slope applied to negative inputs.

## One-Sentence Summary

> ReLU applies $\max(0,x)$ to every convolution output, introducing the nonlinearity that allows a deep CNN to learn complex visual representations.
