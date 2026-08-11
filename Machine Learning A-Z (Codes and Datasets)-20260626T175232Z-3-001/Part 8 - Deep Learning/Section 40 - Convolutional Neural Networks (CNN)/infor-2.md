# Step 1: Convolution

In the previous lesson, we introduced convolutional neural networks and learned that they classify images by detecting useful visual features. We will now examine the first major CNN operation: **convolution**.

## What Is Convolution?

In mathematics and signal processing, convolution combines two functions to describe how one modifies the other. In image processing, the same central idea is applied to arrays: a small matrix moves across an image and produces a new numerical representation.

For a two-dimensional input image $I$ and a filter $K$, a discrete convolution can be written as:

$$
(I * K)(i,j)=\sum_m\sum_n I(i-m,j-n)K(m,n)
$$

Deep-learning libraries generally implement the closely related **cross-correlation** operation, in which the filter is not flipped:

$$
Y(i,j)=\sum_m\sum_n I(i+m,j+n)K(m,n)
$$

Despite this technical distinction, the operation is conventionally called convolution in deep learning. Because the filter values are learned during training, the distinction does not usually affect how a CNN is designed or understood.

For a beginner-friendly mathematical treatment, see Jianxin Wu’s *Introduction to Convolutional Neural Networks*. The notes explain CNN foundations and their underlying mathematics in an accessible way.

## Input Image and Feature Detector

To keep the first example simple, imagine that an image is represented by a matrix containing only zeros and ones. We then define a smaller matrix—commonly $3 \times 3$—called a:

- **Filter**
- **Kernel**
- **Feature detector**

These names are often used interchangeably. The filter does not have to be $3 \times 3$; other sizes are possible. However, $3 \times 3$ filters are common because stacked small filters can build large receptive fields efficiently while introducing nonlinearities between layers.

## How the Operation Works

Place the filter over the top-left region of the input image. Multiply every filter value by the input value directly beneath it, then add the products. The sum becomes one value in the output.

For example:

$$
\begin{bmatrix}
1 & 0 & 1\\
0 & 1 & 0\\
1 & 0 & 1
\end{bmatrix}
\odot
\begin{bmatrix}
1 & 1 & 0\\
0 & 1 & 0\\
1 & 0 & 0
\end{bmatrix}
$$

The element-wise products are added:

$$
(1\times1)+(0\times1)+(1\times0)
+(0\times0)+(1\times1)+(0\times0)
+(1\times1)+(0\times0)+(1\times0)=3
$$

Therefore, the corresponding output position receives the value $3$. A large response means that the image region aligns strongly with the pattern represented by the filter, although the precise interpretation also depends on the filter’s weights and any bias or activation function used.

The filter then moves to the next valid position, and the calculation is repeated. Applying the same filter throughout the image is called **parameter sharing**. It allows the network to detect the same pattern regardless of where it appears.

## Feature Maps

The complete output produced by one filter is called a:

- **Feature map**
- **Activation map**—often used after an activation function has been applied
- **Convolved feature**

Each value in the feature map indicates how strongly the filter responded at a particular spatial location. Because nearby output values correspond to nearby regions of the input, convolution preserves spatial structure.

The usual sequence in a CNN layer is:

```text
Input → convolution → bias → activation function → activation map
```

For example, a ReLU activation replaces negative values with zero:

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

## Stride

The **stride** is the number of pixels by which the filter moves at each step.

- A stride of 1 moves the filter one pixel at a time.
- A stride of 2 moves it two pixels at a time.
- A larger stride produces fewer output positions and therefore a smaller feature map.

A stride of 1 is very common when preserving spatial detail is important. A stride of 2 is often used when the architecture deliberately needs to downsample the feature maps. There is no single stride that is best for every convolutional layer.

## Padding and Output Size

Without padding, a filter cannot be centred on pixels near the border without extending beyond the image. This is known as **valid convolution**, and it normally reduces the spatial dimensions.

**Padding** adds values—usually zeros—around the border of the input. Padding can preserve dimensions and allow border pixels to influence more output positions.

For a square input of size $N$, filter size $F$, padding $P$, and stride $S$, the output size along one spatial dimension is:

$$
\text{output size}
=\left\lfloor\frac{N+2P-F}{S}\right\rfloor+1
$$

For example, applying a $3 \times 3$ filter with stride 1 and no padding to a $7 \times 7$ input produces:

$$
\left\lfloor\frac{7-3}{1}\right\rfloor+1=5
$$

The resulting feature map is therefore $5 \times 5$.

If one layer of zero-padding is added ($P=1$), the result is:

$$
\left\lfloor\frac{7+2-3}{1}\right\rfloor+1=7
$$

The spatial dimensions are preserved. This is commonly described as **same padding** when the output height and width match the input for stride 1.

## Does Convolution Lose Information?

Convolution may reduce spatial size, especially when no padding or a stride greater than 1 is used. Consequently, some fine-grained information may be lost. However, reducing the image is not the sole or primary purpose of convolution; its defining purpose is to **extract local features**.

The network learns filters that retain task-relevant patterns. A filter may respond to an edge, corner, texture, colour transition, or another pattern that is useful for prediction. Later layers combine simple patterns into more complex ones.

Downsampling can reduce memory and computation, but it must be balanced against the need to retain spatial detail. CNN architectures control this balance through filter size, stride, padding, pooling, and the number of layers.

## Why Use Multiple Filters?

One filter detects one type of pattern. A convolutional layer therefore applies many different filters to the same input.

```text
Input image
 ├─ Filter 1 → Feature map 1
 ├─ Filter 2 → Feature map 2
 ├─ Filter 3 → Feature map 3
 └─ ...      → ...
```

The feature maps are stacked to form the layer’s output volume. If a layer contains 32 filters, it produces 32 output feature maps. Thus, the number of filters determines the number of output channels.

During supervised training, backpropagation adjusts the filters to reduce the model’s prediction error. The filters are not normally chosen by hand. Some may learn recognizable patterns such as edges, while others—particularly in deeper layers—may represent patterns that do not have a simple human-readable name.

## Convolution with Colour Images

An RGB image has three input channels: red, green, and blue. A filter applied to such an image must span all three channels.

For example, a $3 \times 3$ filter operating on an RGB image has dimensions:

$$
3 \times 3 \times 3
$$

The filter produces one two-dimensional feature map by multiplying across all spatial positions and all input channels, then summing the results. Multiple filters produce multiple output channels.

More generally, if the input has $C_{in}$ channels and the layer contains $C_{out}$ filters, the weight tensor has shape:

$$
F_H \times F_W \times C_{in} \times C_{out}
$$

when using a channels-last convention. Frameworks may store these dimensions in a different order, but the underlying relationship is the same.

## Fixed Image-Processing Kernels

Traditional image-processing programs use manually designed kernels to produce effects such as:

- Sharpening
- Blurring
- Edge enhancement
- Edge detection
- Embossing

For example, a simple box-blur kernel averages nearby pixels:

$$
\frac{1}{9}
\begin{bmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{bmatrix}
$$

A sharpening kernel may emphasize a centre pixel relative to its neighbours:

$$
\begin{bmatrix}
0&-1&0\\
-1&5&-1\\
0&-1&0
\end{bmatrix}
$$

An edge detector responds strongly where pixel intensity changes rapidly. These fixed kernels demonstrate how different filters produce different feature maps from the same image.

A trained CNN uses the same general mechanism, but its filters are learned from data rather than selected from a predefined list of effects. The model discovers the patterns that are useful for its particular task.

## Hierarchical Feature Learning

CNNs typically learn a hierarchy of visual patterns:

1. **Early layers** detect simple local patterns such as edges, orientations, and colour contrasts.
2. **Middle layers** combine them into textures, curves, and object parts.
3. **Deeper layers** respond to more task-specific arrangements and high-level patterns.

This hierarchy is one reason CNNs are so effective: complex objects can be represented as combinations of increasingly sophisticated features.

## Key Takeaway

The purpose of convolution is to detect useful local patterns and record their responses in feature maps while preserving spatial relationships. A CNN uses multiple learned filters so that it can represent many kinds of visual evidence at once.

---

# Study Notes

## Essential Definitions

| Term | Definition |
|---|---|
| Convolution | A local weighted-sum operation applied repeatedly across an input |
| Filter/kernel | A small matrix of weights used to detect a pattern |
| Feature detector | An intuitive name for a learned convolutional filter |
| Feature map | The spatial output generated by one filter |
| Stride | The number of pixels the filter moves at each step |
| Padding | Extra border values added around the input |
| Valid convolution | Convolution without padding |
| Same padding | Padding chosen to preserve spatial size, commonly with stride 1 |
| Parameter sharing | Reusing the same filter weights at every spatial position |
| Receptive field | The region of the original input that can affect a particular activation |
| Output channel | One feature map produced by one filter |

## Hand-Calculation Procedure

To calculate one value in a feature map:

1. Place the filter over an input region of the same height and width.
2. Multiply corresponding values element by element.
3. Add all the products.
4. Add the filter’s bias, if one is used.
5. Store the result at the corresponding output position.
6. Move the filter by the specified stride and repeat.
7. Apply the activation function when required.

## Output-Shape Formula

For each spatial dimension:

$$
O=\left\lfloor\frac{N+2P-F}{S}\right\rfloor+1
$$

Where:

- $O$ = output size
- $N$ = input size
- $P$ = padding
- $F$ = filter size
- $S$ = stride

### Quick Examples

| Input | Filter | Padding | Stride | Output |
|---:|---:|---:|---:|---:|
| $7\times7$ | $3\times3$ | 0 | 1 | $5\times5$ |
| $7\times7$ | $3\times3$ | 1 | 1 | $7\times7$ |
| $7\times7$ | $3\times3$ | 1 | 2 | $4\times4$ |
| $32\times32$ | $5\times5$ | 0 | 1 | $28\times28$ |

## Parameter Count

For a convolutional layer with filter height $F_H$, filter width $F_W$, $C_{in}$ input channels, and $C_{out}$ filters:

$$
\text{weights}=F_HF_WC_{in}C_{out}
$$

If every filter has one bias:

$$
\text{total parameters}
=(F_HF_WC_{in}+1)C_{out}
$$

The parameter count does **not** depend on the input image’s height or width because each filter is shared across spatial locations.

### Example

A layer with 32 filters of size $3\times3$ applied to an RGB input has:

$$
(3\times3\times3+1)\times32=896
$$

trainable parameters.

## Why Convolution Is Efficient

- **Local connectivity:** each output examines only a small region of the input.
- **Parameter sharing:** the same weights are reused across all positions.
- **Spatial preservation:** feature-map positions correspond to regions in the input.
- **Hierarchical composition:** deeper layers combine simpler features into complex ones.

## Common Misconceptions

**“The main purpose of convolution is to shrink the image.”**  
Its primary role is feature extraction. Shrinking occurs only with particular choices of padding and stride.

**“Stride 2 is the conventional choice for every layer.”**  
Stride 1 is common for detailed feature extraction; stride 2 is commonly used for deliberate downsampling.

**“Every learned filter is understandable to humans.”**  
Some early filters resemble familiar edge detectors, but many learned features—especially deeper ones—do not have a simple visual interpretation.

**“A large activation always means an exact match.”**  
It means the learned weighted sum is large. With real-valued weights, biases, negative values, and activations, this is more nuanced than counting matching binary entries.

**“Each RGB channel is convolved independently to produce three outputs.”**  
A standard convolutional filter spans all input channels and sums across them to produce one output feature map.

## Self-Check Questions

1. What are three common names for the small matrix applied across an image?
2. How is a single feature-map value calculated?
3. What does stride control?
4. Why does a valid convolution usually reduce spatial dimensions?
5. What is the purpose of padding?
6. Why does a convolutional layer use multiple filters?
7. What determines the number of output channels?
8. How deep must a standard filter be for an RGB input?
9. What is parameter sharing, and why is it useful?
10. How do fixed image-processing kernels differ from CNN filters?

## Short Answers

1. Filter, kernel, and feature detector.
2. Multiply corresponding filter and input values, sum the products, and normally add a bias before applying an activation.
3. How many pixels the filter moves between calculations.
4. The filter can occupy only positions where it fits entirely within the unpadded input.
5. To control output dimensions and give border pixels greater participation in the operation.
6. Each filter can learn a different visual pattern.
7. The number of filters in the layer.
8. Three channels—the complete depth of the RGB input.
9. It is the reuse of one filter at every position; it reduces parameters and detects a pattern wherever it appears.
10. Traditional kernels are manually specified, whereas CNN filters are normally learned by optimizing the network on training data.

## One-Sentence Summary

> Convolution slides learned filters across an input, producing spatial feature maps that indicate where useful local patterns occur.
