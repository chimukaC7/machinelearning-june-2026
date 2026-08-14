# Convolutional Neural Networks — Study Notes

> Based on SuperDataScience's [The Ultimate Guide to Convolutional Neural Networks (CNN)](https://www.superdatascience.com/blogs/the-ultimate-guide-to-convolutional-neural-networks-cnn), published August 27, 2018. These notes paraphrase and consolidate the article while adding technical clarifications where its intuitive explanations are simplified.

## 1. What Is a CNN?

A **convolutional neural network (CNN)** is a neural network designed to learn useful patterns from grid-like data, especially images. Instead of treating every pixel independently, convolutional layers examine small local regions and learn filters that respond to visual patterns.

CNNs are widely associated with tasks such as:

- image classification;
- object detection;
- facial and expression analysis;
- medical-image analysis;
- autonomous-driving perception; and
- any problem in which local spatial structure matters.

The article uses an analogy with human vision: recognition depends on detecting and combining informative features. This is useful intuition, but a CNN is a mathematical model rather than a literal simulation of the brain.

## 2. How Images Are Represented

An image is represented numerically as a grid of pixels.

### Grayscale images

A grayscale image commonly has one channel. In an 8-bit representation, each pixel usually has a value from 0 to 255:

- 0 represents black;
- 255 represents white; and
- intermediate values represent shades of gray.

A grayscale image with height $H$ and width $W$ can be represented with shape:

$$
H \times W \times 1
$$

The channel dimension is sometimes omitted when discussing the image conceptually.

### Color images

An RGB image contains three channels: red, green, and blue. Each pixel is described by three values, so its usual channels-last shape is:

$$
H \times W \times 3
$$

For example, the RGB triplet $(255, 105, 180)$ represents a shade of pink.

Although 0–255 is common for stored 8-bit images, CNN inputs are often rescaled to $[0,1]$ or transformed using a model-specific normalization scheme before training.

## 3. The CNN Pipeline

The article presents the following conceptual sequence:

$$
\text{Input image}
\rightarrow \text{Convolution}
\rightarrow \text{ReLU}
\rightarrow \text{Pooling}
\rightarrow \text{Flattening}
\rightarrow \text{Fully connected layers}
\rightarrow \text{Prediction}
$$

Convolution, activation, and downsampling are normally repeated in several blocks. Modern architectures may also use batch normalization, residual connections, dropout, or global pooling.

## 4. Step 1: Convolution

Convolution uses a small matrix of learned values called a **filter** or **kernel**. A common kernel size is $3 \times 3$, although other sizes are possible.

The kernel moves across the input. At every location, the network multiplies the kernel values by the corresponding input values and sums the products. A bias may then be added:

$$
z_{i,j,k}
= \sum_{u,v,c} x_{i+u,j+v,c}w_{u,v,c,k}+b_k
$$

where:

- $x$ is the input;
- $w$ contains the weights of filter $k$;
- $b_k$ is that filter's bias; and
- $z_{i,j,k}$ is one output value in the resulting feature map.

The output produced by one filter is a **feature map**, also called an activation map after an activation has been applied. Multiple filters produce multiple feature maps.

### Important distinction

- A **filter/kernel** is a learned set of weights.
- A **feature map** is the output produced by applying a filter to an input.
- A **convolutional layer** usually contains many filters and therefore produces many feature maps.

The article illustrates convolution by counting matching cells. That is an intuitive binary example; real CNNs calculate weighted sums rather than simply counting matches.

### Stride

The **stride** determines how far the kernel moves at each step:

- stride 1 moves one pixel at a time;
- stride 2 moves two pixels at a time and reduces the spatial output size more quickly.

### Padding

Padding controls how borders are handled:

- **valid padding** adds no padding, so the output usually becomes smaller;
- **same padding** adds values around the border so that a stride-1 convolution preserves height and width.

For a one-dimensional spatial size $N$, kernel size $K$, padding $P$, and stride $S$, the output size is:

$$
\left\lfloor \frac{N+2P-K}{S} \right\rfloor + 1
$$

### What filters learn

During training, the network learns filter values through backpropagation. Early filters often respond to simple patterns such as edges or orientations. Deeper filters combine earlier activations and may respond to textures, shapes, object parts, or highly abstract patterns.

Convolution is effective because of:

- **local connectivity:** each output examines a small region;
- **weight sharing:** the same filter is applied across the image; and
- **multiple learned filters:** the layer can detect several kinds of patterns.

These properties preserve spatial structure and require far fewer parameters than connecting every image pixel directly to a dense layer.

## 5. Step 1(b): ReLU

After convolution, CNNs commonly apply the **Rectified Linear Unit (ReLU)**:

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

ReLU keeps positive inputs and replaces negative inputs with zero. Its central role is to introduce **non-linearity**, allowing a stack of layers to learn relationships that a sequence of purely linear transformations could not represent.

It is more precise to say that ReLU makes the model non-linear than to say it “removes linearity from the image.” ReLU operates on learned activations, not on the semantic content of the original image.

## 6. Step 2: Pooling

Pooling downsamples each feature map. The article focuses on **max pooling**, which selects the largest activation in each local window:

$$
y_{i,j,c}=\max_{(u,v)\in\mathcal{R}_{i,j}}x_{u,v,c}
$$

A common configuration is a $2 \times 2$ window with stride 2. It reduces each spatial dimension by roughly half and the number of spatial values by roughly 75%.

### Why pool?

Pooling can:

- reduce spatial dimensions;
- reduce computation and memory use;
- preserve the strongest local responses;
- enlarge the effective receptive field of later layers; and
- provide limited robustness to small translations or local distortions.

The article describes this property as “spatial variance” in one place and “spatial invariance” elsewhere. The intended idea is **limited local translation invariance or robustness**. Pooling does not make a CNN automatically invariant to large movements, rotation, scale changes, or arbitrary viewpoint changes.

Pooling has no trainable parameters. It may reduce overfitting by shrinking the representation, but it does not guarantee that overfitting will be prevented.

Other pooling operations include average pooling and global average pooling. Some modern CNNs use strided convolutions instead of separate pooling layers.

## 7. Step 3: Flattening

After the convolutional blocks, the feature maps can be flattened into one vector.

If the input to `Flatten` has shape:

$$
H \times W \times C
$$

the output vector has length:

$$
HWC
$$

For example:

$$
14 \times 14 \times 32 = 6{,}272
$$

Flattening has no trainable parameters; it changes only the tensor shape.

A large flattened vector connected to a dense layer can create many parameters. Modern classification networks therefore often replace flattening with **global average pooling**, which averages each feature map into one value.

## 8. Step 4: Full Connection

The flattened representation can be passed to one or more **fully connected**, or **dense**, layers. Every dense neuron receives all outputs from the preceding layer:

$$
z_j=\sum_i w_{ji}x_i+b_j
$$

Dense layers combine the extracted visual evidence into representations useful for the final decision.

The article describes the final hidden neurons as casting weighted “votes” for different classes. This is helpful intuition. Mathematically, the output layer calculates learned weighted sums followed by an appropriate activation function.

The features used at this depth should not usually be interpreted literally as individual human-readable concepts such as “nose” or “ear.” A neuron may respond to a distributed and abstract combination of patterns.

## 9. Output Layers

The output design depends on the task.

| Task | Typical output activation | Typical loss |
| --- | --- | --- |
| Binary, mutually exclusive classes | One sigmoid output | Binary cross-entropy |
| Multiclass, mutually exclusive classes | One softmax output per class | Categorical cross-entropy |
| Multilabel classification | One sigmoid output per label | Binary cross-entropy |
| Regression | Linear output(s) | MSE, MAE, or another regression loss |

## 10. Softmax

For $K$ mutually exclusive classes, the output layer first produces unrestricted scores called **logits**. Softmax converts them into probabilities:

$$
P(y=i)=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

Softmax outputs:

- lie between 0 and 1; and
- sum to 1 across the classes.

The output neurons do not communicate in order to coordinate their probabilities. Their values are normalized together through the shared softmax denominator.

Softmax is appropriate when exactly one class is correct. It is not appropriate when several labels may simultaneously be true; independent sigmoid outputs are normally used for multilabel problems.

## 11. Cross-Entropy

Categorical cross-entropy compares the true label distribution $y$ with the predicted probabilities $\hat{y}$:

$$
L=-\sum_{i=1}^{K}y_i\log(\hat{y}_i)
$$

For a one-hot target, only the correct class contributes:

$$
L=-\log(\hat{y}_{\text{correct}})
$$

Therefore:

- assigning high probability to the correct class gives low loss;
- assigning low probability to the correct class gives high loss; and
- confidently incorrect predictions are penalized strongly.

Examples using the natural logarithm:

| Probability assigned to true class | Cross-entropy loss |
| ---: | ---: |
| 0.99 | 0.010 |
| 0.90 | 0.105 |
| 0.50 | 0.693 |
| 0.10 | 2.303 |
| 0.01 | 4.605 |

Cross-entropy is usually preferred to mean squared error for classification because it matches the probabilistic model and supplies useful gradients. For softmax plus categorical cross-entropy, the gradient with respect to a logit has the simple form:

$$
\frac{\partial L}{\partial z_i}=\hat{y}_i-y_i
$$

Accuracy alone is unsuitable as a training objective because it changes only when the winning class changes and does not provide useful gradients. Two models can have the same accuracy while assigning very different probabilities to the correct classes.

## 12. How a CNN Learns

Training occurs end to end:

1. A batch of images moves through the model in a **forward pass**.
2. The model produces predictions.
3. A loss function compares predictions with the true labels.
4. **Backpropagation** applies the chain rule to compute gradients.
5. An optimizer uses the gradients to update trainable parameters.
6. The process repeats over many batches and epochs.

Training updates both:

- the convolutional filters that determine which features are extracted; and
- the weights and biases that determine how those features are combined.

Feature maps themselves are input-dependent activations, not persistent parameters. The article occasionally speaks of feature maps being trained; more precisely, the **filters that produce them** are trained.

An **epoch** is one full pass through the training dataset. A **batch** is a subset processed before one optimizer update.

## 13. Hierarchical Feature Learning

The power of a CNN comes from composing learned representations:

$$
\text{pixels}
\rightarrow \text{edges and local patterns}
\rightarrow \text{textures and shapes}
\rightarrow \text{object parts}
\rightarrow \text{class evidence}
$$

This hierarchy is learned from data rather than manually programmed. The exact meaning of an individual activation is often difficult to interpret, and important information may be distributed across many channels.

## 14. Benefits and Limitations

### Benefits

- Learns features automatically from raw or minimally processed images.
- Exploits local spatial structure.
- Shares filters across locations, reducing parameter counts.
- Builds increasingly abstract hierarchical representations.
- Supports efficient computation on modern accelerators.

### Limitations

- Usually requires substantial labeled data or transfer learning.
- Can be computationally expensive.
- May learn shortcuts or biases present in the training data.
- Is not automatically robust to rotation, scale, lighting, occlusion, or domain shift.
- Can be confidently wrong.
- Often provides limited interpretability without additional analysis.

## 15. Technical Corrections to Remember

The article is an intuitive introduction from 2018, so several statements should be interpreted carefully:

- Real convolution computes weighted sums; it does not merely count matching cells.
- ReLU introduces non-linearity into the model's transformations rather than changing whether an image itself is linear.
- The learned objects are filters and biases; feature maps are the outputs created for particular inputs.
- Pooling offers limited local robustness, not complete invariance to angle, scale, texture, or distortion.
- Pooling can reduce model complexity but does not automatically prevent overfitting.
- Softmax outputs are normalized probabilities derived from logits, not arbitrary “real-value probabilities” before normalization.
- Loss and cost are not named according to whether the model is an ANN or CNN; terminology varies by source.
- A binary classifier does not require two softmax outputs; one sigmoid output is often simpler.
- CNNs are inspired loosely by aspects of biological vision but should not be treated as replicas of the human brain.

## 16. Compact Reference Table

| Component | Input-to-output role | Trainable parameters? |
| --- | --- | --- |
| Convolution | Detect local patterns and produce feature maps | Yes |
| ReLU | Introduce non-linearity | No |
| Pooling | Downsample spatial dimensions | No |
| Flatten | Reshape feature maps into a vector | No |
| Dense layer | Combine extracted features globally | Yes |
| Sigmoid | Convert one logit into a binary score | No |
| Softmax | Normalize class logits into a distribution | No |
| Cross-entropy | Measure disagreement with the target | No |
| Backpropagation | Compute parameter gradients | No parameters of its own |
| Optimizer | Update parameters using gradients | May maintain optimizer state |

## 17. Quick Revision Questions

1. What is the difference between a filter and a feature map?
2. Why is weight sharing useful in convolution?
3. What does ReLU contribute to a CNN?
4. What does a $2 \times 2$ max-pooling layer retain from each window?
5. Does flattening learn parameters?
6. Why can a large flattened vector be expensive?
7. When should softmax be used instead of independent sigmoid outputs?
8. Why is cross-entropy more informative than classification error during training?
9. Which CNN parameters are updated by end-to-end training?
10. Does max pooling guarantee invariance to rotation or viewpoint?

## 18. Answers

1. A filter is a learned set of weights; a feature map is the activation produced by applying that filter to an input.
2. It allows the same pattern detector to operate at every location while using far fewer parameters.
3. ReLU introduces non-linearity by replacing negative activations with zero.
4. It retains the maximum activation and discards the other values in that window.
5. No. It only reshapes the tensor.
6. Every flattened value may connect to every dense neuron, creating a very large weight matrix.
7. Use softmax when exactly one of several classes is correct; use independent sigmoids when multiple labels may be true.
8. Cross-entropy considers how much probability the model assigned to the true class and supplies differentiable gradients.
9. Convolutional filters and biases, dense weights and biases, and parameters in any other trainable layers.
10. No. It provides only limited robustness to small local shifts and distortions.

## Final Takeaway

A CNN learns **which local visual patterns matter** and **how to combine them into a prediction**. Convolution extracts features, ReLU introduces non-linearity, pooling compresses spatial information, and dense or global-pooling heads convert the learned representation into an output. Cross-entropy measures prediction quality, and backpropagation trains the entire pipeline jointly.
