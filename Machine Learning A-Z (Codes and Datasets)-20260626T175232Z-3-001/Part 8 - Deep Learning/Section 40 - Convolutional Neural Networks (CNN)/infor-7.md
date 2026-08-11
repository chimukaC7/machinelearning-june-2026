# Convolutional Neural Networks: Section Summary

This section introduced the complete workflow of a convolutional neural network (CNN), from receiving an input image to producing a class prediction.

## The Complete CNN Pipeline

### 1. Convolution

The process begins with an input image. The convolutional layer applies multiple learnable **filters**, also called **kernels** or **feature detectors**, to the image. Each filter produces a **feature map** that highlights a particular kind of pattern, such as an edge, texture, shape, or a more abstract feature.

Early convolutional layers typically detect simple local patterns. Deeper layers combine these patterns into increasingly complex representations.

### 2. ReLU Activation

After convolution, the feature maps are commonly passed through the **Rectified Linear Unit (ReLU)** activation function:

$$
\operatorname{ReLU}(x) = \max(0, x)
$$

ReLU introduces non-linearity into the model. Without a non-linear activation, stacking convolutional layers would still behave like a single linear transformation and would severely limit what the network could learn.

ReLU preserves positive values and replaces negative values with zero, allowing the network to model complex visual relationships efficiently.

### 3. Pooling

A pooling layer reduces the spatial dimensions of each feature map. In this section, the pooling operation used was **max pooling**, which keeps the largest value within each local window.

Pooling offers several benefits:

- it reduces the amount of data passed to later layers;
- it lowers computation and memory requirements;
- it provides some robustness to small translations and local distortions;
- it increases the effective receptive field of later neurons; and
- by reducing the representation size, it can help limit overfitting.

Pooling does not make a model completely invariant to rotations, large shifts, or major changes in viewpoint. Instead, it provides limited local robustness while retaining prominent detected features.

### 4. Flattening

The pooled feature maps are then converted into a one-dimensional vector. This operation is called **flattening**.

Flattening changes only the arrangement of the values; it does not learn any parameters. The resulting vector contains the features extracted by the preceding convolutional and pooling layers.

For example, feature maps with the shape $H \times W \times C$ are reshaped into a vector containing:

$$
H \times W \times C
$$

values.

### 5. Fully Connected Layers

The flattened feature vector is passed into one or more fully connected, or **dense**, layers. These layers learn how to combine the extracted features to distinguish between the target classes.

The final output layer produces the prediction. Its structure depends on the task:

- binary classification commonly uses one sigmoid output;
- multiclass classification commonly uses one softmax output per class; and
- regression commonly uses one or more linear outputs.

The neurons in the final hidden layer can be viewed as providing weighted evidence for the possible classes. The output layer combines this evidence to determine the predicted probability of each class.

## End-to-End Training

A CNN learns through repeated forward and backward passes.

During the **forward pass**:

1. the image passes through the convolutional layers;
2. activation functions introduce non-linearity;
3. pooling reduces the feature maps;
4. the representation is flattened; and
5. the dense layers produce a prediction.

A loss function then compares the prediction with the correct label. During **backpropagation**, gradients flow backward through the entire network. An optimizer uses those gradients to update the trainable parameters.

This means the network trains both:

- the weights and biases in the fully connected layers; and
- the convolutional filters responsible for extracting features.

The filters are not manually chosen and then held fixed. They are refined through gradient-based optimization so that the network learns the features most useful for its task.

Training repeats across many batches and epochs until the model develops a useful mapping from input images to predictions.

## CNNs as an Architecture Problem

Designing a neural network is an architectural and experimental process. A practitioner must decide, among other things:

- how many convolutional blocks to use;
- how many filters each layer should contain;
- which kernel and pooling sizes to use;
- where to apply normalization, regularization, or dropout;
- whether to flatten or use global pooling;
- how many dense layers to include; and
- which optimizer, loss function, and learning rate to use.

There is rarely one universally optimal design. Architecture choices depend on the dataset, task, computational budget, and desired balance between accuracy and efficiency. Effective models are developed by combining sound principles with careful experimentation and validation.

## Suggested Further Reading

The 2016 article by Adit Deshpande, *The 9 Deep Learning Papers You Need to Know About (Understanding CNNs Part 3)*, surveys influential CNN research and architectures. Studying established networks can reveal why particular architectural choices were made and how CNN design evolved.

Some papers may initially contain unfamiliar ideas. It can be helpful to revisit them after completing the practical exercises and gaining experience training CNNs.

## Study Notes

### Pipeline at a glance

$$
\text{Image}
\rightarrow \text{Convolution}
\rightarrow \text{ReLU}
\rightarrow \text{Pooling}
\rightarrow \text{Flattening}
\rightarrow \text{Dense layers}
\rightarrow \text{Prediction}
$$

Convolution, activation, and pooling blocks are usually repeated several times before the final classifier.

### What each stage contributes

| Stage | Main role | Trainable parameters? |
| --- | --- | --- |
| Convolution | Detects useful local patterns | Yes: filters and biases |
| ReLU | Introduces non-linearity | No |
| Pooling | Downsamples feature maps | No in standard max or average pooling |
| Flattening | Reshapes feature maps into a vector | No |
| Dense layer | Combines extracted features | Yes: weights and biases |
| Output layer | Produces scores, probabilities, or values | Yes: weights and biases |

### Shape example

Suppose a layer produces 32 feature maps, each measuring $14 \times 14$.

After flattening, the dense layer receives:

$$
14 \times 14 \times 32 = 6{,}272
$$

input values.

If this vector connects to a dense layer with 128 neurons, that layer alone contains:

$$
(6{,}272 \times 128) + 128 = 802{,}944
$$

trainable parameters. The extra 128 parameters are the biases. This illustrates why flattening can create a large model and why modern CNNs sometimes use **global average pooling** instead.

### Important corrections and nuances

- ReLU stands for **Rectified Linear Unit**, not “Reified Linear Unit.”
- ReLU does not remove linearity from an image; it introduces non-linearity into the network's transformations.
- The useful term is **spatial invariance** or, more precisely, limited robustness to small spatial changes.
- Max pooling can reduce overfitting risk, but it does not guarantee that overfitting will be prevented.
- Backpropagation calculates gradients; the optimizer uses those gradients to update parameters.
- One pass through the complete training dataset is an **epoch**. One parameter update normally occurs per **batch**, not per epoch.

### Parameters learned during training

In a typical CNN, training can update:

- convolutional filter values;
- convolutional biases;
- dense-layer weights;
- dense-layer biases; and
- trainable parameters in other layers, such as batch-normalization scale and shift values.

ReLU, ordinary pooling, and flattening do not contain trainable parameters.

### Questions for revision

1. What is the difference between a filter and a feature map?
2. Why does a CNN require non-linear activation functions?
3. What information does max pooling preserve, and what information can it discard?
4. Does flattening learn any parameters?
5. Why can flattening cause the parameter count to grow rapidly?
6. Which parts of a CNN are updated during end-to-end training?
7. What is the difference between backpropagation and an optimizer?

### Answers

1. A filter is a learned matrix of weights; a feature map is the result of applying that filter to an input.
2. Non-linear activations allow the model to learn complex relationships that cannot be represented by stacked linear transformations alone.
3. Max pooling preserves the strongest activation in each window but discards exact locations and lower-valued activations.
4. No. Flattening only reshapes the data.
5. Every flattened value may connect to every dense neuron, producing a very large weight matrix.
6. Trainable convolutional filters, biases, dense-layer weights, and any other trainable parameters are updated.
7. Backpropagation computes the gradients; the optimizer decides how to use those gradients to change the parameters.

## Final Takeaway

A CNN first learns **what visual features to detect** and then learns **how to combine those features to make a prediction**. Because the entire network is trained jointly, the feature extractor and classifier improve together. Once trained, the model can apply this learned representation to previously unseen images and assign them to the appropriate classes.
