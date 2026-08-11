# Step 4: Full Connection

After convolution, pooling, and flattening, the next step is to attach a fully connected neural network to the CNN.

The earlier stages act as a **feature extractor**:

1. **Convolution** applies learned filters to detect useful local patterns.
2. **Pooling** reduces the spatial dimensions while retaining important information.
3. **Flattening** converts the resulting feature maps into a one-dimensional feature vector.
4. **Full connection** uses those extracted features to make the final prediction.

The flattened vector becomes the input to one or more dense (fully connected) layers. These layers combine the extracted features into higher-level representations that are more useful for classification. For example, instead of relying on one isolated visual pattern, the network can learn that a particular combination of shapes, textures, and edges provides strong evidence that an image contains a dog rather than a cat.

## Why Is It Called a Fully Connected Layer?

A fully connected layer is a particular kind of hidden layer in which every neuron receives input from every neuron in the preceding layer. The terms **dense layer** and **fully connected layer** are commonly used interchangeably.

Not every hidden layer in a neural network must be fully connected. Convolutional layers, for instance, use local connections and shared weights. The name therefore distinguishes dense layers from the convolutional part of the model.

## From Extracted Features to Class Predictions

Consider a simplified classifier with:

- five input features from the flattened CNN output;
- a first fully connected layer containing six neurons;
- a second fully connected layer containing eight neurons; and
- an output layer for predicting either a dog or a cat.

Each neuron in the dense layers calculates a weighted combination of its inputs, adds a bias, and applies an activation function:

$$
z_j = \sum_i w_{ji}x_i + b_j
$$

$$
a_j = f(z_j)
$$

The weights determine how strongly each learned feature influences a neuron. During training, the model learns which features and feature combinations are most useful for each class.

The last dense layer may contain neurons that respond strongly to different combinations of visual evidence. These features are usually abstract rather than directly interpretable as simple concepts such as “whiskers,” “floppy ears,” or “a large nose.” Such labels are useful illustrations, but deep CNN features normally represent more complex learned patterns.

## Choosing the Output Layer

The correct output configuration depends on the prediction task.

### Binary classification

For two mutually exclusive classes, such as dog and cat, a common design uses one output neuron with a sigmoid activation:

$$
P(\text{dog}) = \sigma(z)
$$

The remaining probability can be interpreted as:

$$
P(\text{cat}) = 1 - P(\text{dog})
$$

This setup is commonly trained using **binary cross-entropy**.

It is also possible to use two output neurons with a softmax activation, although this is redundant for a simple binary problem.

### Multiclass classification

For three or more mutually exclusive classes—such as dog, cat, and bird—the output layer normally contains one neuron per class. A softmax activation converts the output scores, called **logits**, into probabilities that sum to one:

$$
P(y=k) = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

This setup is commonly trained using **categorical cross-entropy** or **sparse categorical cross-entropy**, depending on how the labels are encoded.

### Regression

When predicting a continuous numerical value, the network often uses a single output neuron with a linear activation. Mean squared error is one possible loss function for this kind of task.

## How the CNN Learns

Training is an end-to-end process:

1. An image passes through convolution, pooling, flattening, and the fully connected layers. This is the **forward pass**.
2. The network produces a prediction.
3. A loss function compares the prediction with the true label.
4. Backpropagation calculates how each trainable parameter contributed to the loss.
5. An optimizer, such as gradient descent or one of its variants, updates the parameters to reduce future loss.
6. The process repeats over many batches and epochs.

The updates are not random. Gradients indicate the direction in which the parameters should move to reduce the loss.

Importantly, backpropagation updates more than the weights in the fully connected layers. It also updates the convolutional filters, or feature detectors. Consequently, the network learns both:

- **which features to extract** in the convolutional layers; and
- **how to combine those features** in the fully connected layers.

Features that consistently help the model make correct predictions become more influential. Features that are unhelpful tend to receive less influence or are reshaped into more useful detectors as training continues.

## An Intuitive View of the Final Prediction

The neurons in the final hidden layer can be imagined as casting weighted “votes.” A strong activation means that a learned feature or combination of features is present. The output-layer weights determine how much that evidence matters to each class.

For example:

- some final-layer activations may provide strong evidence for a dog;
- others may provide strong evidence for a cat; and
- the same activation can affect different classes differently because its outgoing weights are different.

The output layer combines all this evidence. With softmax, it produces a probability distribution across the classes; with a single sigmoid output, it produces the probability of the positive class. The class with the strongest final probability becomes the model’s prediction.

This “voting” description is a helpful intuition, but the actual computation is a learned weighted sum followed by an activation function.

## Training vs. Inference

During **training**, the true label is available. The network uses it to calculate the loss and update its parameters through backpropagation.

During **inference**, the true label is not available to the model. The trained filters and weights are kept fixed, and a new image simply moves through the network in a forward pass. The final output probabilities are then used to select the predicted class.

## Study Notes

### Key terms

- **Feature extractor:** The convolution and pooling stages that learn useful representations from an image.
- **Flattening:** Reshaping multidimensional feature maps into a vector. It changes the shape but does not itself learn parameters.
- **Fully connected/dense layer:** A layer in which each neuron is connected to every output from the preceding layer.
- **Logit:** An output score before sigmoid or softmax converts it into a probability.
- **Loss function:** A measure of error for a training example or batch.
- **Backpropagation:** The application of the chain rule to compute gradients through the network.
- **Optimizer:** The algorithm that uses gradients to update the trainable parameters.
- **Epoch:** One complete pass through the training dataset.
- **Batch:** A subset of training examples processed before one parameter update.

### Important distinctions

| Concept | Meaning |
| --- | --- |
| Loss vs. cost | “Loss” often refers to error for one example or batch; “cost” often refers to an aggregate objective. In practice, the terms are sometimes used interchangeably. |
| Activation vs. weight | An activation is a neuron’s output for a particular input. A weight is a learned parameter controlling the influence of one value on another neuron. |
| Training vs. inference | Training uses labels and updates parameters; inference uses the learned parameters without updating them. |
| Feature map vs. feature vector | Convolution produces spatial feature maps; flattening or global pooling converts them into a representation suitable for dense layers. |
| Binary vs. multiclass output | Binary classification commonly uses one sigmoid output; multiclass classification commonly uses one softmax output per class. |

### Pipeline to remember

$$
\text{Image} \rightarrow \text{Convolution} \rightarrow \text{Pooling}
\rightarrow \text{Flattening} \rightarrow \text{Dense layers}
\rightarrow \text{Output probabilities}
$$

During training:

$$
\text{Prediction} + \text{True label}
\rightarrow \text{Loss} \rightarrow \text{Backpropagation}
\rightarrow \text{Parameter updates}
$$

### Quick self-check

1. What does flattening do, and does it contain trainable parameters?
2. Why can a fully connected layer combine features more effectively than treating each feature independently?
3. Which output activation and loss would you choose for a dog-versus-cat classifier?
4. Which output activation and loss would you choose for dog, cat, and bird classification?
5. Which parameters are updated when a CNN is trained end to end?
6. What is the difference between an activation, a weight, and a logit?

### Answers

1. Flattening reshapes feature maps into a vector; it has no trainable parameters.
2. Dense neurons learn weighted combinations of all preceding features and can therefore represent higher-level feature interactions.
3. One sigmoid output with binary cross-entropy is the usual choice.
4. Three softmax outputs with categorical or sparse categorical cross-entropy are the usual choice.
5. Both the convolutional filters and the weights and biases in the dense layers are updated.
6. An activation is a neuron’s computed output, a weight is a learned parameter, and a logit is an output-layer score before probability normalization.

## Summary

The fully connected stage transforms the features learned by the convolutional part of a CNN into a final prediction. During training, the loss is propagated backward through the entire model, allowing both the feature detectors and the classifier to improve together. Once training is complete, a new image passes through the fixed network, and the output layer reports the model’s predicted class probabilities.
