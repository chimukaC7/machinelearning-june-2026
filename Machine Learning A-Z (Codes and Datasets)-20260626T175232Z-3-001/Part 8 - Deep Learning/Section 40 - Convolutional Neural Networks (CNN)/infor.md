# Introduction to Convolutional Neural Networks (CNNs)

Today, we begin our study of **convolutional neural networks**, commonly called **CNNs** or **ConvNets**. CNNs are especially effective at processing images because they learn to recognize visual features and use those features to classify what they see.

## Visual Perception and Feature Recognition

Consider an ambiguous image that can be interpreted in two different ways. From one side, it may appear to show a person looking directly at you; from the other, it may look like a person facing to the right. Your interpretation changes depending on which visual details are closest to your focus.

This illustrates an important idea: the brain interprets visual information by detecting **features**. Depending on which features receive the most attention, the brain may classify the same image in different ways.

The famous **young woman–old woman illusion** demonstrates the same effect. One interpretation shows a young woman looking into the distance: her face, hair, coat, and feather form the relevant features. The other interpretation shows an older woman looking downward: the same lines become her nose, mouth, chin, and headscarf. The image does not change, but the features selected by the brain do.

Another classic example is the **duck–rabbit illusion**. The same shape can be interpreted as either a duck or a rabbit, depending on how its features are grouped. Some ambiguous images are even harder to resolve because the competing interpretations appear equally plausible. In those cases, perception may alternate rapidly between them.

These illusions show that visual recognition is not simply a matter of recording everything in front of us. The brain detects patterns, selects features, and uses them to form a likely interpretation.

This also explains everyday recognition errors. When you glance at something quickly, you might mistake a cat for a ball or a shadow for a car. You may not have had enough time—or enough useful visual information—to identify the correct features.

CNNs process images in a broadly similar way. Rather than being manually told every feature to look for, a trained CNN learns useful patterns from data and combines them to make predictions.

## How an Image Classifier Reports Predictions

In an image-classification experiment discussed by Geoffrey Hinton, a CNN was shown several images and asked to predict their labels.

For a photograph of a cheetah, the network’s leading predictions included:

- Cheetah
- Leopard
- Snow leopard
- Egyptian cat

The correct label was **cheetah**, and it received the highest predicted probability.

For an image of a bullet train, the leading predictions included:

- Bullet train
- Passenger car
- Subway train
- Electric locomotive

Again, the correct class received the highest score.

A more ambiguous image showed a hand lens, but its shape also resembled scissors, a frying pan, or a stethoscope. The network ranked **scissors** first and the correct class, **hand lens**, second. The scores were closer together, indicating greater uncertainty.

When reading this type of classification result, distinguish between:

1. **The ground-truth label** — the correct label assigned to the image in the dataset.
2. **The model’s predictions** — the classes the network considers most likely.
3. **The prediction scores or probabilities** — values representing the model’s confidence across the possible classes.

A model can produce a confident prediction and still be wrong. A confidence score describes the model’s own estimate; it does not guarantee correctness.

## Why CNNs Matter

CNNs have become important because image recognition supports many practical applications, including:

- Detecting pedestrians, vehicles, traffic lights, and road signs for assisted or autonomous driving
- Recognizing and identifying faces in photographs
- Classifying objects and scenes
- Analyzing medical images
- Reading handwriting and printed documents
- Detecting defects in industrial images
- Recognizing facial expressions and other visual patterns

Two influential researchers in deep learning and computer vision are **Geoffrey Hinton** and **Yann LeCun**. Hinton made foundational contributions to neural networks and deep learning. LeCun made major contributions to convolutional neural networks, particularly through work on handwritten-document recognition in the 1980s and 1990s.

## The Basic CNN Task

At a high level, an image classifier follows a simple flow:

> Input image → convolutional neural network → output class scores

During training, the CNN receives examples paired with their correct labels. It learns parameter values that help it associate visual patterns with those labels. After training, it can process an unseen image and estimate the probability of each known class.

For example, a CNN trained to recognize facial expressions might assign high probability to **happy** for a smiling face and high probability to **sad** for a frowning face. If the expression is unclear, the output probabilities may be less decisive.

## How Images Are Represented Digitally

Before a neural network can process an image, the image must be represented numerically.

### Grayscale Images

A grayscale image can be represented as a two-dimensional array (matrix):

$$
\text{image shape} = \text{height} \times \text{width}
$$

In a common 8-bit representation, each pixel has an integer value from **0 to 255**, giving **256 possible intensity values**:

- `0` represents black.
- `255` represents white.
- Values between them represent different shades of gray.

For example, a $2 \times 2$ grayscale image contains four intensity values:

```text
[[  0, 128],
 [200, 255]]
```

### Colour Images

A standard RGB colour image is commonly represented as a three-dimensional array:

$$
\text{image shape} = \text{height} \times \text{width} \times \text{channels}
$$

The three channels are:

- Red (R)
- Green (G)
- Blue (B)

Each pixel therefore contains three intensity values. For an 8-bit RGB image, each channel normally ranges from 0 to 255. Combining the three values produces the pixel’s colour. For example:

```text
Red:   [255, 0,   0]
Green: [0,   255, 0]
Blue:  [0,   0, 255]
```

Although we see shapes and colours, the network receives arrays of numbers. In many machine-learning pipelines, these values are converted to floating-point numbers and normalized—for example, from the range 0–255 to 0–1—before being passed to the model.

## Simplified Binary Images

To explain CNN operations by hand, it is useful to simplify an image so that every pixel is either white or black. For example, we might define:

- `0` = white
- `1` = black

A small smiling face can then be represented as a matrix of zeros and ones. This is only a teaching simplification. The same CNN concepts extend to grayscale and RGB images with many possible intensity values.

## The Main Stages of a Basic CNN

The introductory CNN architecture in this section consists of four main stages:

1. **Convolution**
2. **Max pooling**
3. **Flattening**
4. **Full connection**

### 1. Convolution

A small matrix called a **filter** or **kernel** moves across the input image. At each position, it computes a weighted combination of nearby pixel values. The resulting matrix is called a **feature map**.

During training, the network learns filter weights that respond to useful patterns. Early layers often learn simple local features such as edges, lines, corners, and textures. Deeper layers can combine these into more complex features such as shapes, object parts, and eventually whole-object patterns.

### 2. Max Pooling

Max pooling reduces the spatial dimensions of a feature map. For each small region, it keeps only the largest value.

This can:

- Reduce computation and memory use
- Retain the strongest detected feature in each region
- Make the representation less sensitive to small shifts in the feature’s position

### 3. Flattening

Flattening converts the multidimensional feature maps into a one-dimensional vector. This allows the extracted features to be passed into standard fully connected layers.

### 4. Full Connection

Fully connected, or **dense**, layers combine the extracted features to produce the final class scores. For a multi-class classification problem, a softmax output layer is often used to convert the scores into a probability distribution across the classes.

## Suggested Reading

A foundational paper is:

> Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, **“Gradient-Based Learning Applied to Document Recognition”** (1998).

The paper describes convolutional neural networks for document and handwriting recognition and includes the influential LeNet-5 architecture. It is an excellent historical reference for understanding the development of modern CNNs.

---

# Study Notes

## Core Idea

A CNN learns **which visual features are useful** for a task and **how to combine them** to make a prediction. Unlike traditional approaches that depend heavily on hand-designed features, CNNs learn feature detectors from training data.

## Essential Vocabulary

| Term | Meaning |
|---|---|
| Pixel | The smallest numerical element of a digital image |
| Channel | One component of an image, such as red, green, or blue |
| Feature | A useful visual pattern, such as an edge, texture, or shape |
| Kernel/filter | A small learned matrix applied across an input |
| Feature map | The output produced by applying a filter to an input |
| Convolution | The operation used to detect local patterns throughout an image |
| Pooling | A downsampling operation that summarizes small spatial regions |
| Flattening | Converting multidimensional feature maps into a vector |
| Dense layer | A layer in which every input is connected to every output unit |
| Ground truth | The correct target label supplied by the dataset |
| Inference | Using a trained model to make predictions on new inputs |

## Shape Cheat Sheet

- Grayscale image: $H \times W$
- RGB image: $H \times W \times 3$
- A batch of RGB images: $N \times H \times W \times 3$ in channels-last format
- Some frameworks instead use channels-first format: $N \times 3 \times H \times W$

Where:

- $N$ = batch size
- $H$ = image height
- $W$ = image width
- $3$ = RGB channels

## CNN Pipeline at a Glance

```text
Image
  ↓
Convolution + activation
  ↓
Feature maps
  ↓
Pooling
  ↓
More convolution/pooling layers (optional)
  ↓
Flattening or global pooling
  ↓
Dense/output layer
  ↓
Class scores or probabilities
```

## Key Points to Remember

- A CNN receives numbers, not visual meaning; meaning is learned from labelled examples.
- Convolution exploits **local connectivity**: nearby pixels often form meaningful patterns.
- The same filter is reused across the image. This is called **parameter sharing**.
- Early CNN layers generally detect simple features; deeper layers combine them into more abstract patterns.
- Pooling reduces spatial resolution while retaining important activations.
- Predicted probability is not the same as certainty or correctness.
- Similar-looking classes tend to receive similar scores, particularly when an image is blurry, cropped, unusual, or ambiguous.
- The final output classes are limited to the task and labels used to train the model.

## Common Misconceptions

**“A confidence of 95% means the model is correct 95% of the time.”**  
Not necessarily. Confidence refers to the score for one prediction. Whether scores reflect real-world accuracy depends on model calibration and evaluation over many examples.

**“CNNs understand an image exactly as humans do.”**  
CNNs can learn hierarchical visual features, which offers a useful analogy to biological vision, but their mechanisms and understanding are not identical to those of the human brain.

**“Pooling is always required.”**  
Pooling is common in introductory CNNs, but modern architectures may downsample using strided convolutions or use other design choices.

**“Flattening is always required.”**  
Some architectures use global average pooling instead of flattening before the classifier.

## Self-Check Questions

1. Why can the same ambiguous image produce two different human interpretations?
2. What is the difference between a ground-truth label and a model prediction?
3. How is a grayscale image represented numerically?
4. What do the three channels of an RGB image represent?
5. What is the purpose of a convolutional filter?
6. What does a feature map contain?
7. Why is pooling used?
8. What happens during flattening?
9. Why might a CNN assign similar probabilities to scissors and a hand lens?
10. Why does a high confidence score not guarantee that a prediction is correct?

## Short Answers

1. Different groups of visual features support different plausible classifications.
2. Ground truth is the expected correct class; a prediction is the model’s estimated class or distribution of class scores.
3. As a height-by-width matrix of pixel intensities, commonly ranging from 0 to 255 for 8-bit images.
4. Red, green, and blue intensity values.
5. To detect a local pattern wherever it appears in the image.
6. The strength and spatial location of the pattern detected by a filter.
7. To reduce spatial size, computation, and sensitivity to small positional changes while preserving strong activations.
8. Multidimensional feature maps are reshaped into a one-dimensional vector.
9. Their visible shapes may share features, and the image may not contain enough clear information to separate the classes confidently.
10. Confidence expresses the model’s estimate, and models can be mistaken or poorly calibrated.

## One-Sentence Summary

> A convolutional neural network transforms pixel arrays into increasingly meaningful feature representations and uses those learned features to estimate the most likely class.
