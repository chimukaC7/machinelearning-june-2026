# Classical NLP vs Deep Learning Models

Natural language processing (NLP) and deep learning overlap, but they describe different things:

- **NLP** is the broad field concerned with processing, understanding, and generating human language.
- **Deep learning** is a family of machine-learning techniques based on multi-layer neural networks.
- **Deep NLP** is the intersection of these fields: it applies deep neural networks to language problems.
- **Sequence-to-sequence models** are an important family within deep NLP. They transform an input sequence into an output sequence.

Understanding the progression from classical NLP to deep NLP makes it easier to see why modern models were developed, what problems they solve, and which applications they support.

## 1. Rule-Based Systems

Early chatbots frequently relied on handcrafted `if/else` rules. The developer created a collection of expected inputs and associated each one with a predetermined response.

For example:

```text
IF input contains "opening hours"
THEN respond with "We are open from 08:00 to 17:00."
```

This approach is deterministic and easy to understand. It can work well when the domain is narrow and the possible requests are predictable. However, it does not naturally understand meaning. A rule will fail when the user expresses a familiar intention in an unexpected way unless that variation has also been anticipated.

### Advantages

- Simple to implement for small, controlled tasks
- Predictable and easy to audit
- Requires little or no training data
- Useful when responses must follow strict business rules

### Limitations

- Requires rules to be written and maintained manually
- Handles unfamiliar wording poorly
- Becomes difficult to manage as the number of rules grows
- Produces rigid conversations rather than flexible, personalized responses

Rule-based systems belong to classical NLP because they process language without using deep neural networks.

## 2. Frequency-Based Speech Analysis

Traditional speech-recognition pipelines can analyze the frequency components of an audio signal without using deep learning. A time-domain speech waveform can be converted into a frequency-domain representation using mathematical operations such as the **Fourier transform**.

In a simplified pipeline:

1. Capture the speech signal.
2. Divide the signal into short time frames.
3. Calculate spectral features that describe the frequencies in each frame.
4. Compare or model those features using an acoustic and language-processing pipeline.
5. Decode the most likely sequence of sounds, words, or characters.

The Fourier transform is only one part of this process; frequencies alone do not directly identify complete words. Classical automatic speech recognition often used engineered features, such as Mel-frequency cepstral coefficients (MFCCs), with statistical models such as hidden Markov models (HMMs) and Gaussian mixture models (GMMs).

Speech recognition sits at the intersection of signal processing and language technology. A traditional system can perform it without a deep neural network, whereas modern speech systems commonly learn acoustic representations using deep learning.

## 3. Bag-of-Words for Text Classification

The **bag-of-words (BoW)** representation converts text into numeric feature vectors. It creates a vocabulary from the training documents and records how often each vocabulary term appears in each document.

Suppose a training set contains comments labelled as either a pass (`1`) or fail (`0`):

| Comment | Label |
|---|---:|
| Great job | 1 |
| Amazing work | 1 |
| Very well written | 1 |
| Poor effort | 0 |
| Try harder next time | 0 |

After tokenization, the vocabulary might include:

```text
[amazing, effort, great, harder, job, poor, try, well, work, written]
```

Each comment is represented by a vector containing the count, or sometimes simply the presence, of each word. A classifier such as logistic regression, naive Bayes, or a support vector machine can then learn which features are associated with each class.

> The bag-of-words representation does not itself make the prediction. It supplies features to a separate classification model.

For example, words such as *great* and *amazing* may provide evidence for the positive class, while *poor* may provide evidence for the negative class. The trained classifier can use these patterns to predict the label of a new comment.

### Advantages

- Simple, fast, and interpretable
- Often effective on small or medium-sized text-classification problems
- Works with many standard machine-learning algorithms
- Provides a strong baseline against which more complex models can be compared

### Limitations

- Discards word order
- Does not naturally capture context or sentence structure
- Treats related words as separate features
- Produces sparse, high-dimensional vectors for large vocabularies

Variations such as n-grams preserve some local word order, while TF-IDF adjusts raw counts to reduce the influence of very common terms.

## 4. Convolutional Neural Networks for Text

A convolutional neural network (CNN) is best known for image processing, but it can also be applied to text classification and other language tasks.

The first step is to convert tokens into dense numerical vectors called **word embeddings**. A sentence then becomes a matrix:

```text
number of tokens × embedding dimensions
```

Convolutional filters slide across groups of adjacent token embeddings and learn local patterns. In text, these patterns can behave like learned n-gram detectors. A pooling operation—commonly max pooling—retains the strongest signal produced by each filter. The pooled features are passed to one or more output layers to make a prediction.

A simplified text-CNN pipeline is:

```text
Text
  ↓
Tokenization
  ↓
Word embeddings
  ↓
One-dimensional convolution
  ↓
Pooling
  ↓
Classification layer
  ↓
Predicted class
```

Unlike bag-of-words, a CNN can learn useful local combinations of words and train its features jointly with the classifier. Because it uses neural-network layers, a CNN applied to language belongs to deep NLP.

CNNs can model local context efficiently, but recurrent models and, later, Transformers became more prominent for tasks that require longer-range dependencies or sequence generation.

## 5. Sequence-to-Sequence Models

A sequence-to-sequence model maps an input sequence to an output sequence. The two sequences do not need to have the same length.

Examples include:

- Translating a sentence from one language to another
- Generating an answer from a question
- Producing a summary from a document
- Converting speech features into a text transcription
- Generating a chatbot response from conversation history

Historically, a common architecture used:

1. An **encoder** to process the input sequence.
2. A representation of the input and its relevant context.
3. A **decoder** to generate the output one element at a time.

Early neural sequence-to-sequence systems commonly used recurrent neural networks (RNNs), especially LSTMs or GRUs. Attention mechanisms improved these systems by allowing the decoder to focus on relevant parts of the input during generation.

Modern sequence-to-sequence systems commonly use the Transformer architecture. Transformers rely on attention rather than recurrence and are well suited to parallel training and long-range contextual relationships.

## Comparing the Approaches

| Approach | Category | Representation or mechanism | Learns features automatically? | Typical use |
|---|---|---|---:|---|
| `if/else` rules | Classical NLP | Handwritten patterns and responses | No | Narrow-domain chatbot |
| Frequency-based analysis | Classical signal processing / speech technology | Engineered spectral features | Usually no | Traditional speech recognition |
| Bag-of-words + classifier | Classical NLP / machine learning | Word counts or TF-IDF | Partly; the classifier learns weights | Sentiment or document classification |
| Text CNN | Deep NLP | Embeddings and learned convolutional filters | Yes | Sentence classification |
| RNN sequence-to-sequence | Deep NLP | Recurrent encoder-decoder, often with attention | Yes | Translation or response generation |
| Transformer | Deep NLP | Self-attention and learned embeddings | Yes | Translation, summarization, and generation |

## Study Notes

### Classical models vs deep models

Classical approaches usually depend more heavily on human-designed rules or features. Deep learning models learn representations and task-specific patterns from data through neural-network training.

This is not a guarantee that a deep model will always be better. The appropriate method depends on:

- The amount and quality of available data
- The complexity of the task
- Available computing resources
- Latency and memory constraints
- Interpretability requirements
- The cost of development and maintenance

For a small, well-defined problem, a rule-based or bag-of-words system may be faster, cheaper, and easier to explain. Deep models become especially useful when the language patterns are complex and sufficient data and computing resources are available.

### Representation matters

```text
Rules:           language → manually matched pattern
Bag-of-words:    language → sparse count vector
Text CNN:        language → dense embeddings → local learned features
Sequence model:  input sequence → contextual representation → output sequence
```

The move from manually specified features toward learned representations is one of the central changes from classical NLP to deep NLP.

### Classification vs sequence generation

- A **classifier** produces a fixed category, such as `positive`, `negative`, `pass`, or `fail`.
- A **sequence-to-sequence model** produces an ordered sequence of output elements, such as words or tokens.

For example:

```text
Sentiment classification:
"The lesson was excellent." → positive

Machine translation:
"The lesson was excellent." → "La leçon était excellente."
```

### Important corrections and cautions

- A Fourier transform reveals frequency information; it does not independently recognize words.
- Bag-of-words is a text representation, not a complete classifier.
- Bag-of-words ignores order unless it is extended with features such as n-grams.
- Word embeddings preserve dense learned relationships between tokens; they are different from sparse word-count vectors.
- CNNs process local token patterns, while sequence models are designed to represent ordered dependencies.
- Deep learning is powerful, but greater complexity does not automatically make it the best choice for every problem.

### Quick revision questions

1. Why do rule-based chatbots struggle with unexpected wording?
2. What information does a Fourier transform reveal about an audio signal?
3. Why is bag-of-words considered an order-independent representation?
4. Which component makes a prediction after bag-of-words features have been created?
5. How is a sentence represented before it is processed by a text CNN?
6. What kinds of language patterns can convolutional filters learn?
7. How does classification differ from sequence-to-sequence generation?
8. What roles do the encoder, attention mechanism, and decoder play?
9. Why might a classical model still be preferable to a deep model?
10. Which architecture dominates modern sequence modelling?

### Key takeaway

Classical NLP relies mainly on explicit rules or engineered representations, while deep NLP learns representations and patterns using neural networks. Sequence-to-sequence architectures extend deep NLP from predicting a fixed label to generating a complete output sequence.
