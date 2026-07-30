# Types of Natural Language Processing

Natural language processing (NLP) and deep learning are related fields, but they are not the same. A Venn diagram provides a useful way to understand their relationship and the areas covered in this course.

## Natural Language Processing

Natural language processing includes algorithms and models that enable computers to process, analyze, interpret, or generate human language. An NLP technique does not necessarily use a neural network; traditional statistical and rule-based methods also belong to this field.

Examples of NLP tasks include:

- Sentiment analysis
- Text classification
- Information extraction
- Question answering
- Text summarization
- Machine translation

## Deep Learning

Deep learning includes models based on multi-layer neural networks. It is a much broader field than NLP because deep learning can be applied to many kinds of data and problems, including:

- Image classification
- Object detection
- Speech recognition
- Time-series forecasting
- Natural language processing

Therefore, a deep learning model is not automatically an NLP model.

## Deep NLP

The overlap between NLP and deep learning is often called **deep NLP**. This area contains neural-network-based models designed to process or generate natural language.

Deep NLP models can learn useful representations and complex patterns directly from data. This often makes them more capable than traditional NLP approaches, especially when large, high-quality datasets and sufficient computing resources are available.

The course examines both traditional NLP and deep NLP so that their approaches can be compared and their development over time can be understood.

> **Note:** The sizes of the circles in the Venn diagram are illustrative only. They do not represent the relative importance, scope, or number of techniques in each field.

## Sequence-to-Sequence Models

Sequence-to-sequence models form an important subfield of deep NLP. A sequence-to-sequence model receives one sequence as input and produces another sequence as output. The input and output sequences may have different lengths.

Common applications include:

- Neural machine translation
- Text summarization
- Question answering
- Conversational agents and chatbots
- Speech recognition
- Image captioning, when an image representation is used to generate a word sequence

The course progresses from traditional NLP techniques to deep NLP and then to sequence-to-sequence models. Although chatbots are a central application, the underlying models are useful across many other language and sequence-processing tasks.

## Study Notes

### Core relationship

```text
NLP ∩ Deep Learning = Deep NLP
Deep NLP ⊃ Sequence-to-Sequence Models
```

- **NLP** is defined by the problem domain: working with human language.
- **Deep learning** is defined by the modelling approach: using deep neural networks.
- **Deep NLP** applies deep neural networks to language problems.
- **Sequence-to-sequence** models map an input sequence to an output sequence.

### Important distinctions

| Area | Main focus | Must use a neural network? | Example |
|---|---|---:|---|
| Traditional NLP | Processing human language | No | Bag-of-words sentiment classifier |
| Deep learning | Learning with deep neural networks | Yes | Image classifier |
| Deep NLP | Applying deep learning to language | Yes | Neural text classifier |
| Sequence-to-sequence | Transforming one sequence into another | Yes, in this course context | Machine translation |

### Sequence-to-sequence intuition

Given an input sequence:

```text
"How are you?"
```

a model might produce:

```text
"Comment allez-vous ?"
```

The model learns a mapping from the complete input sequence to an appropriate output sequence instead of predicting only a single class.

### Historical and modern context

Classic sequence-to-sequence systems commonly used recurrent neural networks such as LSTMs or GRUs, often with an encoder-decoder architecture and attention. Modern NLP systems usually use **Transformers**, which also support sequence-to-sequence learning and form the foundation of many contemporary language models.

### Quick revision questions

1. Why is every deep NLP model an NLP model, but not every NLP model a deep NLP model?
2. Can a deep learning model operate outside NLP? Give two examples.
3. What distinguishes a sequence-to-sequence task from ordinary classification?
4. Why can the input and output of a sequence-to-sequence model have different lengths?
5. Name three applications of sequence-to-sequence learning.

### Key takeaway

NLP describes **what kind of data and problem** a model handles, while deep learning describes **how the model learns**. Their overlap is deep NLP, and sequence-to-sequence learning is an important approach within that overlap.
