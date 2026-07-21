# Introduction to Classification

## Definition

**Classification** is a machine learning technique that uses patterns learned from labeled training data to assign new observations to predefined categories, or **classes**.

Given an observation's features `X`, a classifier predicts a class label `y`:

$$
f(X) \rightarrow y
$$

For example, the features of an email may include its words, sender, links, and formatting, while its label may be `spam` or `not spam`.

## Classification versus regression

Classification and regression are both supervised learning tasks, but they predict different types of outputs:

| Task | Output | Example |
| --- | --- | --- |
| Classification | A discrete category | Will a customer stay or leave? |
| Regression | A continuous numerical value | How much will a customer spend? |

A classifier may calculate class probabilities internally, but its final prediction is usually a category. For example, a model might estimate an 82% probability that a customer will leave and then assign the label `churn`.

## Why classification is supervised learning

Classification is a form of **supervised learning** because each training observation is paired with a known target label. The algorithm compares the features with those labels and learns a relationship that can be applied to unseen observations.

For a churn model, historical customer records might contain:

- **Features:** contract length, monthly charges, service usage, complaints, and payment history.
- **Target label:** `stayed` or `left`.

During prediction, the trained model receives the features of a current customer but not the future outcome. It estimates the most likely class from the patterns learned in the historical data.

## Types of classification problems

- **Binary classification:** predicts one of two classes, such as `stay` or `leave`.
- **Multiclass classification:** predicts one of several mutually exclusive classes, such as `primary`, `promotional`, `social`, or `spam` email.
- **Multilabel classification:** assigns more than one class to an observation, such as labeling a photograph with both `dog` and `outdoors`.

## Application 1: Customer churn modeling

**Customer churn** occurs when a customer stops using a company's product or service. A churn classifier predicts whether each customer is likely to stay or leave within a specified future period.

A useful churn workflow is:

1. Collect historical customer data and known churn outcomes.
2. Train a binary classifier using those labeled examples.
3. Estimate the churn probability for active customers.
4. Identify high-risk customers.
5. Use appropriate retention measures, such as requesting feedback, improving support, or making a relevant offer.
6. Measure whether the intervention actually reduces churn.

Churn predictions help a business focus its retention resources, but the results should be used responsibly. The input data should be relevant, lawful, and checked for bias, and customers should not be manipulated or unfairly treated because of a prediction.

## Application 2: Email classification

Email services use classification to organize and protect a user's inbox. A message may be categorized as:

- Primary or normal mail
- Important or urgent
- Social
- Promotional
- Spam or phishing

Possible email features include:

- Words and phrases in the subject and body
- Sender reputation
- Number and destination of links
- Presence and type of attachments
- Message formatting and metadata
- The user's previous interactions with similar messages

Spam filtering is an example of a system that must adapt over time because malicious senders continually change their behavior. Models therefore need monitoring, feedback, and periodic retraining.

## Application 3: Image recognition

An image classifier assigns a category to an image. For example, it may distinguish photographs of dogs from photographs of cats.

The image's pixel values are the raw input features. Modern systems often learn higher-level visual patterns such as edges, shapes, textures, and object parts before producing a class prediction.

Image classification can be used for:

- Medical image analysis
- Product and defect recognition
- Handwritten character recognition
- Plant or animal identification
- Content organization and moderation

Image classification predicts a label for an entire image. This differs from **object detection**, which also locates individual objects inside the image.

## From training to prediction

A typical classification process contains four main stages:

1. **Training data:** collect observations with known labels.
2. **Learning:** fit an algorithm that relates the input features to the labels.
3. **Prediction:** use the fitted model to classify unseen observations.
4. **Evaluation:** compare predictions with the true labels using suitable metrics.

The train and test observations must remain separate. Testing a model on the same data used to train it can give an overly optimistic estimate of its ability to generalize.

## Key terminology

- **Observation (sample):** one item being classified, such as one customer or email.
- **Feature:** an input measurement or characteristic used to make a prediction.
- **Label (target):** the known category associated with a training observation.
- **Class:** one of the possible output categories.
- **Classifier:** the trained model that assigns observations to classes.
- **Decision boundary:** the dividing surface that separates predicted classes in the feature space.
- **Class probability:** the model's estimated probability that an observation belongs to a particular class.

## Key takeaway

Classification learns from labeled examples to predict categories for new observations. It supports applications ranging from customer retention and inbox filtering to medical diagnosis and image recognition. Its value comes not only from producing a class label, but from helping people make informed decisions based on the likelihood of different outcomes.
