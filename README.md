# Variational-Attention-and-Disentanglement-with-Regularization-for-Conversational-Question-Answering

## Abstract
With the advances in neural deep learning and the availability of human conversations on social media, the conversational agent systems have drawn increasing attention from the research community of natural language processing and artificial intelligence. The way to build a robust and intelligent conversational agent system such like chatbot is to ensure the ability of machines to understand the given knowledge and answer a series of interconnected questions that appear in a conversation. One of the most important challenges in this research is to effectively extract the information from the given knowledge and the historical conversation which are related to the current question. Currently, many studies have focused on designing the delicate attention modules to handle this challenge. Most of them adopt a kind of soft attention module in different information sources due to its simplicity and ease of optimization. However, people often dive into a different topic in open-ended conversation. Such an attention will make the model receive too much irrelevant information because soft attention modules always consider all previous conversations with different weights. Accordingly, it is important that the model needs to effectively identify the irrelevant conversations and response without any previous conversations. This study deals with these challenging issues and proposes an informative pre-trained model and a novel stochastic attention module for conversational question answering. 

This dissertation explores a unified approach to handle the aforementioned challenges with two stages. First, in order to respond effectively in conversational question answering, certain parts of the model are pre-trained for single-round question answering tasks. However, certain question answering tasks suffer from the language prior problem, which is caused by superficial correlations between question and some answer words learned during training. This problem results in the circumstances that the model will fail to ground questions in knowledge content and perform poorly in real-world settings. To alleviate the issue of language prior, this study introduces the information-theoretic latent disentanglement which learns the decomposed linguistic representation from questions. In addition, an information gained regularization scheme is developed for latent disentanglement with constraint in learning representation. Second, this study presents the Bernoulli variational attention mechanism which is a kind of stochastic attention module to handle the issues in conversational question answering. Inspired by the discrete stochastic neural networks, we present the variational attentive distribution in the solution by using local reparameterization trick which makes the training process differentiable. The parameters of this Bayesian framework are optimized by using a learned prior for regularization.

## How to train

For pretraining VQA model:

```
python main.py --pretrain
```

For training visual dialog model:

```
python main.py
```


## How to evaluate

For evaluating VQA model:

```
python evaluation.py --evaluate vqa
```

For training visual dialog model:

```
python evaluation.py --evaluate visdial
```
