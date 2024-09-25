# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class DAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, word_embeddings):
        super(DAN, self).__init__()
        self.embedding = word_embeddings.get_initialized_embedding_layer()
        self.V = nn.Linear(input_size, hidden_size)
        self.g = nn.Tanh()
        self.W = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
    
    def forward(self, x):
        output = self.embedding(x)
        x = output.mean(dim=1)
        return self.logsoftmax(self.W(self.g(self.V(x))))


class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, dan, word_embeddings):
        self.dan = dan
        self.word_embeddings = word_embeddings

    def predict(self, ex_words: List[str]) -> int:
        # Convert words to indices using the word embeddings indexer and ignore unknown words
        word_indices = []
        for word in ex_words:
            idx = self.word_embeddings.word_indexer.index_of(word)
            if idx != -1:
                word_indices.append(idx)

        # Convert the list of word indices to a tensor and add a batch dimension
        word_indices_tensor = torch.LongTensor([word_indices])  # Shape: (1, sequence_length)

        # Get the log probabilities from the model
        prediction = self.dan(word_indices_tensor)  # Shape: (1, 2)

        # Return the index of the highest probability (0 or 1)
        return torch.argmax(prediction, dim=1).item()


    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        return [self.predict(ex_words) for ex_words in all_ex_words]


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    # Initialize the DAN model
    dan = DAN(input_size=word_embeddings.get_embedding_length(),
              hidden_size=args.hidden_size,
              output_size=2,  # Output size is 2 because it's binary classification (positive/negative sentiment)
              word_embeddings=word_embeddings)
    
    # Wrap DAN into a NeuralSentimentClassifier
    model = NeuralSentimentClassifier(dan, word_embeddings)
    
    # Define the loss function (NLLLoss since we're using LogSoftmax in the forward pass)
    loss_function = nn.NLLLoss()
    
    # Define the optimizer (Adam optimizer with learning rate from args)
    optimizer = optim.Adam(dan.parameters(), lr=args.lr)
    
    # Training loop for the number of epochs
    for epoch in range(args.num_epochs):
        # Shuffle the training examples each epoch
        random.shuffle(train_exs)
        total_loss = 0.0
        
        # Batch processing
        batch_size = args.batch_size
        for batch_start in range(0, len(train_exs), batch_size):
            batch_exs = train_exs[batch_start:batch_start + batch_size]

            x = [] # list of word indices
            y = [] # list of labels
            for ex in batch_exs:
                sentence = []
                for word in ex.words:
                    idx = word_embeddings.word_indexer.index_of(word)
                    if idx != -1:
                        sentence.append(idx)
                x.append(torch.LongTensor(sentence))
                y.append(ex.label)

            # Convert the list of sentences into a tensor after padding
            x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
            y = torch.LongTensor(y)

            dan.zero_grad()
            probs = dan(x)  # x is now a padded tensor
            loss = loss_function(probs, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Print loss for every epoch
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_exs):.4f}")  # Normalize loss by total examples

    return model