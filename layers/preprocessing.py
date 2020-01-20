import tensorflow as tf
import os
import json
from tqdm import tqdm
# Tokenize, filter and pad sentences


def tokenize_and_filter(inputs, outputs, tokenizer_1, tokenizer_2, max_length):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in tqdm(zip(inputs, outputs), total=len(inputs)):
        # tokenize sentence
        sentence1 = [tokenizer_1.vocab_size] + tokenizer_1.encode(sentence1.strip()) + [tokenizer_1.vocab_size + 1]
        sentence2 = [tokenizer_2.vocab_size] + tokenizer_2.encode(sentence2.strip()) + [tokenizer_2.vocab_size + 1]
        # check tokenized sentence max length
        if len(sentence1) <= max_length and len(sentence2) <= max_length:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    # pad tokenized sentences
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_inputs, maxlen=max_length, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_outputs, maxlen=max_length, padding='post')

    return tokenized_inputs, tokenized_outputs
