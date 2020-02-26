# MIT License
# 
# Copyright (c) 2020 Clement Viricel
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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


def parse_json_line_skills(filepath, lang_1, lang_2):
    # Skills
    file_1 = os.path.join(filepath, "Skills/{}.json".format(lang_1))
    file_2 = os.path.join(filepath, "Skills/{}.json".format(lang_2))
    with open(file_1, encoding='utf-8') as f:
        lines = f.read().splitlines()
        json_lines_1 = {}
        for i in range(1, len(lines)):
            try:
                json_lines_1[json.loads(lines[i])['id']] = json.loads(lines[i])['text']
            except json.decoder.JSONDecodeError as e:
                raise SystemExit("Json decoder error at line " + str(i + 1) + " : " + str(e))
    with open(file_2, encoding='utf-8') as f:
        lines = f.read().splitlines()
        json_lines_2 = {}
        for i in range(1, len(lines)):
            try:
                json_lines_2[json.loads(lines[i])['id']] = json.loads(lines[i])['text']
            except json.decoder.JSONDecodeError as e:
                raise SystemExit("Json decoder error at line " + str(i + 1) + " : " + str(e))
    eval_input = []
    eval_output = []
    for id_ in json_lines_1:
        eval_input.append(json_lines_1[id_])
        eval_output.append(json_lines_2[id_])

    # JOBS
    file_1 = os.path.join(filepath, "Jobs/{}.json".format(lang_1))
    file_2 = os.path.join(filepath, "Jobs/{}.json".format(lang_2))
    with open(file_1, encoding='utf-8') as f:
        lines = f.read().splitlines()
        json_lines_1 = {}
        for i in range(1, len(lines)):
            try:
                json_lines_1[json.loads(lines[i])['id']] = json.loads(lines[i])['text']
            except json.decoder.JSONDecodeError as e:
                raise SystemExit("Json decoder error at line " + str(i + 1) + " : " + str(e))
    with open(file_2, encoding='utf-8') as f:
        lines = f.read().splitlines()
        json_lines_2 = {}
        for i in range(1, len(lines)):
            try:
                json_lines_2[json.loads(lines[i])['id']] = json.loads(lines[i])['text']
            except json.decoder.JSONDecodeError as e:
                raise SystemExit("Json decoder error at line " + str(i + 1) + " : " + str(e))
    for id_ in json_lines_1:
        eval_input.append(json_lines_1[id_])
        eval_output.append(json_lines_2[id_])
    return eval_input, eval_output
