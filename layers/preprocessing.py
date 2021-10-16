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

def tokenize_and_filter(inputs, outputs, tokenizer_1, tokenizer_2, max_length, batch_size):
    tokenized_inputs, tokenized_outputs = [], []
    step = 0
    with tqdm(total=len(inputs)) as pbar:
        while step < 10000:
            sentences1 = inputs[step : step + batch_size]
            sentences2 = outputs[step : step + batch_size]

            # tokenize sentence
            tokenized_inputs.extend(tokenizer_1(sentences1, padding=True, truncation=True, return_tensors="tf", max_length=max_length)['input_ids'])
            tokenized_outputs.extend(tokenizer_2(sentences2, padding=True, truncation=True, return_tensors="tf", max_length=max_length)['input_ids'])
            step += batch_size
            pbar.update(batch_size)
    return tokenized_inputs, tokenized_outputs
