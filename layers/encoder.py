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
from .multi_head_attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding


class Encoder_layer(tf.keras.layers.Layer):
    def __init__(self, units, d_model, num_heads, dropout, name="encoder_layer"):
        super(Encoder_layer, self).__init__(name=name)
        self.dropout = dropout
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads

        self.mha = MultiHeadAttention(d_model, num_heads, name="attention")

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.drop_1 = tf.keras.layers.Dropout(rate=dropout)
        self.drop_2 = tf.keras.layers.Dropout(rate=dropout)

        self.dense_relu = tf.keras.layers.Dense(units=units, activation='relu')
        self.dense = tf.keras.layers.Dense(units=d_model)

    def call(self, inputs, padding_mask):
        attention, attention_weights = self.mha(inputs, inputs, inputs, padding_mask)
        attention = self.drop_1(attention)
        attention = self.norm_1(inputs + attention)

        outputs = self.dense_relu(attention)
        outputs = self.dense(outputs)
        outputs = self.drop_2(outputs)
        outputs = self.norm_2(attention + outputs)

        return outputs, attention_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.emb = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(vocab_size, d_model)
        self.drop = tf.keras.layers.Dropout(rate=dropout)
        self.encoder_layers = []
        for i in range(num_layers):
            self.encoder_layers.append(
                Encoder_layer(
                    units=self.units,
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    name="encoder_layer_{}".format(i),
                )
            )

    def call(self, inputs, padding_mask):
        embeddings = self.emb(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = self.pos_encoding(embeddings)

        outputs = self.drop(embeddings)
        attention_weights_layers = []
        for i in range(self.num_layers):
            outputs, attention_weights = self.encoder_layers[i](outputs, padding_mask)
            attention_weights_layers.append(attention_weights)
        return outputs, attention_weights_layers
