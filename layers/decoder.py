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


class Decoder_layer(tf.keras.layers.Layer):
    def __init__(self, units, d_model, num_heads, dropout, name="decoder_layer"):
        super(Decoder_layer, self).__init__(name=name)
        self.dropout = dropout
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads

        self.mha_1 = MultiHeadAttention(d_model, num_heads, name="self_attention")
        self.mha_2 = MultiHeadAttention(d_model, num_heads, name="input_output_attention")

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.drop_1 = tf.keras.layers.Dropout(rate=dropout)
        self.drop_2 = tf.keras.layers.Dropout(rate=dropout)
        self.drop_3 = tf.keras.layers.Dropout(rate=dropout)

        self.dense_relu = tf.keras.layers.Dense(units=units, activation='relu')
        self.dense = tf.keras.layers.Dense(units=d_model)

    def call(self, dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attention1, attention_weights_1 = self.mha_1(dec_inputs, dec_inputs, dec_inputs, look_ahead_mask)
        attention1 = self.drop_1(attention1)
        attention1 = self.norm_1(attention1 + dec_inputs)

        attention2, attention_weights_2 = self.mha_2(attention1, enc_outputs, enc_outputs, dec_padding_mask)
        attention2 = self.drop_2(attention2)
        attention2 = self.norm_2(attention2 + attention1)

        outputs = self.dense_relu(attention2)
        outputs = self.dense(outputs)
        outputs = self.drop_3(outputs)
        outputs = self.norm_3(outputs + attention2)

        return outputs, attention_weights_1, attention_weights_2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout, name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.emb = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(vocab_size, d_model)
        self.drop = tf.keras.layers.Dropout(rate=dropout)
        self.decoder_layers = []
        for i in range(num_layers):
            self.decoder_layers.append(
                Decoder_layer(
                    units=self.units,
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    name="decoder_layer_{}".format(i),
                )
            )

    def call(self, inputs, enc_outputs, look_ahead_mask, padding_mask):
        embeddings = self.emb(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = self.pos_encoding(embeddings)

        outputs = self.drop(embeddings)
        decoder_attention_weights_layers = []
        for i in range(self.num_layers):
            outputs, attention_weights_1, attention_weights_2 = self.decoder_layers[i](
                outputs, enc_outputs, look_ahead_mask, padding_mask)
            decoder_attention_weights_layers.append({"block_1": attention_weights_1, "block_2": attention_weights_2})
        return outputs, decoder_attention_weights_layers
