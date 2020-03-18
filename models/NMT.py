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
from ..layers.transformer import Transformer


class TransformerNMT(tf.keras.Model):
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 num_layers,
                 units,
                 d_model,
                 num_heads,
                 dropout=0,
                 name="transformer_nmt"):
        super(TransformerNMT, self).__init__(name=name)
        self.transformer = Transformer(input_vocab_size,
                                       output_vocab_size,
                                       num_layers,
                                       units,
                                       d_model,
                                       num_heads,
                                       dropout=0,
                                       name="transformer")
        self.dense = tf.keras.layers.Dense(units=output_vocab_size, name="outputs")

    def call(self, inputs, get_attention=False):
        enc_inputs = inputs[0]
        dec_inputs = inputs[1]
        dec_outputs, encoder_attention_weights_layers, decoder_attention_weights_layers = self.transformer(
            enc_inputs, dec_inputs)
        outputs = self.dense(dec_outputs)
        if get_attention:
            return outputs, encoder_attention_weights_layers, decoder_attention_weights_layers
        return outputs
