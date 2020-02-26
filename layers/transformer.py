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
from .encoder import Encoder
from .decoder import Decoder
from .mask import create_padding_mask, create_look_ahead_mask


class Transformer(tf.keras.layers.Layer):
    def __init__(self,
                 input_vocab_size,
                 output_vocab_size,
                 num_layers,
                 units,
                 d_model,
                 num_heads,
                 dropout=0,
                 name="transformer"):
        super(Transformer, self).__init__(name=name)
        self.lambda_1 = tf.keras.layers.Lambda(create_padding_mask,
                                               output_shape=(1, 1, None),
                                               name='enc_padding_mask')
        self.encoder = Encoder(vocab_size=input_vocab_size,
                               num_layers=num_layers,
                               units=units,
                               d_model=d_model,
                               num_heads=num_heads,
                               dropout=dropout,
                               name="encoder")

        self.decoder = Decoder(vocab_size=output_vocab_size,
                               num_layers=num_layers,
                               units=units,
                               d_model=d_model,
                               num_heads=num_heads,
                               dropout=dropout,
                               name="decoder"
                               )
        self.lambda_2 = tf.keras.layers.Lambda(create_look_ahead_mask,
                                               output_shape=(1, None, None),
                                               name='look_ahead_mask')

        self.lambda_3 = tf.keras.layers.Lambda(create_padding_mask,
                                               output_shape=(1, 1, None),
                                               name='dec_padding_mask')

    def call(self, inputs, dec_inputs):
        enc_padding_mask = self.lambda_1(inputs)
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = self.lambda_2(dec_inputs)
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = self.lambda_3(inputs)

        enc_outputs, encoder_attention_weights_layers = self.encoder(inputs, enc_padding_mask)

        dec_outputs, decoder_attention_weights_layers = self.decoder(dec_inputs,
                                                                     enc_outputs,
                                                                     look_ahead_mask,
                                                                     dec_padding_mask)
        return dec_outputs, encoder_attention_weights_layers, decoder_attention_weights_layers
