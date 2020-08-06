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
from ..layers.encoder import Encoder
from ..layers.mask import create_padding_mask


class EncoderTextClassification(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 num_layers,
                 units,
                 d_model,
                 num_heads,
                 dropout,
                 final_units=1,
                 activation=None,
                 name="encoder_text_classification",
                 pooling="average",
                 kernel_regularizer=None):
        super(EncoderTextClassification, self).__init__(name=name)
        self.pooling = pooling
        self.lambda_1 = tf.keras.layers.Lambda(create_padding_mask,
                                               output_shape=(1, 1, None),
                                               name='enc_padding_mask')
        self.encoder = Encoder(vocab_size=vocab_size,
                               num_layers=num_layers,
                               units=units,
                               d_model=d_model,
                               num_heads=num_heads,
                               dropout=dropout,
                               name="encoder",
                               kernel_regularizer=kernel_regularizer)
        self.dense = tf.keras.layers.Dense(units=final_units,
                                           activation=activation,
                                           name='outputs')

    def call(self, inputs, get_attention=False):
        enc_padding_mask = self.lambda_1(inputs)
        enc_outputs, attention_weights_layers = self.encoder(inputs, enc_padding_mask)
        if self.pooling == 'average':
            enc_pool = tf.keras.layers.GlobalAveragePooling1D()(enc_outputs)
        elif self.pooling == 'max':
            enc_pool = tf.keras.layers.GlobalMaxPooling1D()(enc_outputs)
        elif self.pooling == 'global_max':
            outputs_max = tf.keras.layers.GlobalMaxPooling1D()(enc_outputs)
            outputs_avg = tf.keras.layers.GlobalAveragePooling1D()(enc_outputs)
            enc_pool = tf.keras.Concatenate()([outputs_max, outputs_avg])
        elif self.pooling == "first":
            enc_pool = tf.squeeze(enc_outputs[:, 0:1, :])
        outputs = self.dense(enc_pool)
        if get_attention:
            return outputs, attention_weights_layers
        return outputs


class HGEncoderTextClassification(tf.keras.Model):
    def __init__(self,
                 encoder,
                 output_attentions=False,
                 num_classes=1,
                 activation=None,
                 name="encoder_text_classification",
                 pooling="average"):
        super(HGEncoderTextClassification, self).__init__(name=name)
        self.output_attentions = output_attentions
        self.pooling = pooling
        self.encoder = encoder
        self.dense = tf.keras.layers.Dense(units=num_classes,
                                           activation=activation,
                                           name='final_outputs')

    def call(self, inputs):
        attention_mask = tf.keras.layers.Masking(mask_value=0)(inputs)
        if self.output_attentions:
            enc_outputs, pooler_output, attentions = self.encoder(inputs, attention_mask=attention_mask)
        else:
            enc_outputs, pooler_output = self.encoder(inputs, attention_mask=attention_mask)
        if self.pooling == 'average':
            enc_pool = tf.keras.layers.GlobalAveragePooling1D()(enc_outputs)
        elif self.pooling == 'max':
            enc_pool = tf.keras.layers.GlobalMaxPooling1D()(enc_outputs)
        elif self.pooling == 'global_max':
            outputs_max = tf.keras.layers.GlobalMaxPooling1D()(enc_outputs)
            outputs_avg = tf.keras.layers.GlobalAveragePooling1D()(enc_outputs)
            enc_pool = tf.keras.Concatenate()([outputs_max, outputs_avg])
        elif self.pooling == "first":
            enc_pool = pooler_output
        outputs = self.dense(enc_pool)
        if self.output_attentions:
            return outputs, attentions
        return outputs


class TransformerTextClassification(tf.keras.Model):
    def __init__(self,
                 input_vocab_size,
                 num_layers,
                 units,
                 d_model,
                 num_heads,
                 num_classes,
                 dropout=0,
                 pooling='average',
                 activation='softmax',
                 name="transformer_text_classification"):
        super(TransformerTextClassification, self).__init__(name=name)
        self.pooling = pooling
        self.transformer = Transformer(input_vocab_size,
                                       input_vocab_size,
                                       num_layers,
                                       units,
                                       d_model,
                                       num_heads,
                                       dropout=0,
                                       name="transformer")
        self.dense = tf.keras.layers.Dense(units=num_classes, activation=activation, name="outputs")

    def call(self, inputs, get_attention=False):
        dec_inputs = inputs[:, :-1]
        dec_outputs, encoder_attention_weights_layers, decoder_attention_weights_layers = self.transformer(
            inputs, dec_inputs)
        if self.pooling == 'average':
            outputs = tf.keras.layers.GlobalAveragePooling1D()(dec_outputs)
        elif self.pooling == 'max':
            outputs = tf.keras.layers.GlobalMaxPooling1D()(dec_outputs)
        elif self.pooling == 'global_max':
            outputs_max = tf.keras.layers.GlobalMaxPooling1D()(dec_outputs)
            outputs_avg = tf.keras.layers.GlobalAveragePooling1D()(dec_outputs)
            outputs = tf.keras.Concatenate()([outputs_max, outputs_avg])
        elif self.pooling == "first":
            outputs = tf.squeeze(dec_outputs[:, 0:1, :])
        outputs = self.dense(outputs)
        if get_attention:
            return outputs, encoder_attention_weights_layers, decoder_attention_weights_layers
        return outputs


class CNNTextClassification(tf.keras.Model):
    def __init__(self,
                 kernel_sizes,
                 filters,
                 num_classes=1,
                 dropout_rate=0,
                 pooling='average',
                 activation=None,
                 name="transformer_text_classification",
                 pre_embedded=False,
                 vocab_size=0,
                 units=128):
        super(CNNTextClassification, self).__init__(name=name)
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.pooling = pooling
        self.activation = activation
        self.pre_embedded = pre_embedded
        self.vocab_size = vocab_size
        self.units = units
        self.pooling = pooling
        if not pre_embedded:
            self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                       output_dim=units,
                                                       embeddings_initializer='random_normal',
                                                       name='embedding')
        else:
            self.embedding = None
        self.convs = []
        for kernel_size in kernel_sizes:
            self.convs.append(tf.keras.layers.Conv1D(filters,
                                                     kernel_size=kernel_size,
                                                     activation='relu',
                                                     padding='same',
                                                     name="conv1d_{}".format(kernel_size)))
        if pooling == "average":
            self.pool = tf.keras.layers.GlobalAveragePooling1D()
        elif pooling == "max":
            self.pool = tf.keras.layers.GlobalMaxPool1D()
        else:
            raise AttributeError("Pooling type not defined")
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.concat = tf.keras.layers.Concatenate()
        self.flat = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=units//2, activation='relu', name="outputs")
        self.final_dense = tf.keras.layers.Dense(units=num_classes, activation=activation, name="outputs")

    def get_config(self):
        return {"kernel_sizes": self.kernel_sizes,
                "filters": self.filters,
                "num_classes": self.num_classes,
                "dropout": self.dropout_rate,
                "pooling": self.pooling,
                "activation": self.activation,
                "name": self.name,
                "pre_embedded": self.pre_embedded,
                "vocab_size": self.vocab_size,
                "units": self.units}

    def call(self, inputs, get_filters=False):
        if self.embedding:
            inputs = self.embedding(inputs)  # batch_size, words, emb
        conv_outputs = []
        argmax_filters = []
        for conv in self.convs:
            conv_output = conv(inputs)  # batch_size, words, filters
            argmax_filters.append(tf.argmax(conv_output, axis=1))
            conv_output = self.pool(conv_output)  # batch_size, filters
            conv_output = self.dropout(conv_output)  # batch_size, words, filters
            conv_outputs.append(conv_output)  # len(kerner_sizes), batch_size, words, filters
        concat_convs = self.concat(conv_outputs)  # batch_size, words, len(kerner_sizes) * filters
        flatten_conv = self.flat(concat_convs)
        outputs = self.dense(flatten_conv)
        outputs = self.final_dense(outputs)
        if get_filters:
            return outputs, conv_outputs, argmax_filters
        return outputs
