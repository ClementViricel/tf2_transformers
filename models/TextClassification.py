import tensorflow as tf
from layers.transformer import Transformer
from layers.encoder import Encoder


class EncoderTextClassification(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 num_layers,
                 units,
                 d_model,
                 num_heads,
                 dropout,
                 get_attention=False,
                 final_units=1,
                 name="encoder_text_classification"):
        super(EncoderTextClassification, self).__init__(name=name)
        self.lambda_1 = tf.keras.layers.Lambda(create_padding_mask,
                                               output_shape=(1, 1, None),
                                               name='enc_padding_mask')
        self.encoder = Encoder(vocab_size=vocab_size,
                               num_layers=num_layers,
                               units=units,
                               d_model=d_model,
                               num_heads=num_heads,
                               dropout=dropout,
                               name="encoder")
        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(units=final_units, name='outputs')

    def call(self, inputs, get_attention=False):
        enc_padding_mask = self.lambda_1(inputs)
        enc_outputs, attention_weights_layers = self.encoder(inputs, enc_padding_mask)
        enc_pool = self.avg_pool(enc_outputs)
        outputs = self.dense(enc_pool)
        if get_attention:
            return outputs, attention_weights_layers
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