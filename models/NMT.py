import tensorflow as tf
from layers.transformer import Transformer


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

    def call(self, inputs, dec_inputs):
        dec_outputs, encoder_attention_weights_layers, decoder_attention_weights_layers = self.transformer(
            inputs, dec_inputs)
        outputs = self.dense(dec_outputs)
        if get_attention:
            return outputs, encoder_attention_weights_layers, decoder_attention_weights_layers
        return outputs
