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
import tensorflow_text
import tensorflow_hub as hub
from layers.transformer import Transformer


class TransformerNMT(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 units,
                 d_model,
                 num_heads,
                 dropout=0,
                 use_tokenizer=True,
                 input_vocab_size=0,
                 output_vocab_size=0,
                 name="transformer_nmt"):
        super(TransformerNMT, self).__init__(name=name)
        self.use_tokenizer = use_tokenizer
        if use_tokenizer:
            preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
            self.tokenizer = hub.KerasLayer(preprocessor)
            vocab_size = preprocessor.tokenize.get_special_tokens_dict()['vocab_size']
            output_vocab_size = vocab_size
            self.start_id = preprocessor.tokenize.get_special_tokens_dict()['start_of_sequence_id']
            self.stop_id = preprocessor.tokenize.get_special_tokens_dict()['end_of_segment_id']
            self.mask_id = preprocessor.tokenize.get_special_tokens_dict()['mask_id']
            self.transformer = Transformer(vocab_size,
                                           vocab_size,
                                           num_layers,
                                           units,
                                           d_model,
                                           num_heads,
                                           dropout=dropout,
                                           name="transformer")
        else:
            self.transformer = Transformer(input_vocab_size,
                                        output_vocab_size,
                                        num_layers,
                                        units,
                                        d_model,
                                        num_heads,
                                        dropout=dropout,
                                        name="transformer")
        self.dense = tf.keras.layers.Dense(units=output_vocab_size, name="outputs")
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        if self.use_tokenizer:
            y = self.tokenizer(y)['input_word_ids']
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, inputs, get_attention=False):
        enc_inputs = inputs[0]
        dec_inputs = inputs[1]
        if self.use_tokenizer:
            enc_inputs = self.tokenizer(enc_inputs)['input_word_ids']
            dec_inputs = self.tokenizer(dec_inputs)['input_word_ids']
        dec_outputs, encoder_attention_weights_layers, decoder_attention_weights_layers = self.transformer(
            enc_inputs, dec_inputs)
        outputs = self.dense(dec_outputs)
        if get_attention:
            return outputs, encoder_attention_weights_layers, decoder_attention_weights_layers
        return outputs
