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

import os
import pickle
import numpy as np
from pathlib import Path
from os.path import join
import tensorflow as tf
from models.NMT import TransformerNMT
from layers.optimize import CustomSchedule, loss_function
from layers.preprocessing import tokenize_and_filter

from transformers import AutoTokenizer

lang_1 = 'en'
lang_2 = 'fr'
parent_dir = join('data', "{}-{}".format(lang_1, lang_2))
file_names = ["Europarl", "EUbookshop"]
LIMIT = 10E6
SAVE_NAME = "test-{}-{}".format(lang_1, lang_2)
MAX_LENGTH = 40

EPOCHS = 10
BATCH_SIZE = 32

NUM_LAYERS = 1
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

FROM_LAST_CHECKPOINT = ""
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
print("Building datasets...")
if not Path("dataset.pickle").exists():
    inputs = []
    outputs = []
    for file in file_names:
        if not Path(Path(parent_dir) / f'{file}.{lang_1}').exists():
            print("Downloading data...")
            Path('data').mkdir(exist_ok=True)
            # Should pdo that though opus python pkg...
            os.system("opus_read -d {} -s {} -t {} -S 1 -T 1 -w {}/{}.{} data/{}.{} -wm moses -m {} -v".format(file,
                                                                                                            lang_1,
                                                                                                            lang_2,
                                                                                                            parent_dir,
                                                                                                            file,
                                                                                                            lang_1,
                                                                                                            parent_dir,
                                                                                                            file,
                                                                                                            lang_2,
                                                                                                            LIMIT))

        with open(join(parent_dir, f'{file}.{lang_1}'), 'r', encoding='utf-8') as in_file:
            inputs.extend(in_file.read().splitlines())
        with open(join(parent_dir, f'{file}.{lang_2}'), 'r', encoding='utf-8') as in_file:
            outputs.extend(in_file.read().splitlines())
    print("Filtering datasets")
    inputs, outputs = tokenize_and_filter(
        inputs,
        outputs,
        tokenizer,
        tokenizer,
        MAX_LENGTH,
        BATCH_SIZE
    )
    dataset = {'inputs': inputs, 'outputs': outputs}
    assert(len(inputs) == len(outputs))
    with open('dataset.pickle', 'wb') as file_o:
        pickle.dump(dataset, file_o)
else:
    with open('dataset.pickle', 'rb') as file_o:
        dataset = pickle.load(file_o)
        inputs = dataset['inputs']
        outputs = dataset['outputs']
print("End.")
print("Building model...")
model = TransformerNMT(
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT,
    use_tokenizer=False,
    input_vocab_size=tokenizer.vocab_size,
    output_vocab_size=tokenizer.vocab_size
)
print("End")
model_save_path = join("checkpoints", SAVE_NAME.format(lang_1, lang_2))
Path(model_save_path).mkdir(exist_ok=True)

model_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=join(model_save_path,
                                                     "epoch-{epoch:02d}-{sparse_categorical_accuracy:.2f}"),
                                       monitor='sparse_categorical_accuracy',
                                       save_weights_only=True,
                                       save_freq='epoch',
                                       verbose=0)
]

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

ckpt = tf.train.Checkpoint(transformer=model,
                           optimizer=optimizer)

model.compile(optimizer=optimizer, loss=loss_function, metrics=['sparse_categorical_accuracy'])
print("Beggining training...")
model.fit(
    x=(np.array(inputs), np.array(outputs)),
    y=np.array(outputs),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    shuffle=True,
    callbacks=model_callbacks,
)
print("End")
ckpt.save(join(model_save_path, "transformer_nmt_whole"))
