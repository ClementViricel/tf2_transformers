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
from pathlib import Path
from os.path import join
import tensorflow as tf
import tensorflow_hub as hub
from models.NMT import TransformerNMT
from layers.optimize import CustomSchedule, loss_function

from transformers import AutoTokenizer

lang_1 = 'en'
lang_2 = 'fr'
parent_dir = join('data', "{}-{}".format(lang_1, lang_2))
file_names = ["Europarl", "EUbookshop"]
LIMIT = int(10E6)
SAVE_NAME = "test-{}-{}".format(lang_1, lang_2)
MAX_LENGTH = 40

EPOCHS = 10
BUFFER_SIZE = 640
BATCH_SIZE = 64

NUM_LAYERS = 1
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1

FROM_LAST_CHECKPOINT = ""

print("Building datasets...")
data_sets_1 = []
data_sets_2 = []
for file in file_names:
    if not Path(Path(parent_dir) / f'{file}.{lang_1}').exists():
        print("Downloading data...")
        Path(parent_dir).mkdir(parents=True, exist_ok=True)
        # Should pdo that though opus python pkg...
        os.system(f"opus_read -d {file} -s {lang_1} -t {lang_2} -S 1 -T 1 -w {parent_dir}/{file}.{lang_1} {parent_dir}/{file}.{lang_2} -wm moses -m {LIMIT} -v")

    data_1 = tf.data.TextLineDataset(join(parent_dir, f'{file}.{lang_1}'))
    data_sets_1.append(data_1)
    data_2 = tf.data.TextLineDataset(join(parent_dir, f'{file}.{lang_2}'))
    data_sets_2.append(data_2)

all_data_sets_1 = data_sets_1[0]
all_data_sets_2 = data_sets_2[0]
for d_1, d_2 in zip(data_sets_1[1:], data_sets_2[1:]):
    all_data_sets_1 = all_data_sets_1.concatenate(d_1)
    all_data_sets_2 = all_data_sets_2.concatenate(d_2)

print("End.")

def split_target(x, y):
    return (x, y), y

print("Preprocessing datasets...")
train_dataset = tf.data.Dataset.zip((all_data_sets_1, all_data_sets_2))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.map(split_target)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
print("End.")

model = TransformerNMT(
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
)

model_save_path = join("checkpoints", SAVE_NAME.format(lang_1, lang_2))
Path(model_save_path).mkdir(parents=True, exist_ok=True)

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

if FROM_LAST_CHECKPOINT:
    model.load_weights(tf.train.latest_checkpoint(FROM_LAST_CHECKPOINT))

model.compile(optimizer=optimizer, loss=loss_function, metrics=['sparse_categorical_accuracy'])
model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=model_callbacks,
    use_multiprocessing=True,
    workers=32
)

ckpt.save(join(model_save_path, "transformer_nmt_whole"))
