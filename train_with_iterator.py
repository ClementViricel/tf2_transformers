import time
import os
from os.path import join
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from math import floor
from layers.transformer import transformer
from layers.optimize import CustomSchedule, loss_function
from layers.preprocessing import tokenize_and_filter

lang_1 = 'en'
lang_2 = 'fr'
parent_dir = join('data', "{}-{}".format(lang_1, lang_2))
file_names = ["Europarl", "EUbookshop"]
LIMIT = 10E6
SAVE_NAME = "test-{}-{}".format(lang_1, lang_2)
MAX_LENGTH = 40

EPOCHS = 10
BUFFER_SIZE = 20000
BATCH_SIZE = 256

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
    if not os.path.exist(join(parent_dir, '{}.{}'.format(file, lang_1))):
        print("Downloading data...")
        os.makedirs('data')
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

    data_1 = tf.data.TextLineDataset(join(parent_dir, '{}.{}'.format(file, lang_1)))
    data_sets_1.append(data_1)
    data_2 = tf.data.TextLineDataset(join(parent_dir, '{}.{}'.format(file, lang_2)))
    data_sets_2.append(data_2)

all_data_sets_1 = data_sets_1[0]
all_data_sets_2 = data_sets_2[0]
for d_1, d_2 in zip(data_sets_1[1:], data_sets_2[1:]):
    all_data_sets_1 = all_data_sets_1.concatenate(d_1)
    all_data_sets_2 = all_data_sets_2.concatenate(d_2)

print("End.")
print("Building Tokenizers...")
if not os.paath.exists("tokenizers/tokenizer_{}".format(lang_1)):
    tokenizer_1 = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (i.numpy() for i in all_data_sets_1), target_vocab_size=2**16)
    tokenizer_1.save_to_file("tokenizers/tokenizer_{}".format(lang_1))

    tokenizer_2 = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (i.numpy() for i in all_data_sets_2), target_vocab_size=2**16)
    tokenizer_2.save_to_file("tokenizers/tokenizer_{}".format(lang_2))
else:
    tokenizer_1 = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizers/tokenizer_{}".format(lang_1))
    tokenizer_2 = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizers/tokenizer_{}".format(lang_2))
print("End.")


def spend_time(start):
    delta_secs = int(time.time() - start)
    if delta_secs < 60:
        return "{}s".format(delta_secs)
    elif delta_secs < 60 * 60:
        delta_mins = floor(delta_secs / 60)
        delta_secs = delta_secs % 60
        return "{}m{}s".format(delta_mins, delta_secs)
    elif delta_secs < 60 * 60 * 24:
        delta_hours = floor(delta_secs / (60 * 60))
        delta_secs = delta_secs % (60 * 60)
        delta_mins = floor(delta_secs / 60)
        delta_secs = delta_secs % 60
        return "{}h{}m{}s".format(delta_hours, delta_mins, delta_secs)


def encode(lang1, lang2):
    lang1 = [tokenizer_1.vocab_size] + tokenizer_1.encode(
        lang1.numpy()) + [tokenizer_1.vocab_size+1]

    lang2 = [tokenizer_2.vocab_size] + tokenizer_2.encode(
        lang2.numpy()) + [tokenizer_2.vocab_size+1]

    return lang1, lang2


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


def tf_encode(lang_1, lang_2):
    return tf.py_function(encode, [lang_1, lang_2], Tout=[tf.int64, tf.int64])


def split_target(x, y):
    return (x, y[:, :-1]), y[:, 1:]


print("Preprocessing datasets...")
train_dataset = tf.data.Dataset.zip((all_data_sets_1, all_data_sets_2))
train_dataset = train_dataset.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]), drop_remainder=True)
train_dataset = train_dataset.map(split_target)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
print("End.")

model = transformer(
    input_vocab_size=tokenizer_1.vocab_size + 2,
    output_vocab_size=tokenizer_2.vocab_size + 2,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
)

model_save_path = join("checkpoints", SAVE_NAME.format(lang_1, lang_2))
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

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
