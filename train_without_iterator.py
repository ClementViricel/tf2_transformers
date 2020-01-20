import time
import os
import pickle
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
inputs = []
outputs = []
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

    with open(join(parent_dir, '{}.{}'.format(file, lang_1)), 'r', encoding='utf-8') as in_file:
        inputs.extend(in_file.read().splitlines())
    with open(join(parent_dir, '{}.{}'.format(file, lang_2)), 'r', encoding='utf-8') as in_file:
        outputs.extend(in_file.read().splitlines())
print("Load {} training sentence pairs.".format(len(inputs)))

print("Building Tokenizers...")
if not os.paath.exists("tokenizers/tokenizer_{}".format(lang_1)):
    os.makedirs("tokenizers")
    tokenizer_1 = tfds.features.text.SubwordTextEncoder.build_from_corpus(inputs, target_vocab_size=2**16)
    tokenizer_1.save_to_file("tokenizers/tokenizer_{}".format(lang_1))

    tokenizer_2 = tfds.features.text.SubwordTextEncoder.build_from_corpus(outputs, target_vocab_size=2**16)
    tokenizer_2.save_to_file("tokenizers/tokenizer_{}".format(lang_2))
else:
    tokenizer_1 = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizers/tokenizer_{}".format(lang_1))
    tokenizer_2 = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizers/tokenizer_{}".format(lang_2))
print("End.")

if not os.path.exists("dataset.pickle"):
    print("Filtering datasets")
    inputs, outputs = tokenize_and_filter(
        inputs,
        outputs,
        tokenizer_1,
        tokenizer_2,
        MAX_LENGTH
    )
    dataset = (
        {
            'inputs': inputs,
            'dec_inputs': outputs[:, :-1]
        },
        {
            'outputs': outputs[:, 1:]
        },
    )
    assert(len(inputs) == len(outputs))
    with open('dataset.pickle', 'wb') as file_o:
        pickle.dump(dataset, file_o)
else:
    with open('dataset.pickle', 'rb') as file_o:
        dataset = pickle.load(file_o)
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

model.compile(optimizer=optimizer, loss=loss_function, metrics=['sparse_categorical_accuracy'])
model.fit(
    x=dataset[0],
    y=dataset[1],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    shuffle=True,
    callbacks=model_callbacks,
    use_multiprocessing=True,
    workers=32
)

ckpt.save(join(model_save_path, "transformer_nmt_whole"))
