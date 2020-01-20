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
from layers.preprocessing import tokenize_and_filter, parse_json_line_skills

lang_1 = 'en'
lang_2 = 'fr'
parent_dir = join('data', "{}-{}".format(lang_1, lang_2))
file_names = ["Europarl", "jobs", "skills", "EUbookshop"]
EVAL_FOLDER = '/home/cviricel/365talents-evaluation'

print("Building datasets...")
inputs = []
outputs = []
for file in file_names:
    with open(join(parent_dir, '{}.{}'.format(file, lang_1)), 'r', encoding='utf-8') as in_file:
        inputs.extend(in_file.read().splitlines())
    with open(join(parent_dir, '{}.{}'.format(file, lang_2)), 'r', encoding='utf-8') as in_file:
        outputs.extend(in_file.read().splitlines())
eval_inputs, eval_outputs = parse_json_line_skills(EVAL_FOLDER, lang_1, lang_2)
eval_inputs.extend(inputs[:100])
inputs = inputs[100:]
eval_outputs.extend(outputs[:100])
outputs = outputs[100:]
print("Load {} training sentence pairs.".format(len(inputs)))
print("Load {} evaluation sentence pairs.".format(len(eval_inputs)))

print("Building Tokenizers...")
# tokenizer_1 = tfds.features.text.SubwordTextEncoder.build_from_corpus(inputs, target_vocab_size=2**16)
# tokenizer_1.save_to_file("tokenizers/tokenizer_{}".format(lang_1))

# tokenizer_2 = tfds.features.text.SubwordTextEncoder.build_from_corpus(outputs, target_vocab_size=2**16)
# tokenizer_2.save_to_file("tokenizers/tokenizer_{}".format(lang_2))

tokenizer_1 = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizers/tokenizer_{}".format(lang_1))
tokenizer_2 = tfds.features.text.SubwordTextEncoder.load_from_file("tokenizers/tokenizer_{}".format(lang_2))
print("End.")

if not os.path.exists("dataset.pickle"):
    print("Filtering datasets")
    MAX_LENGTH = 40
    inputs, outputs = tokenize_and_filter(
        inputs,
        outputs,
        tokenizer_1,
        tokenizer_2,
        MAX_LENGTH
    )
    eval_inputs, eval_outputs = tokenize_and_filter(
        eval_inputs,
        eval_outputs,
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

    eval_dataset = (
        {
            'inputs': eval_inputs,
            'dec_inputs': eval_outputs[:, :-1]
        },
        {
            'outputs': eval_outputs[:, 1:]
        },
    )
    assert(len(inputs) == len(outputs))
    with open('dataset.pickle', 'wb') as file_o:
        pickle.dump(dataset, file_o)
    with open('eval_dataset.pickle', 'wb') as file_o:
        pickle.dump(eval_dataset, file_o)
else:
    with open('dataset.pickle', 'rb') as file_o:
        dataset = pickle.load(file_o)
    with open('eval_dataset.pickle', 'rb') as file_o:
        eval_dataset = pickle.load(file_o)

BUFFER_SIZE = 20000
BATCH_SIZE = 256
# print("Creating TF dataset...")
# dataset = tf.data.Dataset.from_tensor_slices((
#     {
#         'inputs': inputs,
#         'dec_inputs': outputs[:, :-1]
#     },
#     {
#         'outputs': outputs[:, 1:]
#     },
# ))
# dataset = dataset.cache()
# dataset = dataset.shuffle(BUFFER_SIZE)
# dataset = dataset.batch(BATCH_SIZE)
# dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
print("End.")

NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
EPOCHS = 2

model = transformer(
    input_vocab_size=tokenizer_1.vocab_size + 2,
    output_vocab_size=tokenizer_2.vocab_size + 2,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
)

model_save_path = join("checkpoints", "train-2-{}-{}".format(lang_1, lang_2))
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=join(model_save_path,
                                                     "epoch-{epoch:02d}-{sparse_categorical_accuracy:.2f}"),
                                       monitor='val_sparse_categorical_accuracy',
                                       mode='max',
                                       save_best_only=True,
                                       save_weights_only=True,
                                       verbose=0)
]

learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model.compile(optimizer=optimizer, loss=loss_function, metrics=['sparse_categorical_accuracy'])
model.fit(
    x=dataset[0],
    y=dataset[1],
    validation_data=eval_dataset,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=True,
    callbacks=model_callbacks,
    use_multiprocessing=True,
    workers=32
)
