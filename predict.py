import tensorflow as tf


def evaluate(model, sentence, tokenizer_1, tokenizer_2, MAX_LENGTH=40):
    start_token = [tokenizer_1.vocab_size]
    end_token = [tokenizer_1.vocab_size + 1]
    sentence = tf.expand_dims(
        start_token + tokenizer_1.encode(sentence) + end_token, axis=0)

    output = tf.expand_dims([tokenizer_2.vocab_size], 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, [tokenizer_2.vocab_size + 1]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(model, sentence, tokenizer_1, tokenizer_2):
    prediction = evaluate(model, sentence, tokenizer_1, tokenizer_2)

    predicted_sentence = tokenizer_2.decode(
        [i for i in prediction if i < tokenizer_2.vocab_size])
    return predicted_sentence
