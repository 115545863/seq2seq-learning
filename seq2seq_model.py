import tensorflow as tf
import numpy as np
import time

# 检查 TensorFlow 版本
assert tf.__version__.startswith('2'), '请使用 TensorFlow 2.x 版本'
print('TensorFlow Version:', tf.__version__)

# Encoder 层
def get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size, encoding_embedding_size):
    '''
    构造 Encoder 层
    '''
    # 1. Embedding 层
    encoder_embed = tf.keras.layers.Embedding(input_dim=source_vocab_size, output_dim=encoding_embedding_size)(input_data)

    # 2. 构造 LSTM Cell
    lstm_cells = [tf.keras.layers.LSTMCell(rnn_size) for _ in range(num_layers)]
    encoder_rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(lstm_cells), return_sequences=True, return_state=True)

    # 3. 获取 Encoder 输出
    encoder_output, *encoder_state = encoder_rnn(encoder_embed, mask=tf.sequence_mask(source_sequence_length, dtype=tf.float32))

    return encoder_output, encoder_state

# 预处理 Decoder 输入
def process_decoder_input(data, vocab_to_int, batch_size):
    '''
    在序列前添加 <GO>，去掉最后一个字符
    '''
    ending = data[:, :-1]
    go_tokens = tf.fill([batch_size, 1], vocab_to_int['<GO>'])
    decoder_input = tf.concat([go_tokens, ending], axis=1)

    return decoder_input

# Decoder 层
def decoding_layer(target_vocab_size, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input, batch_size):
    '''
    构造 Decoder 层
    '''
    # 1. Embedding 层
    decoder_embedding = tf.keras.layers.Embedding(input_dim=target_vocab_size, output_dim=decoding_embedding_size)
    decoder_embed_input = decoder_embedding(decoder_input)

    # 2. LSTM Cell
    lstm_cells = [tf.keras.layers.LSTMCell(rnn_size) for _ in range(num_layers)]
    decoder_rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(lstm_cells), return_sequences=True, return_state=True)

    # 3. 输出层
    output_layer = tf.keras.layers.Dense(target_vocab_size, activation='softmax')

    # 4. 训练时使用 Training Helper
    decoder_output, _ = decoder_rnn(decoder_embed_input, initial_state=encoder_state)
    training_decoder_output = output_layer(decoder_output)

    return training_decoder_output

# Seq2Seq 模型
def seq2seq_model(input_data, targets, target_vocab_size, source_vocab_size,
                  encoding_embedding_size, decoding_embedding_size,
                  rnn_size, num_layers, batch_size, target_sequence_length, max_target_sequence_length, vocab_to_int):

    # 1. 获取 Encoder 状态
    encoder_output, encoder_state = get_encoder_layer(input_data, rnn_size, num_layers, target_sequence_length, source_vocab_size, encoding_embedding_size)

    # 2. 处理 Decoder 输入
    decoder_input = process_decoder_input(targets, vocab_to_int, batch_size)

    # 3. 获取 Decoder 输出
    training_decoder_output = decoding_layer(target_vocab_size, decoding_embedding_size, num_layers, rnn_size,
                                             target_sequence_length, max_target_sequence_length, encoder_state, decoder_input, batch_size)

    return training_decoder_output
