#!/usr/bin/env python
#coding=utf-8

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from data_process import data_process


# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15

data_process = data_process()

def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = 7
    return [data_process.source_letter_to_int.get(word, data_process.source_letter_to_int['<UNK>']) for word in text] \
           + [data_process.source_letter_to_int['<PAD>']]*(sequence_length-len(text))


# 输入一个单词
input_word = 'result'
text = source_to_seq(input_word)

checkpoint = "./trained_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(input_word)] * batch_size,
                                      source_sequence_length: [len(input_word)] * batch_size})[0]

pad = data_process.source_letter_to_int["<PAD>"]

print('原始输入:', input_word)

print('\nSource')
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([data_process.source_int_to_letter[i] for i in text])))

print('\nTarget')
print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([data_process.target_int_to_letter[i] for i in answer_logits if i != pad])))