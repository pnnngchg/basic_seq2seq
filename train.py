#!/usr/bin/env python
#coding=utf-8
import tensorflow as tf
from model import seq2seq_model
from model import get_inputs
from data_process import get_batches
from data_process import data_process

# 超参数
# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001

data_process = data_process()

# 构造graph
train_graph = tf.Graph()

with train_graph.as_default():
    # 获得模型输入
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       lr,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len(data_process.source_letter_to_int),
                                                                       len(data_process.target_letter_to_int),
                                                                       encoding_embedding_size,  # 15
                                                                       decoding_embedding_size,  # 15
                                                                       rnn_size,  # 50
                                                                       num_layers)  # 2
    # tf.identity 在计算图中建立建立复制计算节点
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function 对序列logits计算加权交叉熵
        # training_logits是输出层的结果，targets是目标值，
        # masks是我们使用tf.sequence_mask计算的结果，在这里作为权重，也就是说我们在计算交叉熵时不会把<PAD>计算进去
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,   # 训练输出
            targets,           # 正确值
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

# 将数据集分割为train和validation
train_source = data_process.source_int[batch_size:]
train_target = data_process.target_int[batch_size:]
# 留出一个batch进行验证
valid_source = data_process.source_int[:batch_size]
valid_target = data_process.target_int[:batch_size]
(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
    get_batches(valid_target, valid_source, batch_size,
                data_process.source_letter_to_int['<PAD>'],
                data_process.target_letter_to_int['<PAD>']))

display_step = 50  # 每隔50轮输出loss

checkpoint = "trained_model.ckpt"
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(1, epochs + 1):
        for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                get_batches(train_target, train_source, batch_size,
                            data_process.source_letter_to_int['<PAD>'],
                            data_process.target_letter_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: sources_batch,
                 targets: targets_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths})

            if batch_i % display_step == 0:
                # 计算validation loss
                validation_loss = sess.run(
                    [cost],
                    {input_data: valid_sources_batch,
                     targets: valid_targets_batch,
                     lr: learning_rate,
                     target_sequence_length: valid_targets_lengths,
                     source_sequence_length: valid_sources_lengths})

                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(train_source) // batch_size,
                              loss,
                              validation_loss[0]))

    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Trained and Saved')