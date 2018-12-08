#!/usr/bin/env python
#coding=utf-8

import numpy as np


with open('data/letters_source.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()

with open('data/letters_target.txt', 'r', encoding='utf-8') as f:
    target_data = f.read()

# 数据预览
# print(source_data.split('\n')[:3])  # ['bsaqq', 'npy', 'lbwuj']
# print(target_data.split('\n')[:3])  # ['abqqs', 'npy', 'bjluw']

def extract_character_vocab(data):
    '''
    构造映射表
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    set_words = list(set([character for line in data.split('\n') for character in line]))
    # 取前3个为例,set_words:: ['s', 'q', 'l', 'n', 'w', 'j', 'y', 'p', 'b', 'a', 'u']
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    #  int_to_vocab:: {0: '<PAD>', 1: '<UNK>', 2: '<GO>', 3: '<EOS>', 4: 'b', 5: 'n', 6: 'q', 7: 'j', 8: 'a',
    # 9: 'w', 10: 'y', 11: 's', 12: 'p', 13: 'l', 14: 'u'}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int

class data_process:

    def __init__(self):
        self.source_letter_to_int = {}
        self.target_letter_to_int = {}
        self.source_int = []
        self.target_int = []
        self.source_int_to_letter = {}
        self.target_int_to_letter = {}


        # 构造映射表
        self.source_int_to_letter, self.source_letter_to_int = extract_character_vocab(source_data)
        self.target_int_to_letter, self.target_letter_to_int = extract_character_vocab(target_data)

        # 对字母进行转换
        self.source_int = [[self.source_letter_to_int.get(letter, self.source_letter_to_int['<UNK>'])
                       for letter in line] for line in source_data.split('\n')]
        # source_int = [[12, 18, 9, 5, 5], [17, 24, 15], [20, 12, 23, 13, 21]]
        self.target_int = [[self.target_letter_to_int.get(letter, self.target_letter_to_int['<UNK>'])
                       for letter in line] + [self.target_letter_to_int['<EOS>']] for line in target_data.split('\n')]


def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    每个batch内有相同的长度，但是不同的batch间长度可以不同
    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

