#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: author
@Date: 2020-07-13 11:00:51
LastEditTime: 2020-10-19 16:16:52
LastEditors: Please set LastEditors
@Description: Define configuration parameters.
@FilePath: /project_2/model/config.py
'''

from typing import Optional

import torch

# General
hidden_size: int = 512
dec_hidden_size: Optional[int] = 512
embed_size: int = 512

# Data
max_vocab_size = 20000
# embed_file: Optional[str] = '../files/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5'  # use pre-trained embeddings
embed_file: Optional[str] = None  # use pre-trained embeddings
source = 'train'    # use value: train or  big_samples
data_path: str = '../files/{}.txt'.format(source)
val_data_path: Optional[str] = '../files/dev.txt'
test_data_path: Optional[str] = '../files/test.txt'
stop_word_file = '../files/HIT_stop_words.txt'
max_src_len: int = 300  # exclusive of special tokens such as EOS
max_tgt_len: int = 100  # exclusive of special tokens such as EOS
truncate_src: bool = True
truncate_tgt: bool = True
min_dec_steps: int = 30
max_dec_steps: int = 100
enc_rnn_dropout: float = 0.5
enc_attn: bool = True
dec_attn: bool = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0


# Training
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 8
batch_size = 32
max_grad_norm = 2.0
is_cuda = True
DEVICE = torch.device("cuda" if is_cuda else "cpu")
LAMBDA = 1

# 通过以下配置可以训练以下模型：
# 1. PGN，令 pointer = True（默认 source = 'train'） 即可。
# 2. PGN (with coverage)，令 pointer = True 以及 coverage = True
# 3. PGN (fine-tuned with coverage)，令 pointer = True ， coverage = True 以及 fine_tune = True
# 4. PGN (with Weight tying)，令 pointer = True 以及 weight_tying = True
# 5. PGN (with Scheduled sampling)，令 pointer = True ， scheduled_sampling= True
# 6. PGN (training with big_samples.txt)，令 pointer = True 以及 source = 'big_samples'
pointer = True
coverage = False
fine_tune = False
scheduled_sampling = False
weight_tying = False

if pointer:
    if coverage:
        if fine_tune:
            model_name = 'ft_pgn'
        else:
            model_name = 'cov_pgn'
    elif scheduled_sampling:
        model_name = 'ss_pgn'
    elif weight_tying:
        model_name = 'wt_pgn'
    else:
        if source == 'big_samples':
            model_name = 'pgn_big_samples'
        else:
            model_name = 'pgn'
else:
    model_name = 'baseline'

encoder_save_name = '../saved_model/' + model_name + '/encoder.pt'
decoder_save_name = '../saved_model/' + model_name + '/decoder.pt'
attention_save_name = '../saved_model/' + model_name + '/attention.pt'
reduce_state_save_name = '../saved_model/' + model_name + '/reduce_state.pt'
losses_path = '../saved_model/' + model_name + '/val_losses.pkl'
log_path = '../runs/' + model_name


# Beam search
beam_size: int = 3
alpha = 0.2
beta = 0.2
gamma = 0.6
