#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from data.cnews_loader import attention

class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 200      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表达小

    num_layers= 3           # 隐藏层层数
    hidden_dim = 512        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    attention_dim = 512
    l2_reg_lambda = 0.01

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 6          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell_forward = lstm_cell()
                # cell_backward = lstm_cell()
            else:
                cell_forward = gru_cell()
                # cell_backward = gru_cell()
            cell_forward = tf.contrib.rnn.DropoutWrapper(cell_forward, output_keep_prob=self.keep_prob)
            # cell_backward = tf.contrib.rnn.DropoutWrapper(cell_backward, output_keep_prob=self.keep_prob)
            return cell_forward# , cell_backward
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            # cells = [dropout() for _ in range(self.config.num_layers)]
            # rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            #
            # _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            # 不能一起写，要分开写，不能这样
            cell_forward = [dropout() for _ in range(self.config.num_layers)]
            cell_backward = [dropout() for _ in range(self.config.num_layers)]
            lstm_fw = tf.nn.rnn_cell.MultiRNNCell(cell_forward)
            lstm_bw = tf.nn.rnn_cell.MultiRNNCell(cell_backward)
            _outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, embedding_inputs, dtype=tf.float32)
            # _outputs, bi_state = tf.nn.dynamic_rnn(lstm_fw, embedding_inputs, dtype=tf.float32)
            # use attention
            out_fw, out_bw = _outputs
            # 只使用最后一个状态作为句向量
            # out_fw = out_fw[:, -1, :]
            # out_bw = out_bw[:, -1, :]
            
            # 使用att(batch, time_step, hidden_unit * layer)
            _outputs = tf.concat([out_fw, out_bw], axis=2)
            output = attention(_outputs, self.config.attention_dim, self.config.l2_reg_lambda)
            # use last unit to represent the sentence
            # last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
            # output (b*1024)
        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(output, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵 `logits` and `labels` must have the same shape
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
