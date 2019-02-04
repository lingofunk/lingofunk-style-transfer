from copy import deepcopy

import tensorflow as tf

from lingofunk_transfer_style.config import StyleTransferModelConfig
from lingofunk_transfer_style.nn import *
from lingofunk_transfer_style.utils import strip_eos


class BeamState(object):
    def __init__(self, h, inp, sent, nll):
        self.h, self.inp, self.sent, self.nll = h, inp, sent, nll


class Decoder(object):
    def __init__(self, sess, vocab, model, model_config: StyleTransferModelConfig, beam_width: int):
        dim_h = model_config.dim_y + model_config.dim_z
        dim_emb = model_config.dim_emb
        n_layers = model_config.n_layers
        self.vocab = vocab
        self.model = model
        self.max_len = model_config.max_seq_length
        self.beam_width = beam_width
        self.sess = sess

        cell = create_cell(dim_h, n_layers, dropout=1)

        self.inp = tf.placeholder(tf.int32, [None])
        self.h = tf.placeholder(tf.float32, [None, dim_h])

        tf.get_variable_scope().reuse_variables()
        embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_h, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        with tf.variable_scope('generator'):
            inp = tf.nn.embedding_lookup(embedding, self.inp)
            outputs, self.h_prime = cell(inp, self.h)
            logits = tf.matmul(outputs, proj_W) + proj_b
            log_lh = tf.log(tf.nn.softmax(logits))
            self.log_lh, self.indices = tf.nn.top_k(log_lh, self.beam_width)

    def decode(self, h):
        go = self.vocab.word2id['<go>']
        batch_size = len(h)
        init_state = BeamState(h, [go] * batch_size,
            [[] for i in range(batch_size)], [0] * batch_size)
        beam = [init_state]

        for t in range(self.max_len):
            exp = [[] for i in range(batch_size)]
            for state in beam:
                log_lh, indices, h = self.sess.run(
                    [self.log_lh, self.indices, self.h_prime],
                    feed_dict={self.inp: state.inp, self.h: state.h})
                for i in range(batch_size):
                    for l in range(self.beam_width):
                        exp[i].append(BeamState(h[i], indices[i,l],
                            state.sent[i] + [indices[i,l]],
                            state.nll[i] - log_lh[i,l]))

            beam = [deepcopy(init_state) for _ in range(self.beam_width)]
            for i in range(batch_size):
                a = sorted(exp[i], key=lambda k: k.nll)
                for k in range(self.beam_width):
                    beam[k].h[i] = a[k].h
                    beam[k].inp[i] = a[k].inp
                    beam[k].sent[i] = a[k].sent
                    beam[k].nll[i] = a[k].nll

        return beam[0].sent

    def rewrite(self, batch):
        model = self.model
        h_ori, h_tsf= self.sess.run([model.h_ori, model.h_tsf],
            feed_dict={model.dropout: 1,
                       model.batch_size: batch['size'],
                       model.enc_inputs: batch['enc_inputs'],
                       model.labels: batch['labels']})
        ori = self.decode(h_ori)
        ori = [[self.vocab.id2word[i] for i in sent] for sent in ori]
        ori = strip_eos(ori)

        tsf = self.decode(h_tsf)
        tsf = [[self.vocab.id2word[i] for i in sent] for sent in tsf]
        tsf = strip_eos(tsf)

        return ori, tsf
