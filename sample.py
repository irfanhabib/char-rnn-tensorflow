#!/usr/bin/env python

from __future__ import print_function

import argparse
import os
from six.moves import cPickle

from six import text_type

parser = argparse.ArgumentParser(
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_dir', type=str, default='models',
                    help='model directory to store checkpointed models')
parser.add_argument('-n', type=int, default=500,
                    help='number of characters to sample')
parser.add_argument('--prime', type=text_type, default=u'',
                    help='prime text')
parser.add_argument('--sample', type=int, default=1,
                    help='0 to use max at each timestep, 1 to sample at '
                         'each timestep, 2 to sample on spaces')

args = parser.parse_args()

import tensorflow as tf
from model import Model

def sample(args, save_dir):
    tf.reset_default_graph()
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    #Use most frequent char if no prime is given
    if args.prime == '':
        args.prime = chars[0]
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            return model.sample(sess, chars, vocab, args.n, args.prime, args.sample)
        sess.close()

if __name__ == '__main__':

    outputData = dict()
    length = 0
    numberOfModels = 0
    for dirName in os.listdir(args.model_dir):
        outputData[numberOfModels] = sample(args, args.model_dir + "/" + dirName).split("\n")
        length = len(outputData[numberOfModels])
        numberOfModels += 1

    csvOutput = []
    for x in range(length):
        entry = ""
        for n in range(numberOfModels):
            entry = entry + "" + str(outputData[n][x]) + ","
        csvOutput.append(entry)

    text_file = open("output.csv", "w")
    for i in range(length):
        text_file.write(csvOutput[i] + "\n")
    text_file.close()

