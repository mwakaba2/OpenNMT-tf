"""Defines a multi source sequence to sequence model. Source sequences are read
from 2 files, encoded separately, and the encoder outputs are concatenated in
time.
"""

import tensorflow as tf
import opennmt as onmt

def model():
    return onmt.models.SequenceClassifier(
      inputter=onmt.inputters.ParallelInputter([
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_1",
              embedding_size=None,
              embedding_file_key="words_embedding",
              trainable=True),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_2",
              embedding_size=512)]),
      encoder=onmt.encoders.ParallelEncoder([
          onmt.encoders.BidirectionalRNNEncoder(
              num_layers=2,
              num_units=512,
              reducer=onmt.utils.ConcatReducer(),
              cell_class=tf.contrib.rnn.LSTMCell,
              dropout=0.3,
              residual_connections=False),
          onmt.encoders.BidirectionalRNNEncoder(
              num_layers=2,
              num_units=512,
              reducer=onmt.utils.ConcatReducer(),
              cell_class=tf.contrib.rnn.LSTMCell,
              dropout=0.3,
              residual_connections=False)],
          outputs_reducer=onmt.utils.ConcatReducer(axis=1)),
      labels_vocabulary_file_key="target_vocabulary")
