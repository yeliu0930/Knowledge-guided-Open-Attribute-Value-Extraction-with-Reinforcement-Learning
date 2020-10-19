import jieba, time
from functools import reduce
from operator import mul
import json, os, ljqpy
import random, os, sys, h5py
import numpy as np
import re
time.clock()

MODE="train"
name = 'bidaf_sru'
embedding_dim = 64
char_embd_dim = 64

lstm_dim = 64
batch_size = 200

from databunch import DataBunch, ReadQuestionAnswers, MakeVocab, Train
from databunch import maxQLen, maxPLen, maxWordLen, vocab_size, char_size
from databunch import trainFile, validateFile

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
import keras.backend as K
from keras.layers import *
from keras.optimizers import *
from keras.models import *

dropout_rate = 0.5

from sru import SRU

quest_input = Input(shape=(maxQLen,), dtype='int32')
cand_input = Input(shape=(maxPLen,), dtype='int32')

questC_input = Input(shape=(maxQLen,maxWordLen), dtype='int32')
candC_input = Input(shape=(maxPLen,maxWordLen), dtype='int32')

embed_layer = Embedding(vocab_size+2, embedding_dim) 
quest_emb = SpatialDropout1D(dropout_rate)( embed_layer(quest_input) )
cand_emb  = SpatialDropout1D(dropout_rate)( embed_layer(cand_input) )

cembed_layer = Embedding(char_size+2, char_embd_dim)

char_input = Input(shape=(maxWordLen,), dtype='int32')
c_emb = SpatialDropout1D(dropout_rate)( cembed_layer(char_input) )
cc = Conv1D(char_embd_dim, 3, padding='same')(c_emb)
cc = LeakyReLU()(cc)
cc = Lambda(lambda x:K.mean(x, 1), output_shape=lambda d:(d[0], d[2]))(cc)
#cc = Flatten()(cc)
char_model = Model(char_input, cc)

qc_emb = TimeDistributed(char_model)(questC_input)
cc_emb = TimeDistributed(char_model)(candC_input)

quest_emb = concatenate([quest_emb, qc_emb])
cand_emb  = concatenate([cand_emb, cc_emb])

Q = Bidirectional(SRU(lstm_dim, return_sequences=True, dropout=dropout_rate) )(quest_emb)
C = Bidirectional(SRU(lstm_dim, return_sequences=True, dropout=dropout_rate) )(cand_emb)

CC = Lambda(lambda C:K.repeat_elements(C[:,:,None,:], maxQLen, 2))(C)
QQ = Lambda(lambda Q:K.repeat_elements(Q[:,None,:,:], maxPLen, 1))(Q)
S_hat = concatenate([CC, QQ, multiply([CC, QQ])])
S = Reshape((maxPLen, maxQLen))( TimeDistributed( TimeDistributed( Dense(1, use_bias=False) ) )(S_hat) )

aa = Activation('softmax')(S)
U_hat = Lambda(lambda x:K.batch_dot(x[0], x[1]))([aa, Q])

bb = Activation('softmax')( Lambda( lambda x: K.max(x, 2))(S) )
h_hat = Lambda( lambda x: K.batch_dot(K.expand_dims(x[0],1), x[1]) )([bb, C]) 
H_hat = Lambda( lambda x: K.repeat_elements(x, maxPLen, 1) )(h_hat)

G = concatenate( [C, U_hat, multiply([C,U_hat]), multiply([C,H_hat])] )
M1 = Bidirectional(SRU(lstm_dim, return_sequences=True, dropout=dropout_rate) )(G)
M2 = Bidirectional(SRU(lstm_dim, return_sequences=True, dropout=dropout_rate) )(M1)

GM1 = concatenate([G, M1])
GM2 = concatenate([G, M2])

final_start = Activation('sigmoid',name='s')(Flatten()( TimeDistributed(Dense(1, use_bias=False))(GM1) ))
final_end   = Activation('sigmoid',name='e')(Flatten()( TimeDistributed(Dense(1, use_bias=False))(GM2) ))

from weight_norm import AdamWithWeightnorm
mm = Model(inputs=[quest_input, cand_input, questC_input, candC_input], outputs=[final_start, final_end])
mm.compile(AdamWithWeightnorm(0.01, 0.5), 'binary_crossentropy')
			
if __name__ == "__main__":
	MakeVocab()
	mm.summary()
	if MODE=="train":
		ReadQuestionAnswers()
		db_train = DataBunch(trainFile)
		print('train: %d questions, %d samples' % (db_train.numQuestions, db_train.numSamples))
		db_test = DataBunch(validateFile, True)
		print('test: %d questions, %d samples' % (db_test.numQuestions, db_test.numSamples))
		print('load ok %.3f' % time.clock())

		print(len(db_train.contextRaw))
		print(len(db_test.contextRaw))
		
		Train(name, mm, db_train, db_test)
	if MODE=="test":
		db_test = DataBunch(validateFile, True)
		model = BidafModel()
		model.Construct()

	print('completed %.3f' % time.clock())