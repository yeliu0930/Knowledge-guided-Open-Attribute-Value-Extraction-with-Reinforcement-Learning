import jieba, time
from functools import reduce
from operator import mul
import json, os, ljqpy
import random, os, sys, h5py
import numpy as np
import re, h5py
time.clock()

MODE = "train"
name = 'qanet_mixall_alter0'
#name = 'qanet_72.04'
embedding_dim = 50
char_embd_dim = 30

lstm_dim = 60

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
from keras.regularizers import *
K.set_image_data_format('channels_last')

with h5py.File('w2v.h5', 'r') as dfile: w2v = dfile['w2v'][:]

dropout_rate = 0.5

quest_input = Input(shape=(maxQLen,), dtype='int32')
cand_input = Input(shape=(maxPLen,), dtype='int32')

questC_input = Input(shape=(maxQLen,maxWordLen), dtype='int32')
candC_input = Input(shape=(maxPLen,maxWordLen), dtype='int32')

questA_input = Input(shape=(maxQLen,2))
candA_input  = Input(shape=(maxPLen,4))

def MaskWord(x, dropout=0.2):
	# Mask to 1
	mask = K.cast(K.random_uniform(K.shape(x)) < dropout, 'float32')
	mask = K.cast(K.not_equal(x, 0), 'float32') * mask
	xm = K.cast(x, 'float32') * (1 - mask) + mask
	return K.in_train_phase(K.cast(xm, 'int32'), K.cast(x, 'int32'))
MaskWordLayer = Lambda(MaskWord)

qi  = MaskWordLayer(quest_input)
ci  = MaskWordLayer(cand_input)
qci = MaskWordLayer(questC_input)
cci = MaskWordLayer(candC_input)

embed_layer = Embedding(vocab_size+2, embedding_dim, weights=[w2v]) 
quest_emb = Dropout(dropout_rate, (None, 1, None))( embed_layer(qi) )
cand_emb  = Dropout(dropout_rate, (None, 1, None))( embed_layer(ci) )

cembed_layer = Embedding(char_size+2, char_embd_dim)

class Highway:
	def __init__(self, dim, layers=2):
		self.linears = [Dense(dim, activation='relu') for _ in range(layers+1)]
		self.gates = [Dense(dim, activation='sigmoid') for _ in range(layers)]
	def __call__(self, x):
		for linear, gate in zip(self.linears, self.gates):
			g = gate(x)
			z = linear(x)
			x = Lambda(lambda x:x[0]*x[1]+(1-x[0])*x[2])([g, z, x])
		return x

char_input = Input(shape=(maxWordLen,), dtype='int32')
c_emb = Dropout(dropout_rate, (None, 1, None))( cembed_layer(char_input) )
c_mask = Lambda(lambda x:K.cast(K.not_equal(x, 0), 'float32'))(char_input)
cc = Conv1D(char_embd_dim, 3, padding='same')(c_emb)
cc = LeakyReLU()(cc)
cc = multiply([cc, Reshape((-1,1))(c_mask)])
cc = Lambda(lambda x:K.sum(x, 1))(cc)
char_model = Model(char_input, cc)

qc_emb = TimeDistributed(char_model)(qci)
cc_emb = TimeDistributed(char_model)(cci)

quest_emb = concatenate([quest_emb, qc_emb, questA_input])
cand_emb  = concatenate([cand_emb, cc_emb, candA_input])

highway1 = Highway(dim=128, layers=2)
highway2 = Highway(dim=128, layers=2)

quest_emb = Dense(128, activation='relu')(quest_emb)
cand_emb  = Dense(128, activation='relu')(cand_emb)

quest_emb = highway1(quest_emb)
cand_emb  = highway1(cand_emb)

class FeedForward():
	def __init__(self, d_hid, dropout=0.1):
		self.forward = Dense(d_hid, activation='relu')
		self.dropout = Dropout(dropout)
	def __call__(self, x):
		x = self.forward(x) 
		x = self.dropout(x)
		return x

class LayerDropout:
	def __init__(self, dropout=0.2):
		self.dropout = dropout
	def __call__(self, old, new):
		def func(args):
			old, new = args
			pred = K.random_uniform([]) < self.dropout
			ret = K.switch(pred, old, old+K.dropout(new, self.dropout))
			return K.in_train_phase(ret, old+new)
		return Lambda(func)([old, new])

from transformer import QANet_Encoder

emb_enc1  = QANet_Encoder(128, n_head=2, n_conv=4, n_block=1, kernel_size=7)
emb_enc2  = QANet_Encoder(128, n_head=2, n_conv=4, n_block=1, kernel_size=7)
main_enc1 = QANet_Encoder(128, n_head=2, n_conv=2, n_block=2, kernel_size=5)
main_enc2 = QANet_Encoder(128, n_head=2, n_conv=2, n_block=2, kernel_size=5)
main_enc3 = QANet_Encoder(128, n_head=2, n_conv=2, n_block=2, kernel_size=5)

Qmask = Lambda(lambda x:K.cast(K.not_equal(x,0), 'float32'))(quest_input)
Cmask = Lambda(lambda x:K.cast(K.not_equal(x,0), 'float32'))(cand_input)

Q = emb_enc1(quest_emb, Qmask)
C = emb_enc1(cand_emb,  Cmask)

# the length is same as seq1
def BiDAF_Attention(seq1, seq2, len1, len2, seq1_mask=None, seq2_mask=None):
	if seq1_mask is not None:
		sseq1_mask = Lambda(lambda x:(1-x)*(-1e+9))(seq1_mask)
		sseq2_mask = Lambda(lambda x:(1-x)*(-1e+9))(seq2_mask)
	C, Q = seq1, seq2
	CC = Lambda(lambda x:K.repeat_elements(x[:,:,None,:], len2, 2))(C)
	QQ = Lambda(lambda x:K.repeat_elements(x[:,None,:,:], len1, 1))(Q)
	S_hat = concatenate([CC, QQ, multiply([CC, QQ])])
	S = Reshape((len1, len2))( TimeDistributed(TimeDistributed(Dense(1, use_bias=False)))(S_hat) )
	if seq1_mask is not None: S = add([S, sseq2_mask])
	aa = Activation('softmax')(S) 
	U_hat = Lambda(lambda x:K.batch_dot(x[0], x[1]))([aa, Q])
	SS = Lambda(lambda x: K.max(x, 2))(S)
	if seq1_mask is not None: SS = add([SS, sseq1_mask])
	bb = Activation('softmax')(SS)
	h_hat = Lambda( lambda x: K.batch_dot(K.expand_dims(x[0],1), x[1]) )([bb, C]) 
	H_hat = Lambda( lambda x: K.repeat_elements(x, len1, 1) )(h_hat)
	G = concatenate([C, U_hat, multiply([C,U_hat]), multiply([C,H_hat])])
	return G

G = BiDAF_Attention(C, Q, maxPLen, maxQLen, Cmask, Qmask)

G0 = main_enc1(G,  Cmask)
M1 = main_enc2(G0, Cmask)
M2 = main_enc3(M1, Cmask)

GM1 = concatenate([G0, M1])
GM2 = concatenate([G0, M2])

F1 = TimeDistributed(Dense(1, use_bias=False))(GM1)
F2 = TimeDistributed(Dense(1, use_bias=False))(GM2)

final_start = Activation('sigmoid', name='s')(Flatten()( F1 ))
final_end   = Activation('sigmoid', name='e')(Flatten()( F2 ))

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.not_equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
		return - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + 1e-8)) - \
				 K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + 1e-8))
	return focal_loss_fixed

from weight_norm import AdamWithWeightnorm
mm = Model(inputs=[quest_input, cand_input, questC_input, candC_input, questA_input, candA_input], \
		outputs=[final_start, final_end])
mm.compile(Adam(0.001, clipvalue=0.1), focal_loss())

if __name__ == "__main__":
	mm.summary()
	if MODE=="train":
		ReadQuestionAnswers()
		db_train = DataBunch(trainFile)
		print('train: %d questions, %d samples' % (db_train.numQuestions, db_train.numSamples))
		db_test = DataBunch(validateFile, True)
		print('test: %d questions, %d samples' % (db_test.numQuestions, db_test.numSamples))
		print('load ok %.3f' % time.clock())

		#db_all = DataBunch('../data/train_factoid_withvalid.json')
		#Train(name, mm, db_all, db_test)

		print(len(db_train.contextRaw))
		print(len(db_test.contextRaw))

		Train(name, mm, db_train, db_test)
	if MODE=="test":
		db_test = DataBunch(validateFile, True)

	print('completed %.3f' % time.clock())