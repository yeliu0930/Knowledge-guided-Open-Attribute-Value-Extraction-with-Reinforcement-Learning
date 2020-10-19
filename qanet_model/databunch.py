import jieba, time
from functools import reduce
from operator import mul
import json, os, ljqpy
import random, os, sys, h5py
import numpy as np
import editdistance
import re
time.clock()

MODE = 'train'

maxQLen = 30
maxPLen = 50
maxWordLen = 8

vocab_size = 4500
char_size = 3000

batch_size = 64

trainFile    = r'./train_data/train_factoid.json'
validateFile = r'./train_data/valid_factoid.json'

qidAnswers = {}

def CutSentence(s):
	return jieba.lcut(s, HMM=False)

def FullToHalf(s):
	n = []
	for char in s:
		num = ord(char)
		if num == 0x3000:
			num = 32
		elif 0xFF01 <= num <= 0xFF5E:
			num -= 0xfee0
		num = chr(num)
		n.append(num)
	return ''.join(n)

def MakeVocab():
	global id2w, w2id, id2c, c2id
	vocabFile = 'data/wordlist.txt'
	charFile  = 'data/charlist.txt'
	if os.path.exists(vocabFile):
		freqw = ljqpy.LoadCSV(vocabFile)
		freqc = ljqpy.LoadCSV(charFile)
	else:
		freqw = {}; freqc = {}
		for line in ljqpy.LoadCSVg(trainFile):
			line = ''.join(line)
			thisJson = json.loads(line.strip().lower())
			question = thisJson["query"]
			question = re.sub(r'\s+', ' ', question.strip())
			questionTokens = CutSentence(question)
			for t in questionTokens: 
				for c in t: freqc[c] = freqc.get(c, 0) + 10
				t = ChangeToken(t)
				freqw[t] = freqw.get(t, 0) + len(thisJson["passages"]) 
			for passage in thisJson["passages"]:
				context = passage["passage_text"]
				context = FullToHalf(context)
				context = re.sub(r'\s+', ' ', context.strip())
				contextTokens = CutSentence(context)
				for t in contextTokens: 
					for c in t: freqc[c] = freqc.get(c, 0) + 1
					t = ChangeToken(t)
					freqw[t] = freqw.get(t, 0) + 1
		freqw = ljqpy.FreqDict2List(freqw)
		ljqpy.SaveCSV(freqw, vocabFile)
		freqc = ljqpy.FreqDict2List(freqc)
		ljqpy.SaveCSV(freqc, charFile)
	id2w = ['<PAD>', '<UNK>'] + [x[0] for x in freqw[:vocab_size]]
	w2id = {y:x for x,y in enumerate(id2w)}
	id2c = ['<PAD>', '<UNK>'] + [x[0] for x in freqc[:char_size]]
	c2id = {y:x for x,y in enumerate(id2c)}

def ChangeToken(token):
	if token.isdigit(): token = '<NUM%d>' % min(len(token), 6)
	elif re.match('^[a-zA-Z]+$', token): token = '<ENG>'
	return token
				
def Tokens2Intlist(tokens, maxSeqLen):
	ret = np.zeros( maxSeqLen )
	tokens = tokens[:maxSeqLen]
	for i, t in enumerate(tokens):
		t = ChangeToken(t)
		ret[i] = w2id.get(t, 1)
	return ret

def Chars2Intlist(tokens, maxSeqLen):
	ret = np.zeros( (maxSeqLen, maxWordLen) )
	tokens = tokens[:maxSeqLen]
	for i, t in enumerate(tokens):
		for j, c in enumerate(t[:maxWordLen]):
			ret[i,j] = c2id.get(c, 1)
	return ret

def ComputeJaccard(qWordList, tokens):
	l = - len(qWordList) // 2
	r =   len(qWordList) // 2
	sWordList = set(qWordList)
	count = 0.0
	for i in range(l,r):
		if i >= 0 and i < len(tokens) and (tokens[i] in sWordList):
			count += 1.0
	ret = np.zeros(len(tokens))
	for i, token in enumerate(tokens):
		ret[i] = count / len(qWordList)
		if l >= 0 and l < len(tokens) and (tokens[l] in sWordList):
			count -= 1.0
		if r+1 >= 0 and r+1 < len(tokens) and (tokens[r+1] in sWordList):
			count += 1.0
		l += 1; r += 1
	return ret

def ComputeEditDistance(question,tokens):
	context=''.join(tokens)
	ret=[]
	i=0
	for token in tokens:
		j=i+(len(token)/2)
		L=int(j-len(question)/2)
		R=int(j+len(question)/2)
		ret.append(1.0*editdistance.eval(context[max(0,L):min(R,len(context))],question)/len(question))
		i+=len(token)
	return ret

def ContextAux(tokens, maxSeqLen, qsent, passageList=None):
	ret = np.zeros( (maxSeqLen, (4 if passageList else 2)) )
	tokens = tokens[:maxSeqLen]
	for i, t in enumerate(tokens):
		ret[i,0] = 1 if t in qsent else 0
		valid = sum(1 for c in t if c in qsent)
		ret[i,1] = valid / len(t)
	if passageList is not None:
		'''
		context = ''.join(tokens)
		temp = ComputeJaccard(qsent, context)
		index = 0
		for i in range(len(tokens)):
			index_new = index + len(tokens[i])
			fz = sum(temp[j] for j in range(index, index_new))
			ret[i,2] = fz / len(tokens[i])
			index = index_new
		temp = ComputeJaccard(CutSentence(qsent), tokens)
		ret[:len(temp),3] = np.array(temp)
		temp = ComputeEditDistance(qsent, tokens)
		ret[:len(temp),4] = np.array(temp)
		'''
		for i, t in enumerate(tokens):
			for passage in passageList:
				if t in passage: ret[i,2] += 1 / len(passageList)
			valid = 0.0; total = 0.0
			for c in t:
				for passage in passageList:
					if c in passage: valid += 1
					total += 1
			ret[i,3] = valid / total
	return ret


def ComputeAnswerIndex(tokens,answer,answers=None):
	ret_start = [0] * (maxPLen)
	ret_end = [0] * (maxPLen)
	for i in range(len(tokens)):
		if i >= maxPLen: break
		tans=""
		for j in range(i,len(tokens)):
			if j >= maxPLen: break
			tans += tokens[j]
			if tans == answer:
				ret_start[i]=1
				ret_end[j]=1
			if (answers is not None) and (tans in answers):
				ret_start[i]=1
				ret_end[j]=1
			#if len(tans) >= len(answer) * 2: break
	return ret_start,ret_end
	if sum(ret_start) == 0:
		achrs = [set(answer)] + [] if answers is None else [set(a) for a in answers]
		for i in range(len(tokens)):
			if i >= maxPLen: break
			tans = ""
			for j in range(i,len(tokens)):
				if j >= maxPLen: break
				tans += tokens[j]
				chrs = set(tans); vratio = 0
				for aa in achrs:
					vcomm = len(chrs.intersection(aa))
					vratio = max(vratio, vcomm / (len(chrs) + len(aa) - vcomm + 1e-10))
				if vratio > 0.5 and vratio > ret_start[i] and vratio > ret_end[j]:
					ret_start[i]=ret_end[j]=vratio
				#if len(tans)>=len(answer) * 3: break
	return ret_start,ret_end


def ReadQuestionAnswers():
	global qidAnswers 
	fnlist = ['./train_data/qid_answer_expand', './train_data/qid_answer_expand.valid']
	qidAnswers = {}
	for fn in fnlist:
		for tokens in ljqpy.LoadCSV(fn):
			if len(tokens)!=3: continue
			qid = tokens[0]
			answers = tokens[2].split('|')
			qidAnswers[qid]= set(answers)
	
class DataBunch():
	def __init__(self, dataFile, needtest=False, onejson=None):
		self.xQuestion = []
		self.xContext = []
		self.xQuestionC = []
		self.xQuestionA = []
		self.xContextC = []
		self.xContextA = []
		self.y_start = []
		self.y_end = []
		self.startEnd = []
		self.contextRaw = []

		self.realAnswer = []
		self.questionRaw = []
		self.questionId = []
		
		if dataFile is None:
			self.ParseJson(onejson)
			self.ConvertNPArr()
		else:
			if not os.path.isdir('gen_data'): os.mkdir('gen_data')
			self.h5name = 'gen_data/' + os.path.split(dataFile)[-1] + '.h5'
			if os.path.exists(self.h5name): 
				self.Load()
			else:
				print('MAKE H5')
				bad = 0; ii = 0
				for line in ljqpy.LoadCSVg(dataFile):
					line = ''.join(line)
					ii += 1; 
					if ii % 500 == 0: print(ii)
					thisJson = json.loads(line.strip().lower())
					bad += self.ParseJson(thisJson)
				print('bad training samples:', bad)
				self.ConvertNPArr()
				self.Save()

		self.numSamples = self.xQuestion.shape[0]
		self.numQuestions = len(self.questionId)

	def ConvertNPArr(self):
		self.xQuestion = np.array(self.xQuestion)
		self.xContext = np.array(self.xContext)
		self.xQuestionC = np.array(self.xQuestionC)
		self.xQuestionA = np.array(self.xQuestionA)
		self.xContextC = np.array(self.xContextC)
		self.xContextA = np.array(self.xContextA)
		self.y_start = np.array(self.y_start)
		self.y_end = np.array(self.y_end)
		self.startEnd = np.array(self.startEnd)

	def ParseJson(self, thisJson):
		bad = 0
		question = thisJson["query"]
		question = re.sub(r'\s+', ' ', question.strip())
		questionTokens = CutSentence(question)
		xq = Tokens2Intlist(questionTokens, maxQLen)
		xqc = Chars2Intlist(questionTokens, maxQLen)
		qid = thisJson["query_id"]
		thisStart = len(self.xQuestion)		
		passageList=[]
		for passage in thisJson["passages"]:
			context = passage["passage_text"]
			context = FullToHalf(context)
			context = re.sub(r'\s+', ' ', context.strip())
			passageList.append(context)

		for context in passageList:
			contextTokens = CutSentence(context)
			for pci in range(1):
				partcTokens = contextTokens[pci*maxPLen:pci*maxPLen+maxPLen]
				if len(partcTokens) == 0: break
				t_start, t_end = ComputeAnswerIndex(partcTokens,thisJson["answer"],qidAnswers.get(qid))
				if sum(t_start) == 0: 
					if pci > 0 and not needtest: continue
					bad += 1
				self.xContext.append(Tokens2Intlist(partcTokens, maxPLen))
				self.xContextC.append(Chars2Intlist(partcTokens, maxPLen))
				self.xContextA.append(ContextAux(partcTokens, maxPLen, question, passageList))
				self.xQuestion.append(xq)
				self.xQuestionC.append(xqc)
				self.xQuestionA.append(ContextAux(questionTokens, maxQLen, context))
				self.y_start.append(t_start)
				self.y_end.append(t_end)
				self.contextRaw.append(partcTokens)
		self.startEnd.append([thisStart, len(self.xQuestion)-1])
		self.questionId.append(str(qid))
		self.questionRaw.append(question)
		self.realAnswer.append(thisJson["answer"])
		return bad

	def Save(self):
		with h5py.File(self.h5name, 'w') as dfile:
			dfile.create_dataset('xQuestion', data=self.xQuestion)
			dfile.create_dataset('xContext', data=self.xContext)
			dfile.create_dataset('xQuestionC', data=self.xQuestionC)
			dfile.create_dataset('xQuestionA', data=self.xQuestionA)
			dfile.create_dataset('xContextC', data=self.xContextC)
			dfile.create_dataset('xContextA', data=self.xContextA)
			dfile.create_dataset('y_start', data=self.y_start)
			dfile.create_dataset('y_end', data=self.y_end)
			dfile.create_dataset('startEnd', data=self.startEnd)
		ljqpy.SaveCSV(zip(self.questionId, self.questionRaw, self.realAnswer), self.h5name+'.txt')
		if len(self.contextRaw) > 0: ljqpy.SaveCSV(self.contextRaw, self.h5name+'.c.txt')

	def Load(self):
		with h5py.File(self.h5name) as dfile:
			self.xQuestion = dfile['xQuestion'][:]
			self.xContext = dfile['xContext'][:]
			self.xQuestionC = dfile['xQuestionC'][:]
			self.xQuestionA = dfile['xQuestionA'][:]
			self.xContextC = dfile['xContextC'][:]
			self.xContextA = dfile['xContextA'][:]
			self.y_start = dfile['y_start'][:]
			self.y_end = dfile['y_end'][:]
			self.startEnd = dfile['startEnd'][:]
		data = ljqpy.LoadCSV(self.h5name+'.txt')
		self.questionId = [x[0] for x in data]
		self.questionRaw = [x[1] for x in data]
		self.realAnswer = [x[2] for x in data]
		if os.path.exists(self.h5name+'.c.txt'): self.contextRaw = ljqpy.LoadCSV(self.h5name+'.c.txt')

	batch_pointer = 0
	def GetBatch(self, batch_size):
		u, v = self.batch_pointer, self.batch_pointer+batch_size
		r = (self.xQuestion[u:v], self.xContext[u:v], self.y_start[u:v], self.y_end[u:v])
		self.batch_pointer += batch_size
		if self.batch_pointer >= self.numSamples: self.batch_pointer = 0
		return r

	def GetData(self):
		return [self.xQuestion, self.xContext, self.xQuestionC, self.xContextC, self.xQuestionA, self.xContextA], \
				[self.y_start, self.y_end]
	
def GetAnswerList(db_test, y_pred_start, y_pred_end):
	def GetAnswer(i,start,end): return ''.join(db_test.contextRaw[i][start:end+1])
	predAnswerList=[]
	for ii in range(db_test.numQuestions):
		canswerScore={}
		for i in range(db_test.startEnd[ii][0],db_test.startEnd[ii][1]+1):
			thisMax=-1
			canswer="NaN**"
			for j1 in range(maxPLen):
				for j2 in range(j1, min(maxPLen, j1+8)):
					score = y_pred_start[i][j1]*y_pred_end[i][j2]
					if score > thisMax or canswer=="NaN**":
						temp = GetAnswer(i,j1,j2)
						if temp in db_test.questionRaw[ii]: continue
						thisMax, canswer = score, temp
			canswerScore[canswer] = canswerScore.get(canswer, 0) + thisMax
		canswerScore_sort=sorted(canswerScore.items(), key=lambda x:x[1], reverse=True)
		predAnswerList.append(canswerScore_sort[0])
	return predAnswerList

def GetAnswerListNew(db_test, y_pred_start, y_pred_end, mfunc=sum):
	def GetAnswer(i,start,end): return ''.join(db_test.contextRaw[i][start:end+1])
	predAnswerList=[]
	for ii in range(db_test.numQuestions):
		cs, cspos = {}, {}
		for i in range(db_test.startEnd[ii][0], db_test.startEnd[ii][1]+1):
			thisMax = -1e+5; canswer = "NaN**"
			for j1 in range(maxPLen):
				for j2 in range(j1, min(maxPLen, j1+8)):
					score = y_pred_start[i][j1] * y_pred_end[i][j2]
					if score > thisMax or canswer == "NaN**":
						temp = GetAnswer(i,j1,j2)
						if temp in db_test.questionRaw[ii]: continue
						thisMax, canswer = score, temp
						mj1, mj2 = j1, j2
			# one passage
			if thisMax >= 0:
				cs.setdefault(canswer, []).append(thisMax)
				cspos.setdefault(canswer, []).append((i, mj1, mj2))
		anslist = []
		for ans, cslist in cs.items():
			#score = sum(x**2 for x in cslist) / (1 + sum(cslist))
			score = mfunc(cslist)
			anslist.append( (ans, score) )
		anslist.sort(key=lambda x:x[1], reverse=True)
		fans = anslist[0][0]
		#for x in anslist: print(x)
		predAnswerList.append( (fans, cspos[fans], anslist[0][1]) )
	return predAnswerList

		
def Evaluate(db_test, predAnswerList, printResult=True):
	fm=0.0; fz=0.0
		
	if True:
		#print(db_test.numQuestions)
		#print(len(predAnswerList))
		#print(len(db_test.realAnswer))
		for ii in range(db_test.numQuestions):
			fm+=1
			if predAnswerList[ii][0] in qidAnswers[db_test.questionId[ii]]: fz+=1
	if fm==0: fm=1.0
		
	if printResult:
		with open("answer.txt","w",encoding="utf-8") as writer:
			for ii in range(db_test.numQuestions):
				writer.write("{}\t{}\t{}\n".format(db_test.questionRaw[ii],db_test.realAnswer[ii],predAnswerList[ii]))
			writer.close()

	if MODE=="test":
		with open(outputFile,"w",encoding="utf-8") as writer:
			for ii in range(len(db_test.realAnswer)):
				writer.write("{}\t{}\n".format(db_test.questionId[ii], predAnswerList[ii]))
			writer.close()
		
	return fz/fm

MakeVocab()
try:
	import keras.backend as K
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	plt.style.use('ggplot')
except: pass

def LoadModel(net):
	name = net.name
	mm = net.mm
	mfile = '%s.h5' % name
	print(mfile)
	try: mm.load_weights(mfile)
	except: print('\n\nnew model\n')

def Train(name, mm, db_train, db_test):
	mfile = '%s.h5' % name
	try: mm.load_weights(mfile)
	except:	print('\n\nnew model\n')
	test_ans_method = False
	accus, val_losses, train_losses = [], [], []
	num_epochs = 5; max_accu = 0
	for epoch in range(num_epochs):
		print("epoch: %d/%d" % (epoch+1, num_epochs))
		X, Y = db_train.GetData()
		if not test_ans_method:
			hist = mm.fit(X, Y, batch_size=batch_size, epochs=1)
		X, Y = db_test.GetData(); ys, ye = Y
		loss = mm.evaluate(X, Y, batch_size=batch_size)
		y_pred_start,y_pred_end = mm.predict(X, batch_size=batch_size)
		accu = Evaluate(db_test, GetAnswerListNew(db_test, y_pred_start,y_pred_end))
		if test_ans_method:
			print('new_accu:', accu, '  loss: %.4f'%loss[0])
			accu = Evaluate(db_test, GetAnswerList(db_test, y_pred_start,y_pred_end))
			print('old_accu:', accu, '  loss: %.4f'%loss[0])
			break
		if epoch % 4 == 3: 
			newlr = K.get_value(mm.optimizer.lr) * 0.7
			print('newlr: %.7f' % newlr)
			K.set_value(mm.optimizer.lr, newlr)
		if accu > max_accu:
			mm.save_weights(mfile)
			max_accu = accu
		print('accu:', accu, '  loss: %.4f'%loss[0], '  max_accu:', max_accu)
		accus.append(accu); val_losses.append(loss); train_losses.append(hist.history['loss'][0])
		if epoch > 0:
			_, fig = plt.subplots(2, 1, figsize=(12, 8))
			fig[0].plot(train_losses, label='train_loss')
			fig[0].plot(val_losses, label='val_loss')
			fig[1].plot(accus, label='accuracy')
			[i.legend(loc=2) for i in fig]
			plt.savefig('%s.png'%name)
			plt.close('all')
