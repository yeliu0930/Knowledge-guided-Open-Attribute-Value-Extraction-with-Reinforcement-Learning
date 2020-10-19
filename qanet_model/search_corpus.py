import os, sys, requests, json, time
from bs4 import BeautifulSoup
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from databunch import *
import qanet as net
LoadModel(net)


import re
from answertype_prediction import AnswerTypePrediction


############ this should be the decorators.py ##########################
import redis
import json
import hashlib

def get_hash(string):
	sha1 = hashlib.sha1()
	sha1.update(string)
	return sha1.hexdigest()

class LocalCache():
	def __init__(self,MAXSIZE = 100000):
		self.cache = {}
		self.count = 0
		self.refresh =MAXSIZE
		
	def chk_and_refresh(self):
		if len(self.cache) >= self.refresh:
			self.cache = {}
			print("cache","refreshed")
	
	def setLocal(self,key,value):
		#print("Now1")
		self.chk_and_refresh()
		self.cache[key] = value
		return True
	
	def getLocal(self,key):
		#print("Now2")
		if key in self.cache:
			return self.cache[key]
		else:
			return None

	
	def cache_local(self,func):
		'''This is for caching the Non-Class methods, using the key-value to deal with the 1st input and the output'''
		cache = self.cache
		def real_loading(*args,**kwargs):
			self.chk_and_refresh()
			self.count+=1
			key = args[0]
			#print(key)
			try:
				value = self.getLocal(key)
				if value == None:
					raise Exception("GET CACHE FAILED")
				print("cached in local",key)
			except:
				value = func(*args,**kwargs)
				try:
					self.setLocal(key,value)
				except:
					raise Exception("SET CACHE FAILED,MAYBE UNSUPPORTED KEY")
					
			return value

		return real_loading
	
	def cache_local_classmethod(self,func):
		'''This is for caching the Class methods, using the key-value to deal with the 1st input(after the self) and the output'''
		cache = self.cache
		def real_loading(*args,**kwargs):
			self.count += 1
			self.chk_and_refresh()
			key = args[1]
			print(key)
			try:
				value = self.getLocal(key)
				if value == None:
					raise Exception("GET CACHE FAILED")
				print("cached in local",key)
			except:
				value = func(*args,**kwargs)
				try:
					self.setLocal(key,value)
				except:
					raise Exception("SET CACHE FAILED,MAYBE UNSUPPORTED KEY")
			
			return value

		return real_loading


		
class RedisCache():
	def __init__(self,dbinfo,EXPIRE = 3600,MAXSIZE = 5000,use_local = True):
		self.r0 = redis.Redis(socket_timeout = 2,host = dbinfo["host"],port = dbinfo["port"],decode_responses=True,password = dbinfo["auth"],db = dbinfo["dbid"])
		self.expire = EXPIRE
		self.maxsize = MAXSIZE
		self.locals = {}
		
	def setRedis(self,real_key,value):
		#real_key = get_hash(json.dumps(key,sort_keys=True).encode("utf-8"))
		value = json.dumps(value).encode("utf-8")
		#print("here0")
		try:
			ret = self.r0.set(real_key,value,ex=self.expire)
			#print(ret)
			if ret != True:
				raise Exception("REDIS SET FAILED")
		except:
			print("here1")
			#print("INFO",real_key,value)
			raise Exception("REDIS SET FAILED")
		return True
	
	def getRedis(self,real_key):
		#real_key = get_hash(json.dumps(key,sort_keys=True).encode("utf-8"))
		#print("here1")
		try:
			ret = self.r0.get(real_key)
			if ret == None:
				raise Exception("REDIS GET FAILED")
			return json.loads(ret)
		
		except:
			raise Exception("REDIS GET FAILED")
	
	def cache_redis(self,prefix= "CACHE",keypos = 0,keylen=1):
		if prefix not in self.locals:
			self.locals[prefix] = LocalCache(self.maxsize)
		lc = self.locals[prefix]
		
		def decorator(func):
			def real_func(*args,**kwargs):
				keyend = keypos + keylen
				key = args[keypos:keyend]
				key = [key,kwargs]
				real_key = get_hash(json.dumps(key,sort_keys=True).encode("utf-8"))
				
				# try to get from local first
				try:
					ret = lc.getLocal(real_key)
					if ret == None:
						raise Exception("GET CACHE FAILED")
					print(prefix,"cached in local",real_key)
					return ret
				except:
				# if not got in local then go redis
					try:
						# if get success, save to local and return result
						ret = self.getRedis(prefix+real_key)
						print(prefix,"cached in remote",real_key)
						try:
							lc.setLocal(real_key,ret)
							print(prefix,real_key,"setted in LOCAL")
						except:
							print("SET LOCAL FAILED,MAYBE UNSUPPORTED KEY")
						return ret
					except:
					# if get failed , do the things and save to both local and redis
						ret = func(*args,**kwargs)
						
						try:
							lc.setLocal(real_key,ret)
							print(prefix,real_key,"setted in LOCAL")
						except:
							print(prefix,"SET LOCAL FAILED,MAYBE UNSUPPORTED KEY")
							#raise Exception("SET LOCAL FAILED,MAYBE UNSUPPORTED KEY")
						try:	
							self.setRedis(prefix+real_key,ret)
							print(prefix,real_key,"setted in REMOTE")
						except:
							print(prefix,"SET REMOTE FAILED,MAYBE UNSUPPORTED KEY OR VALUE")
							#raise Exception("SET REMOTE FAILED,MAYBE UNSUPPORTED KEY")
							
							
						return ret
			return real_func
		return decorator
##################################################################

# setting the REDISCACHE
cache = RedisCache({"host":"10.141.208.23",
					"port": 6379,
					"dbid": 12,
					"auth":None},
				   EXPIRE= 604800,
				   MAXSIZE=100000)


def GetAnswerListNew1(db_test, y_pred_start, y_pred_end, listans=True, addscore=[]):
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
			if listans:
				tlen = len(db_test.contextRaw[i])
				if mj2 + 1 < tlen and db_test.contextRaw[i][mj2+1] == '、':
					jj2 = mj2 + 1
					while jj2 < tlen:
						token = db_test.contextRaw[i][jj2]; jj2 += 1
						print(token)
						if token[0] in '，。,. 被是': break
						canswer += token; mj2 += 1
			print(canswer, (i, mj1, mj2), thisMax)
			if thisMax >= 0:
				cs.setdefault(canswer, []).append(thisMax)
				cspos.setdefault(canswer, []).append((i, mj1, mj2))
		anslist = []
		idx = 0
		#multiplier = 1

		if listans:
			addsc = {}
			for ans, cslist in cs.items():
				addsc[ans] = 0
				for ans1, cslist1 in cs.items():
					if ans == ans1: continue
					if ans1 in ans: addsc[ans] += sum(cslist1) * 0.5
			for ans, add in addsc.items(): cs[ans].append(add)
			
		for ans, cslist in cs.items():
			#define the sequence importance
			poss = cspos[ans]
			ncslist = []
			score = 0
			for pos, score0 in zip(poss, cslist):
				if pos[0] < len(addscore): 
					score += score0 * addscore[pos[0]]
			score += sum(cslist)
			#score = sum(x**2 for x in cslist) / (1 + sum(cslist))
			anslist.append( (ans, score) )
			idx += 1

		anslist.sort(key=lambda x:x[1], reverse=True)
		for fans,score in anslist:
			#fans = anslist[0][0]
			#for x in anslist: print(x)
			predAnswerList.append( (fans, cspos[fans], score) )
			
	return predAnswerList

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}
def eliminate_zhidao(string):
	patterns = ["- 提问时间: [0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日","- 最新回答: [0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日","- 发帖时间: [0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日","[0-9]{1,2}个回答"]
	for pattern in patterns:
		p = re.compile(pattern)
		flag = True
		while flag:
			a = re.search(string=string,pattern=p)
			if a == None:
				break
			string = string[:a.span()[0]]+" "+string[a.span()[1]:]
	return string
class SearchCorpus:
	def __init__(self):
		random.seed()
		self.session = requests.Session()
		self.session.get("https://www.baidu.com/", timeout=5,headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'})
		#print(resp.text)		#self.session.headers 
		self.search_count = 0
		self.expire_count = random.randint(5,100)
	def restart_session(self):
		self.session = requests.Session()
		self.session.get("https://www.baidu.com/", timeout=5,headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'})
		self.search_count = 0
		self.expire_count = random.randint(5,100)
	
	def after_search(self):
		self.search_count+=1
		if self.search_count == self.expire_count:
			print("restarting Session")
			self.restart_session()
			
	@cache.cache_redis("ASKBAIDU1|",1, 2)
	def AskBaidu(self, q, page=0):
		url = 'https://www.baidu.com/s?wd=%s' % q
		if page > 0: url += '&pn=%d' % (page*10)
		resp = self.session.get(url, timeout=5,headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'})
		#print(resp.text)
		soup = BeautifulSoup(resp.text, 'html.parser')
		self.soup = soup
		rlst = self.soup.find_all('div', 'result')
		corpus = []
		direct_ans = ''
		if soup.find('div', 'op_exactqa_s_answer') is not None:
			ret = soup.find('div', 'op_exactqa_s_answer').text.strip()
			print("EXACT FOUND",ret)
			corpus.append(ret)
			direct_ans = ret
			#self.after_search()
			#return corpus
		for rr in rlst:
			[x.extract() for x in rr.find_all('span', 'm')]
			zz = rr.text.replace('\xa0', ' ')
			corpus.append(zz)
		self.after_search()
		return corpus, direct_ans
		
	def AskBaike(self,q):
		url = 'https://baike.baidu.com/search/none?word=%s' % q
		resp = requests.get(url, timeout=5, headers=headers)
		resp.encoding = "utf-8"
		soup = BeautifulSoup(resp.text, 'html.parser')
		self.soup = soup
		rlst = self.soup.find_all('p',"result-summary")
		corpus = []
		for rr in rlst:
			[x.extract() for x in rr.find_all('p', 'result-summary')]
			zz = rr.text.replace('\xa0', ' ').replace('<em>',"").replace("</em>","")
			corpus.append(zz)
		return corpus
	def AskSogouWeather(self,q):
		url = 'https://www.sogou.com/web?query=%s' % q
		resp = requests.get(url, timeout=5, headers=headers)
		#print(resp.text)
		soup = BeautifulSoup(resp.text, 'html.parser')
		self.soup = soup
		rlst = self.soup.find_all('div', 'vr-weather161227')
		corpus = []
		for rr in rlst:
			[x.extract() for x in rr.find_all('span', 'm')]
			zz = rr.text.replace('\xa0', ' ').replace("\n"," ")
			corpus.append(zz)
		return corpus
	def AskSogou(self,q):
		url = 'https://www.sogou.com/web?query=%s' % q
		resp = requests.get(url, timeout=5, headers=headers)
		#print(resp.text)
		soup = BeautifulSoup(resp.text, 'html.parser')
		self.soup = soup
		rlst = self.soup.find_all('div', 'rb')
		corpus = []
		for rr in rlst:
			[x.extract() for x in rr.find_all('span', 'm')]
			zz = rr.text.replace('\xa0', ' ').replace("\n"," ")
			corpus.append(zz)
		return corpus
	def AskBing(self,q):
		# not now!
		url = 'https://www.bing.com/search?q=%s' % q
		resp = requests.get(url, timeout=5)
		soup = BeautifulSoup(resp.text)
		ans = ''
		for cls in ['b_xlText b_emphText']:
			node = soup.find('div', class_=cls)
			if node is not None: ans = node.text.strip()	
		return ans;

	def MakeJson(self, query, corpus):
		ret = {'answer':'@NULL@', 'query':query, 'query_id':'0'}
		passages = []
		for text in corpus:
			z = {'url':'', 'passage_text':text}
			passages.append(z)
		ret['passages'] = passages
		return ret
	#@cache.cache_redis("SEARCH|",1)
	def Search(self, kw, query=None, isanslist=False):
		if query is None: query = kw
		zz, direct_ans = self.AskBaidu(kw)
		#if len(zz) == 0: return None
		# here to add the AskBaike
		#zz.extend(self.AskBaike(kw))
		#zz.extend(self.AskSogou(kw))# deal with weather
		if "天气" in kw: zz.extend(self.AskSogouWeather(kw))
		if len(zz) == 0: zz.extend(self.AskSogou(kw))
		# delete the noise from the BAIDU ZHIDAO
		zz = [eliminate_zhidao(string) for string in zz]
		if len(zz) == 0: return None
		ret = self.MakeJson(query, zz)
		db = DataBunch(None, False, onejson=ret)
		X, Y = db.GetData()
		self.db = db
		y_pred_start,y_pred_end = net.mm.predict(X, batch_size=128)#;print(y_pred_start,y_pred_end)
		corpus_addscore = [0.5, 0.5]
		if direct_ans != '': corpus_addscore = [10]
		ret = GetAnswerListNew1(db, y_pred_start, y_pred_end, isanslist, addscore=corpus_addscore)
		#print(ret)
		db.direct_ans = direct_ans
		return ret, db

def have_chinese_number(string):
	numset = set("一二三四五六七八九十零")
	for ch in string:
		if ch in numset: return True
	return False
def have_number(string):
	numset = set('1234567890')
	for ch in string:
		if ch in numset: return True
	return False
def have_bad_name_punctuation(string):
	numset = set('。？ ?;；')
	for ch in string:
		if ch in numset: return True
	return False
def is_instance_list(string):
	return re.search('[、，和,及与/]', string) is not None

sc = SearchCorpus()
#sc.Search('iphone11发布时间')

name = 'MCQA'
desc = 'MCQA'
port = 20014
import traceback
def Run(q): 
	isanslist = False
	if re.search('[三四五六七八九]大', q) is not None or '哪些' in q:
		isanslist = True
	try:
		if '\t' in q:
			kw, qq = q.split('\t')
			ret, db = sc.Search(kw, query=qq, isanslist=isanslist)
		else:
			ret, db = sc.Search(q, isanslist=isanslist)
	except Exception as e:
		#print('traceback.print_exc():'); traceback.print_exc()
		repr(e)
		return {'ans': "", 'details': "", 'score': 0}
	print("---------------")
	for line in ret: print(line);
	
	# dealing the penailty here
	# here start to dealing with the needed anstype
	# the qtype
	test = AnswerTypePrediction()
	nscore = []
	# manual penalty
	ret = [list(x) for x in ret]
	if "定都北京" in q:
		for i,info in enumerate(ret):
			if info[0] == "大都":
				ret[i][2] = info[2]*0.5
	# judging the things			
	if test.isUnit(q):
		for ans,mid,score in ret:
			if have_chinese_number(ans) or have_number(ans): # must have number in UNIT
				nscore.append(score)
			else:
				nscore.append(score*0.3)
	elif test.isTime(q):
		#nscore = []
		for ans,mid,score in ret:
			if have_chinese_number(ans) or have_number(ans): # must have number in TIME
				nscore.append(score)
			else:
				nscore.append(0.8*score)
				
	elif test.isLocation(q):
		#nscore = []
		for ans,mid,score in ret: # still no limit for the LOCATION
			nscore.append(score)
	elif test.isPerson(q):
		#nscore = []
		for ans,mid,score in ret:
			if len(ans) > 15: # the person's name should not be TOO LONG
				score = score*0.8			
			if have_number(ans) or have_bad_name_punctuation(ans):	  # the person's name should not contain ?!。or SPACE or arabic number
				nscore.append(score*0.5)
			else:
				nscore.append(score)
	elif isanslist:
		pn = {y:k for k,y in enumerate('xxx三四五六七八九')}
		nn = ljqpy.RM('([三四五六七八九])大', q)
		num = pn.get(nn, 1)
		for ans, mid, score in ret: 
			#if not is_instance_list(ans): score *= 0.5
			nn = re.sub('[、，和,及与/]', '\t', ans).count('\t') + 1
			if nn != num: score *= 0.2
			nscore.append(score)
	else:
		#nscore = []
		for ans,mid,score in ret: # else remain the same
			nscore.append(score)
	nret = []
	for info,nsc in zip(ret,nscore):
		nret.append([info[0],info[1],nsc])
	#print(nret)
	nret.sort(key=lambda x: -x[2]); print("-------------")
	for line in nret: print(line);
	ret = nret[0]
	details = []
	for i, j1, j2 in ret[1]:
		text = ''.join(db.contextRaw[i][:j1])
		text += '@START@'
		text += ''.join(db.contextRaw[i][j1:j2+1])
		text += '@END@'
		text += ''.join(db.contextRaw[i][j2+1:])
		details.append(text)
	ans = ret[0]; score = ret[2]; print(ret[0],ret[2])
	# a bad reducing 
	if "《" not in ans and "》" in ans:
		ans = ans.replace("》","")	 
	if "《" in ans and "》" not in ans:
		ans = ans.replace("《","")	 
	rr = {'ans': ans, 'details': details, 'score': score}
	if db.direct_ans != '': rr['direct_ans'] = db.direct_ans
	return rr

print('completed')