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
    def __init__(self, MAXSIZE=100000):
        self.cache = {}
        self.count = 0
        self.refresh = MAXSIZE

    def chk_and_refresh(self):
        if len(self.cache) >= self.refresh:
            self.cache = {}
            print("cache", "refreshed")

    def setLocal(self, key, value):
        # print("Now1")
        self.chk_and_refresh()
        self.cache[key] = value
        return True

    def getLocal(self, key):
        # print("Now2")
        if key in self.cache:
            return self.cache[key]
        else:
            return None

    def cache_local(self, func):
        '''This is for caching the Non-Class methods, using the key-value to deal with the 1st input and the output'''
        cache = self.cache

        def real_loading(*args, **kwargs):
            self.chk_and_refresh()
            self.count += 1
            key = args[0]
            # print(key)
            try:
                value = self.getLocal(key)
                if value == None:
                    raise Exception("GET CACHE FAILED")
                print("cached in local", key)
            except:
                value = func(*args, **kwargs)
                try:
                    self.setLocal(key, value)
                except:
                    raise Exception("SET CACHE FAILED,MAYBE UNSUPPORTED KEY")

            return value

        return real_loading

    def cache_local_classmethod(self, func):
        '''This is for caching the Class methods, using the key-value to deal with the 1st input(after the self) and the output'''
        cache = self.cache

        def real_loading(*args, **kwargs):
            self.count += 1
            self.chk_and_refresh()
            key = args[1]
            print(key)
            try:
                value = self.getLocal(key)
                if value == None:
                    raise Exception("GET CACHE FAILED")
                print("cached in local", key)
            except:
                value = func(*args, **kwargs)
                try:
                    self.setLocal(key, value)
                except:
                    raise Exception("SET CACHE FAILED,MAYBE UNSUPPORTED KEY")

            return value

        return real_loading


class RedisCache():
    def __init__(self, dbinfo, EXPIRE=3600, MAXSIZE=5000, use_local=True):
        self.r0 = redis.Redis(socket_timeout=2, host=dbinfo["host"], port=dbinfo["port"], decode_responses=True,
                              password=dbinfo["auth"], db=dbinfo["dbid"])
        self.expire = EXPIRE
        self.maxsize = MAXSIZE
        self.locals = {}

    def setRedis(self, real_key, value):
        # real_key = get_hash(json.dumps(key,sort_keys=True).encode("utf-8"))
        value = json.dumps(value).encode("utf-8")
        # print("here0")
        try:
            ret = self.r0.set(real_key, value, ex=self.expire)
            # print(ret)
            if ret != True:
                raise Exception("REDIS SET FAILED")
        except:
            print("here1")
            # print("INFO",real_key,value)
            raise Exception("REDIS SET FAILED")
        return True

    def getRedis(self, real_key):
        # real_key = get_hash(json.dumps(key,sort_keys=True).encode("utf-8"))
        # print("here1")
        try:
            ret = self.r0.get(real_key)
            if ret == None:
                raise Exception("REDIS GET FAILED")
            return json.loads(ret)

        except:
            raise Exception("REDIS GET FAILED")

    def cache_redis(self, prefix="CACHE", keypos=0, keylen=1):
        if prefix not in self.locals:
            self.locals[prefix] = LocalCache(self.maxsize)
        lc = self.locals[prefix]

        def decorator(func):
            def real_func(*args, **kwargs):
                keyend = keypos + keylen
                key = args[keypos:keyend]
                key = [key, kwargs]
                real_key = get_hash(json.dumps(key, sort_keys=True).encode("utf-8"))

                # try to get from local first
                try:
                    ret = lc.getLocal(real_key)
                    if ret == None:
                        raise Exception("GET CACHE FAILED")
                    print(prefix, "cached in local", real_key)
                    return ret
                except:
                    # if not got in local then go redis
                    try:
                        # if get success, save to local and return result
                        ret = self.getRedis(prefix + real_key)
                        print(prefix, "cached in remote", real_key)
                        try:
                            lc.setLocal(real_key, ret)
                            print(prefix, real_key, "setted in LOCAL")
                        except:
                            print("SET LOCAL FAILED,MAYBE UNSUPPORTED KEY")
                        return ret
                    except:
                        # if get failed , do the things and save to both local and redis
                        ret = func(*args, **kwargs)

                        try:
                            lc.setLocal(real_key, ret)
                            print(prefix, real_key, "setted in LOCAL")
                        except:
                            print(prefix, "SET LOCAL FAILED,MAYBE UNSUPPORTED KEY")
                        # raise Exception("SET LOCAL FAILED,MAYBE UNSUPPORTED KEY")
                        try:
                            self.setRedis(prefix + real_key, ret)
                            print(prefix, real_key, "setted in REMOTE")
                        except:
                            print(prefix, "SET REMOTE FAILED,MAYBE UNSUPPORTED KEY OR VALUE")
                        # raise Exception("SET REMOTE FAILED,MAYBE UNSUPPORTED KEY")

                        return ret

            return real_func

        return decorator

##################################################################

# setting the REDISCACHE
cache = RedisCache({"host": "10.141.208.23",
                    "port": 6379,
                    "dbid": 12,
                    "auth": None},
                   EXPIRE=604800,
                   MAXSIZE=100000)

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}


def GetAnswerListNew1(db_test, y_pred_start, y_pred_end, listans=True, addscore=[]):
    def GetAnswer(i, start, end):
        return ''.join(db_test.contextRaw[i][start:end + 1])

    predAnswerList = []
    for ii in range(db_test.numQuestions):
        cs, cspos = {}, {}
        for i in range(db_test.startEnd[ii][0], db_test.startEnd[ii][1] + 1):
            thisMax = -1e+5;
            canswer = "NaN**"
            for j1 in range(maxPLen):
                for j2 in range(j1, min(maxPLen, j1 + 8)):
                    score = y_pred_start[i][j1] * y_pred_end[i][j2]
                    if score > thisMax or canswer == "NaN**":
                        temp = GetAnswer(i, j1, j2)
                        if temp in db_test.questionRaw[ii]: continue
                        thisMax, canswer = score, temp
                        mj1, mj2 = j1, j2
            # one passage
            if listans:
                tlen = len(db_test.contextRaw[i])
                if mj2 + 1 < tlen and db_test.contextRaw[i][mj2 + 1] == '、':
                    jj2 = mj2 + 1
                    while jj2 < tlen:
                        token = db_test.contextRaw[i][jj2];
                        jj2 += 1
                        print(token)
                        if token[0] in '，。,. 被是': break
                        canswer += token;
                        mj2 += 1
            print(canswer, (i, mj1, mj2), thisMax)
            if thisMax >= 0:
                cs.setdefault(canswer, []).append(thisMax)
                cspos.setdefault(canswer, []).append((i, mj1, mj2))
        anslist = []
        idx = 0
        # multiplier = 1

        if listans:
            addsc = {}
            for ans, cslist in cs.items():
                addsc[ans] = 0
                for ans1, cslist1 in cs.items():
                    if ans == ans1: continue
                    if ans1 in ans: addsc[ans] += sum(cslist1) * 0.5
            for ans, add in addsc.items(): cs[ans].append(add)

        for ans, cslist in cs.items():
            # define the sequence importance
            poss = cspos[ans]
            ncslist = []
            score = 0
            for pos, score0 in zip(poss, cslist):
                if pos[0] < len(addscore):
                    score += score0 * addscore[pos[0]]
            score += sum(cslist)
            # score = sum(x**2 for x in cslist) / (1 + sum(cslist))
            anslist.append((ans, score))
            idx += 1

        anslist.sort(key=lambda x: x[1], reverse=True)
        for fans, score in anslist:
            # fans = anslist[0][0]
            # for x in anslist: print(x)
            predAnswerList.append((fans, cspos[fans], score))

    return predAnswerList

##################################################################
num_doc = 25 # at least 25 articles

# get text files from Sogou
class SearchSogou:
    def __init__(self):
        random.seed()
        self.session = requests.Session()
        self.session.get("https://www.baidu.com/", timeout=5, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'})
        # print(resp.text)		#self.session.headers
        self.search_count = 0
        self.expire_count = random.randint(5, 100)

    def restart_session(self):
        self.session = requests.Session()
        self.session.get("https://www.baidu.com/", timeout=5, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'})
        self.search_count = 0
        self.expire_count = random.randint(5, 100)

    def after_search(self):
        self.search_count += 1
        if self.search_count == self.expire_count:
            print("restarting Session")
            self.restart_session()

    def AskSogou(self, q, page=1):
        url = 'https://www.sogou.com/web?query=%s' % q
        if page > 0: url += '&page=%d' % page
        resp = requests.get(url, timeout=5, headers=headers)
        # print(resp.text)
        soup = BeautifulSoup(resp.text, 'html.parser')
        self.soup = soup
        rlst = self.soup.find_all('div', 'rb')
        corpus = []
        for rr in rlst:
            [x.extract() for x in rr.find_all('span', 'm')]
            zz = rr.text.replace('\xa0', ' ').replace("\n", " ")
            corpus.append(zz)
        return corpus

    # @cache.cache_redis("SEARCH|",1)
    def Search(self, kw, query=None):
        if query is None: query = kw
        doc = []
        page = 0
        while len(doc) <= num_doc:
            doc.extend(self.AskSogou(kw, page))
            page += 1
        return doc

# example
# sc = SearchSogou()
# q = "iphone11的发布时间"
# sc.AskSogou(q, page=1)
# all_article = sc.Search(q)



# input is a text and a query, output is the candidate answer extracted from text
def get_answer(zz, query):
    ret = {'answer': '@NULL@', 'query': query, 'query_id': '0'}
    passages = []
    for text in zz:
        z = {'url': '', 'passage_text': text}
        passages.append(z)
    ret['passages'] = passages
    db = DataBunch(None, False, onejson=ret)
    X, Y = db.GetData()
    y_pred_start, y_pred_end = net.mm.predict(X, batch_size=128)  # ;print(y_pred_start,y_pred_end)
    corpus_addscore = [0.5, 0.5]
    isanslist = False
    ret = GetAnswerListNew1(db, y_pred_start, y_pred_end, isanslist, addscore=corpus_addscore)
    # print(ret)
    return ret

# example
# get_answer([all_article[5]], q)

# main function: given a query, extract several texts and get candidate answers for each text
def Extract(q):
    sc = SearchSogou()
    all_text = sc.Search(q)
    output = []
    print('finish collecting articles')
    for i in range(len(all_text)):
        answer_i = get_answer([all_text[i]], q)
        #out = {}
        # answer_i[0][0] is the candidate answer, answer_i[0][1] is the confidence
        output.append([all_text[i],answer_i[0][0], answer_i[0][1]])
        #output[all_text[i]] = answer_i
    return output

q = "iphone11的发布时间"
#q = "iphone7的颜色"
#q = "西红柿首富的导演是谁"
#q = "iphone8的芯片"
out = Extract(q)
out

