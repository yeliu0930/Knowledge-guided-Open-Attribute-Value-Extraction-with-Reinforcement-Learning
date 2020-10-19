import json
import os
os.chdir('/Users/ye/Desktop/MCQA')
from databunch import *
import qanet as net
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
LoadModel(net)

os.chdir('/Users/ye/Desktop/MCQA/search_extraction_dataset/train/search-data')
data = []
for line in open('search-result.json', 'r'):
    data.append(json.loads(line))

entities = [entry['s'] for entry in data]
relation = [entry['p'] for entry in data]
sp_pair = [[entry['s'], entry['p']] for entry in data]
unique_pair_index = [1 if sp_pair.count(sp_pair[i])==1 else 0 for i in range(len(sp_pair))]


def GetAnswerListNew2(db_test, y_pred_start, y_pred_end, listans=True, addscore=[]):
    def GetAnswer(i, start, end):
        return ''.join(db_test.contextRaw[i][start:end + 1])

    answer = []
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

            answer.append([canswer, thisMax])

    return answer

# input is a text and a query, output is the candidate answer extracted from text
def GetCandidateAnswer(zz, query):
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
    ret = GetAnswerListNew2(db, y_pred_start, y_pred_end, isanslist, addscore=corpus_addscore)
    # print(ret)
    return ret



def check_qanet_answer(entry):

    query = entry['s'] + "的 " + entry['p'] + "是什么"
    truth = entry['o']
    text_list = entry['corpus']
    answer_list = GetCandidateAnswer(text_list, query)
    answers = ["".join(a[0]) for a in answer_list]
    right = [int(a.lower()==truth.lower()) for a in answers]
    num_correct = sum(right)

    return num_correct

result = []
for i in range(300):
    entry = data[i]
    result.append(check_qanet_answer(entry))

from collections import Counter
print(Counter(result))