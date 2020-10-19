import os, sys, time, re, random, requests, math, json
from collections import defaultdict
import urllib.parse
#import api_utils as api
import Levenshtein
import numpy as np

def RemoveNum(s):
    return re.sub('[0-9.]', '', s)

def MaskNum(s):
    return re.sub('[0-9]', '1', s)

def Jaccard(a, b):
    u = len(set(a).intersection(b))
    if u == 0: return u
    v = len(set(a).union(b))
    return u / v

def JaccardMin(a, b):
    u = len(set(a).intersection(b))
    if u == 0: return u
    v = min(len(set(a)), len(set(b)))
    return u / v

def GetLCS(u, v):
    u, v = u[-30:], v[-30:]
    lu, lv = len(u), len(v)
    f, g = [[0] * 31 for i in range(31)], [[0] * 31 for i in range(31)]
    f[0][0] = 0
    for i in range(1,lu+1):
        for j in range(1,lv+1):
            f[i][j] = max(f[i-1][j], f[i][j-1], f[i-1][j-1] + (1 if u[i-1]==v[j-1] else 0))
            if f[i-1][j] == f[i][j]: g[i][j] = 1
            elif f[i][j-1] == f[i][j]: g[i][j] = 2
            else: g[i][j] = 3
    return f[lu][lv]
    rr, x, y = [], lu, lv
    while x > 0 and y > 0:
        gg = g[x][y]
        if gg == 3: rr.append(u[x-1])
        x -= gg & 1
        y -= (gg & 2) // 2
    return ''.join(reversed(rr))

def LevenSim(x, y):
    if len(x) * len(y) == 0: return 1
    return 1 - Levenshtein.distance(x, y) / max(len(x), len(y))

def LCSSim(x, y):
    if len(x) * len(y) == 0: return 1
    return GetLCS(x, y) / max(len(x), len(y))

def LCSSimMin(x, y):
    if len(x) * len(y) == 0: return 1
    return GetLCS(x, y) / min(len(x), len(y))

def AllConcepts(es1, es2):
    global cache_concepts
    cands = list(set(es1+es2))
    cc = [x for x in cands if x not in cache_concepts]
    ccs = api.GetEntConceptsMulti(cc) if len(cc) > 0 else {}
    for x, y in ccs.items(): cache_concepts[x] = y
    for x in cands:
        if x not in ccs: ccs[x] = cache_concepts.get(x, [])
    cs1, cs2 = set(), set()
    for e in es1: cs1.update([x[0] for x in ccs.get(e, [])])
    for e in es2: cs2.update([x[0] for x in ccs.get(e, [])])
    return cs1, cs2


def GetFeature(v, ans, mes={}):
    v = v.lower()
    ans = ans.lower()
    gvalue_r = RemoveNum(ans).replace(' ', '')
    gvalue_m = MaskNum(ans).replace(' ', '')
    svalue_r = RemoveNum(v).replace(' ', '')
    svalue_m = MaskNum(v).replace(' ', '')
    f_levenr = LevenSim(svalue_r, gvalue_r)
    f_levenm = LevenSim(svalue_m, gvalue_m)
    f_lcsr = LCSSim(svalue_r, gvalue_r)
    f_lcsm = LCSSim(svalue_m, gvalue_m)
    f_leven = LevenSim(v, ans)
    f_lcs = LCSSim(v, ans)
    ss1, ss2 = mes.get(v, []), mes.get(ans, [])
    # difference in text length
    d_len = abs(len(v) - len(ans))

    f_ent = Jaccard(ss1, ss2)
    cs1, cs2 = AllConcepts(ss1[:3], ss2[:3])
    f_cate = JaccardMin(cs1, cs2)
    return f_levenm, f_lcsm, f_levenr, f_lcsr, f_leven, f_lcs, d_len, f_ent, f_cate
