import numpy as np
import sys, getopt
import csv, math

from sklearn.metrics import auc

PROP = {'A':0,'L':0.21,'R':0.21,'M':0.24,'K':0.26,'Q':0.39,'E':0.40,
        'I':0.41,'W':0.49,'S':0.50,'Y':0.53,'F':0.54,'V':0.61,
        'H':0.61,'N':0.65,'T':0.66,'C':0.68,'D':0.69,'G':1, 'P':3.01}

W = 4
L = 0.5

inputfile = '1MBN_A.fasta'
secondaryfile = '1MBN_SS.fasta'

print 'Opening file ' + inputfile +' ...'
f =  open(inputfile)
lines = f.read().split('>')[1:]
sq = ''.join(lines[0].splitlines()[1:])

print 'Opening file ' + secondaryfile +' ...'
f =  open(secondaryfile)
lines = f.read().split('>')[1:]
ss = ''.join(lines[0].splitlines()[1:])

def getAvgProp(subseq):
    return np.mean([PROP[a] for a in subseq])

def getSS(subseq):
    return 'C' if getAvgProp(subseq) > L else 'H'

def getProp(seq):
    return ''.join([getSS(seq[i-W if i>W else 0:i+W+1 if i<len(seq)-W else -1]) for i in range(len(seq))])

def getRates(pred, true):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for p,t in zip(pred, true):
        if t == 'H':
            if p == 'H':
                TP += 1
            else:
                FN += 1
        else:
            if p == 'H':
                FP += 1
            else:
                TN += 1
    return TP/float(TP+FN), FP/float(FP+TN)

pred = getProp(sq)
print pred
print ss

print getRates(pred, ss)

tpc = []
fpc = []
for i in range(0,1000):
    L = i/float(1000)
    pred = getProp(sq)
    tp, fp = getRates(pred, ss)
    tpc.append(tp)
    fpc.append(fp)

gini = []
for i in range(1, len(tpc)):
    gini.append((fpc[i]-fpc[i-1])*(tpc[i]+fpc[i-1]))
gini = 1-sum(gini)
print gini
auc22 = 1-(gini+1)/float(2)

print auc22

cal = auc(fpc, tpc)
print cal

import matplotlib.pyplot as plt

plt.plot(fpc, tpc)
plt.xlim([0, 1])
plt.ylim([0, 1])

plt.show()
