import numpy as np
import sys, getopt
import csv, math
from numpy.linalg import eig
import pickle
import scipy.sparse as sp
from collections import Counter

np.set_printoptions(edgeitems=10)

netFileA = 'N1.txt'
netFileB = 'N2.txt'

print 'Opening file ' + netFileA +' ...'
f =  open(netFileA)
linesA = f.read().splitlines()
linesA = np.array([l.split(' ') for l in linesA]).T
classA1 = np.unique(np.concatenate((linesA[0],linesA[1]))).tolist()

print 'Opening file ' + netFileB +' ...'
f =  open(netFileB)
linesB = f.read().splitlines()
linesB = np.array([l.split(' ') for l in linesB]).T
classB1 = np.unique(np.concatenate((linesB[0],linesB[1]))).tolist()
'''
def getInteractionMatrix(classes, interactions):
    print classes
    NA = np.zeros((len(classes), len(classes)), dtype=int)

    for vu in interactions.T:
        i, j = classes.index(vu[0]), classes.index(vu[1])

        NA[i,j], NA[j,i]= 1,1
    return NA

def getAMatrix(N1, N2):
    N1sum = np.sum(N1, axis=0)
    N2sum = np.sum(N2, axis=0)
    A = np.zeros((len(N1sum)*len(N2sum), len(N1sum)*len(N2sum)))

    for i,ni in enumerate(N1sum):
        for j,nj in enumerate(N2sum):
            for u,nu in enumerate(N1sum):
                for v,nv in enumerate(N2sum):
                    Ai = (i*len(N1sum))+j
                    Aj = (u*len(N1sum))+v
                    val = 1/float(nu*nv) if N1[i,u] and N2[j,v] else 0
                    A[Ai,Aj] = val
    return A

NA = getInteractionMatrix(classA1, linesA)
NB = getInteractionMatrix(classB1, linesB)

Am = getAMatrix(NA,NB)
np.save('amat', Am)


Am = np.load('amat.npy')

print np.sum(Am, axis=1)
print np.sum(Am, axis=0)

w,v = eig(Am)

np.save('wvec', w)
np.save('vvec', v)
'''
w = np.load('wvec.npy')
v = np.load('vvec.npy')

indexes = np.argsort(w)
v = v[indexes][-1]

Ain = np.concatenate([[i]*len(classB1) for i in range(len(classA1))])
Bin = np.array(range(len(classB1))*len(classA1))

sort = np.argsort(v)[::-1]

resA = []
resB = []
Ain = Ain[sort]
Bin = Bin[sort]
v = v[sort]


while len(Ain) > 0 and len(Bin) > 0 :
    resA.append(Ain[0])
    resB.append(Bin[0])
    whA = np.where(Ain == Ain[0])[0]
    whB = np.where(Bin == Bin[0])[0]
    stwh = np.unique(np.concatenate((whA,whB)))
    Ain = np.delete(Ain, stwh)
    Bin = np.delete(Bin, stwh)

nodesA = [classA1[x] for x in resA]
nodesB = [classB1[x] for x in resB]

for a,b in zip(nodesA, nodesB):
    print a, b, str(1)
