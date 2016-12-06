import numpy as np
import sys, getopt
import csv, math
from scipy import linalg
import pickle
import scipy.sparse as sp
from collections import Counter

np.set_printoptions(edgeitems=500, precision=4)

def main(argv):
    helpMsg = 'H2Q2.py -i <network1> -j <network2>'

    netFileA = 'NT1.txt'
    netFileB = 'NT2.txt'

    try:
        opts, args = getopt.getopt(argv,"hi:")
    except getopt.GetoptError:
        print(helpMsg)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpMsg)
            sys.exit()
        elif opt == "-i":
            netFileA = arg
        elif opt == "-j":
            netFileB = arg



    print ('Opening file ' + netFileA +' ...')
    f =  open(netFileA)
    linesA = f.read().splitlines()
    linesA = np.array([l.split(' ') for l in linesA]).T
    classA1 = np.unique(np.concatenate((linesA[0],linesA[1]))).tolist()

    print ('Opening file ' + netFileB +' ...')
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
        print N1sum
        print N2sum
        A = np.zeros((len(N1sum)*len(N2sum), len(N1sum)*len(N2sum)))

        for i,ni in enumerate(N1sum):
            for j,nj in enumerate(N2sum):
                for u,nu in enumerate(N1sum):
                    for v,nv in enumerate(N2sum):
                        Ai = (i*len(N1sum))+j
                        Aj = (u*len(N1sum))+v
                        print nu, nv, nu*nv
                        val = 1.0/(nu*nv) if N1[i,u] and N2[j,v] else 0
                        print val
                        A[Ai,Aj] = val
        print A
        return A

    NA = getInteractionMatrix(classA1, linesA)
    NB = getInteractionMatrix(classB1, linesB)

    Am = getAMatrix(NA,NB)
    '''
    #np.save('amat', Am)


    Am = np.load('amat.npy')
    #print np.sum(Am, axis=1)
    #print np.sum(Am, axis=0)

    w,v = linalg.eig(Am)
    print (w)
    print (v)

    #np.save('wvec', w)
    #np.save('vvec', v)

    #w = np.load('wvec.npy')
    #v = np.load('vvec.npy')
    indexes = np.argsort(abs(w))
    v = v[indexes][-1]


    Ain = np.concatenate([[i]*len(classB1) for i in range(len(classA1))])
    Bin = np.array([x for x in range(len(classB1))]*len(classA1))

    sort = np.argsort(abs(v))[::-1]


    resA = []
    resB = []
    Ain = Ain[sort]
    Bin = Bin[sort]

    print ([classA1[i] for i in Ain])
    print ([classB1[i] for i in Bin])

    while len(Ain) > 0 and len(Bin) > 0 :
        resA.append(Ain[0])
        resB.append(Bin[0])
        whA = np.where(Ain == Ain[0])[0]
        whA2 = np.where(Bin == Ain[0])[0]
        whB = np.where(Bin == Bin[0])[0]
        whB2 = np.where(Ain == Bin[0])[0]
        stwh = np.unique(np.concatenate((whA,whB,whA2,whB2)))
        Ain = np.delete(Ain, stwh)
        Bin = np.delete(Bin, stwh)

    nodesA = [classA1[x] for x in resA]
    nodesB = [classB1[x] for x in resB]

    for a,b in zip(nodesA, nodesB):
        print (a, b, str(1))

if __name__ == "__main__":
   main(sys.argv[1:])
