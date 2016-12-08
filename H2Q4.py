import numpy as np
import sys, getopt
import csv, math
from scipy import linalg
import pickle
import scipy.sparse as sp
from collections import Counter

import logging


np.set_printoptions(edgeitems=500, precision=4)

def main(argv):
    helpMsg = 'H2Q2.py -i <network1> -j <network2> -t <threshold> -o <outputFile> -l'

    netFileA = 'NT2.txt'
    netFileB = 'NT1.txt'
    outputFile = 'out.txt'
    load = False
    THRESH = 0.00001

    try:
        opts, args = getopt.getopt(argv,"hi:j:t:o:l:")
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
        elif opt == "-t":
            THRESH = arg
        elif opt == "-o":
            outputFile = arg
        elif opt=='-l':
            load = True

    logging.basicConfig(filename=netFileA+netFileB+'.log',level=logging.DEBUG)

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

    def getInteractionMatrix(classes, interactions):
        NA = np.zeros((len(classes), len(classes)), dtype=int)
        for vu in interactions.T:
            i, j = classes.index(vu[0]), classes.index(vu[1])
            NA[i,j], NA[j,i]= 1,1
        return NA

    def getAMatrix(N1, N2):
        N1sum = np.sum(N1, axis=0)
        N2sum = np.sum(N2, axis=0)
        A = np.zeros((len(N1sum)*len(N2sum), len(N1sum)*len(N2sum)))
        print A.shape
        for i,ni in enumerate(N1sum):
            for j,nj in enumerate(N2sum):

                for u,nu in enumerate(N1sum):
                    for v,nv in enumerate(N2sum):

                        Ai = (i*(len(N2sum)))+j
                        Aj = (u*(len(N2sum)))+v

                        #print i,j,Ai
                        val = 1.0/(nu*nv) if N1[i,u] and N2[j,v] else 0
                        A[Ai,Aj] = val
        return A

    if not load:
        NA = getInteractionMatrix(classA1, linesA)
        NB = getInteractionMatrix(classB1, linesB)

        Am = getAMatrix(NA,NB)

        #np.save('data/amat', Am)

        #Am = np.load('data/amat.npy')
        #print np.sum(Am, axis=1)
        #print np.sum(Am, axis=0)

        w,v = linalg.eig(Am)

        #np.save('data/wvec', w)
        #np.save('data/vvec', v)

    #w = np.load('data/wvec.npy')
    #v = np.load('data/vvec.npy')
    indexes = np.argsort(abs(w))
    print w[indexes][-1]

    #v = v[indexes][-1]
    v = v[0]

    Ain = np.concatenate([[i]*len(classB1) for i in range(len(classA1))])

    Bin = np.array([x for x in range(len(classB1))]*len(classA1))


    v = abs(v)


    sort = np.argsort(v)[::-1]

    resA = []
    resB = []
    vals = []
    Ain = Ain[sort]
    Bin = Bin[sort]
    v = v[sort]
    logging.info('----------EigenVector--------')
    logging.info(str(v))

    while len(Ain) > 0 and len(Bin) > 0:
        resA.append(Ain[0])
        resB.append(Bin[0])
        vals.append(v[0])
        whA = np.where(Ain == Ain[0])[0]
        whA2 = np.where(Bin == Ain[0])[0]
        whB = np.where(Bin == Bin[0])[0]
        whB2 = np.where(Ain == Bin[0])[0]
        stwh = np.unique(np.concatenate((whA,whB,whA2,whB2)))
        Ain = np.delete(Ain, stwh)
        Bin = np.delete(Bin, stwh)
        v = np.delete(v, stwh)

    nodesA = [classA1[x] for x in resA]
    nodesB = [classB1[x] for x in resB]

    f = open(outputFile,'w')

    for a,b,val in zip(nodesA, nodesB, vals):
        print a, b, val
        if val >= THRESH:
            f.write(str(a)+' '+str(b)+'\n')
    f.close()

    import csv
    with open(netFileA+netFileB+'.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for a,b,val in zip(nodesA, nodesB, vals):
            spamwriter.writerow([a, b, vals[0]])


if __name__ == "__main__":
   main(sys.argv[1:])
