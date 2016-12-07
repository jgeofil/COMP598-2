import numpy as np
import sys, getopt
import csv, math
import matplotlib.pyplot as plt


PROP = {'A':0,'L':0.21,'R':0.21,'M':0.24,'K':0.26,'Q':0.39,'E':0.40,
        'I':0.41,'W':0.49,'S':0.50,'Y':0.53,'F':0.54,'V':0.61,
        'H':0.61,'N':0.65,'T':0.66,'C':0.68,'D':0.69,'G':1, 'P':3.01}


def main(argv):
    helpMsg = 'H2Q2.py -i <inputFile> -s <secondaryStructureFile>'
    inputfile = 'hw3_proteins.fa'

    inputfile = '1MBN_A.fasta'
    secondaryfile = '1MBN_SS.fasta'
    L = 0.5
    W = 4

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
            inputfile = arg
        elif opt == "-s":
            secondaryfile = arg


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

    def getSS(subseq,L):
        return 'C' if getAvgProp(subseq) >= L else 'H'

    def getProp(seq,L):
        return ''.join([getSS(seq[i-W if i>W else 0:i+W+1 if i<len(seq)-W+1 else len(seq)], L) for i in range(len(seq))])

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

    def calculate(p, pro=''):
        print '----------------------True secondary structure----------------------'
        print ss
        print sq
        print '-------------------Predicted secondary structure--------------------'
        pred = getProp(sq, 0.5)
        print pred
        print sq

        tpc = []
        fpc = []
        for i in np.arange(0,p,0.001):
            L = i
            pred = getProp(sq,L)
            tp, fp = getRates(pred, ss)
            tpc.append(tp)
            fpc.append(fp)

        gini = []
        for i in range(1, len(tpc)):
            gini.append((fpc[i]-fpc[i-1])*(tpc[i]+tpc[i-1]))
        gini = sum(gini)/float(2)

        print '---------------------------------AUC---------------------------------'
        print gini

        plt.plot(fpc, tpc)
        plt.title('ROC curve calculated by varying the threshold $\lambda$\n with a window size of '+str(W)+ pro)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.plot([0,1], [0,1], 'r--')
        plt.text(0.1, 0.8, 'AUC = '+str(gini), fontsize=15)

        plt.show()

    print '####################################################################'
    print 'Running with window = 0'
    W = 0
    calculate(1.1, pro=' and Proline = 3.01')
    print '####################################################################'
    print 'Running with window = 0 and Proline = 1'
    W = 0
    calculate(4, pro=' and Proline = 1')
    print '####################################################################'
    print 'Running with window = 4'
    W = 4
    calculate(1.1)

if __name__ == "__main__":
   main(sys.argv[1:])
