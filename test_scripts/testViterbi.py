import random
import numpy as np
import math
import copy

import sys
sys.path.append("../code")
import hmmModel
import viterbi

class trueID:
    def __init__(self):
        self.state = -1  # starting from 0, doesn't include non-emiting states
        self.isMSD = []
        self.mixID = []  # starting from 0

hmmSetObj = hmmModel.hmmSet()
hmmSetObj.loadHMMsetFromFile("test.mmf")

T = 100
# Suppose we have 10 states, stored in the lab list
lab = [random.randint(1,hmmSetObj.numHMM) for x in range(10)]

r = [random.randint(1,10) for x in range(10)]
temp = sum(r)
r = [x/temp for x in r]
prevT = 0
labTime = [] #(start, end) tuple list
for ratio in r:
    labTime.append([prevT+1, prevT+int(ratio*T)])
    prevT += int(ratio*T)

labTime[-1][1] = T

obs = np.zeros([sum(hmmSetObj.gv.streamVecLen),T])
trueIDList = []
trueLogProb = []
idx = 0
for i in range(10):
    hmmSingle = hmmSetObj.hmmList[lab[i]-1]
    logProb = 0
    idList = []
    #print(i)
    for t in range(labTime[i][1]-labTime[i][0]+1):
        #print(t)
        if t == 0:
            p = hmmSingle.transitionMat[0]
        else:
            p = hmmSingle.transitionMat[idList[-1].state+1]
        thisID = trueID()
        temp = np.random.multinomial(1,p)
        if int(np.nonzero(temp)[0]) == hmmSingle.numStates - 1:
            thisID.state = np.argmax(p) - 1
        else:
            thisID.state = int(np.nonzero(temp)[0]) - 1
        p = p[1:-1]
        logProb += math.log(p[thisID.state])

        if t == labTime[i][1]-labTime[i][0] :
            logProb += math.log(hmmSingle.transitionMat[thisID.state+1][hmmSingle.numStates-1])

        hmmState = hmmSingle.stateList[thisID.state]
        streamLen = 0
        
        for hmmStream in hmmState.streamList:
            if hmmStream.isMSD:
                thisID.isMSD.append(1) 
            else:
                thisID.isMSD.append(0) 
            temp = np.random.multinomial(1,hmmStream.mixWeights)
            thisID.mixID.append(int(np.nonzero(temp)[0]))
            logProb += math.log(hmmStream.mixWeights[thisID.mixID[-1]])
            hmmMixture = hmmStream.mixtureList[thisID.mixID[-1]]
            if len(hmmMixture.meanVector) != 0:
                obs[streamLen:streamLen+hmmStream.vecSize, idx] = \
                    np.squeeze(np.random.multivariate_normal(hmmMixture.meanVector,np.diag([1.0/x for x in hmmMixture.invCov]),1))
                logProb -= 0.5*hmmMixture.gConst
                logProb -= 0.5*sum(np.power(obs[streamLen:streamLen+hmmStream.vecSize,idx] - hmmMixture.meanVector, 2)*hmmMixture.invCov)

                
            streamLen += hmmStream.vecSize

        #print(obs[:, idx])
        #print(logProb)         
        idList.append(thisID)
        idx += 1       

    trueIDList.append(idList)
    trueLogProb.append(logProb)

globalMean = np.mean(obs, axis=1)
globalVariance = np.var(obs, axis=1)

hmmSetInit = hmmModel.hmmSet()
hmmSetInit.loadHMMsetFromFile("test.mmf")
viterbiObj = viterbi.viterbi()
print("trueLogProb:")
print(trueLogProb)
print("sum of trueLogProb")
print(sum(trueLogProb))
for iteration in range(1):
    print("iteration ", iteration)
    sumProb = 0    
    for i in range(10):
        hmmSingle = hmmSetInit.hmmList[lab[i]-1]
        data = obs[:,labTime[i][0]-1:labTime[i][1]]
        segment = viterbi.segment(hmmSingle, data)
        print(i, "label =", lab[i], "logProb =", segment.logProb, "trueLogProb =", trueLogProb[i])
        print("true ID list:")
        for trueID in trueIDList[i]:
            print(trueID.state)
            print(trueID.mixID)
        viterbiObj.processSegment(hmmSingle, segment)

        print("viterbi path:")
        print(segment.path)
        print(i, segment.logProb)
        sumProb += segment.logProb

    print("sum Prob: ", sumProb)
    
    for hmmSingle in hmmSetInit.hmmList:
        
        viterbiObj.updateModel(hmmSingle)
