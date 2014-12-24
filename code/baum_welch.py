
import sys
import math
import numpy as np

import pdb

class myMath:
    def __init__(self):
        self.minLogExp = -math.log(1e+10);
        return
    def logAdd(self, x, y):
        if x == float('-inf'):
            return y
        if y == float('-inf'):
            return x
        if x < y:
            tmp = x
            x = y
            y = tmp
        diff = y-x
        if diff < self.minLogExp:
            return x
        else:
            z = math.exp(diff)
            return x+math.log(1+z)
        

class backward:
    def __init__(self):
        self.math = myMath()
        return
    def backwardInit(self, hmmSet, utterance, betaThre, trueQList, trueStateList):
        beta=utterance.beta
        maxT = float('-inf')
        maxQ = np.empty([utterance.Q])
        maxQ[:] = float('-inf')
        
        for q in range(utterance.Q-1, -1, -1):
            hmmSingle = hmmSet.hmmList[utterance.labSeq[q]]
            maxQ[q] = float('-inf')
            # Nq
            if q != utterance.Q-1:
                hmmSingleP1 = hmmSet.hmmList[utterance.labSeq[q+1]]
                if hmmSingleP1.transitionMat[0][hmmSingleP1.numStates-1] != 0: 
                    beta[utterance.T-1, q, hmmSingle.numStates-1] = math.log(hmmSingleP1.transitionMat[0][hmmSingleP1.numStates-1]) + beta[utterance.T-1, q+1, hmmSingleP1.numStates-1]
                    if beta[utterance.T-1, q, hmmSingle.numStates-1] > maxT:
                        maxT = beta[utterance.T-1, q, hmmSingle.numStates-1]
                    if beta[utterance.T-1, q, hmmSingle.numStates-1] > maxQ[q]:
                        maxQ[q] = beta[utterance.T-1, q, hmmSingle.numStates-1]
            else:
                beta[utterance.T-1, q, hmmSingle.numStates-1] = 0
                if beta[utterance.T-1, q, hmmSingle.numStates-1] > maxT:
                    maxT = beta[utterance.T-1, q, hmmSingle.numStates-1]
                if beta[utterance.T-1, q, hmmSingle.numStates-1] > maxQ[q]:
                    maxQ[q] = beta[utterance.T-1, q, hmmSingle.numStates-1]

            # 2 to Nq-1
            for s in range(hmmSingle.numStates-2, 0, -1):
                trans = hmmSingle.transitionMat[s][hmmSingle.numStates-1]
                if trans != 0:
                    beta[utterance.T-1, q, s] = math.log(trans) + beta[utterance.T-1, q, hmmSingle.numStates-1]
                    if beta[utterance.T-1, q, s] > maxT:
                        maxT = beta[utterance.T-1, q, s]
                    if beta[utterance.T-1, q, s] > maxQ[q]:
                        maxQ[q] = beta[utterance.T-1, q, s]
            #1
            prob = float('-inf')
            
            for s in range(hmmSingle.numStates-2):
                if beta[utterance.T-1, q, s+1] != float('-inf'):
                    trans = hmmSingle.transitionMat[0][s+1]
                    if trans != 0:
                        utterance.populateOutProb(utterance.T-1, q, s, hmmSingle)
                        p = math.log(trans)+utterance.outProb[utterance.T-1, q, s]+beta[utterance.T-1, q, s+1]
                        prob = self.math.logAdd(prob, p)
                        
            beta[utterance.T-1, q, 0] = prob
            if beta[utterance.T-1, q, 0] > maxT:
                maxT = beta[utterance.T-1, q, 0]
            if beta[utterance.T-1, q, 0] > maxQ[q]:
                maxQ[q] = beta[utterance.T-1, q, 0]
            #print("q=",q)
            #print(beta[utterance.T-1, q, :])            

        # pruning
        qLo = utterance.Qlo[utterance.T-1]
        while qLo <= utterance.Qhi[utterance.T-1]:
            if maxQ[qLo] + betaThre >= maxT:
                break
            qLo += 1
        utterance.Qlo[utterance.T-1] = qLo

        if qLo < 0:
            qLo = 0

        #print("t=T qLo=", utterance.Qlo[utterance.T-1], " qHi=", utterance.Qhi[utterance.T-1])

        # populate outProb, store them for forward run and parameter update
        for q in range(utterance.Qhi[utterance.T-1], utterance.Qlo[utterance.T-1]-1, -1):
            hmmSingle = hmmSet.hmmList[utterance.labSeq[q]]
            for s in range(hmmSingle.numStates-2):
                utterance.populateOutProb(utterance.T-1, q, s, hmmSingle)
        
    def backwardStep(self, hmmSet, utterance, t, betaThre, trueQList, trueStateList):
        beta = utterance.beta
        maxT = float('-inf')
        maxQ = np.empty([utterance.Q])
        maxQ[:] = float('-inf')

        qHi = utterance.Qhi[t+1]
        if qHi > utterance.Q - 1:
            qHi = utterance.Q - 1
        qLo = utterance.Qlo[t+1]-1
        if qLo < 0:
            qLo = 0
        
        for q in range(qHi, qLo - 1, -1):
            hmmSingle = hmmSet.hmmList[utterance.labSeq[q]]
            # Nq
            if q != utterance.Q-1:
                hmmSingleP1 = hmmSet.hmmList[utterance.labSeq[q+1]]
                prob = beta[t+1,q+1,0]

                if hmmSingleP1.transitionMat[0][hmmSingleP1.numStates-1] != 0 and beta[t+1, q+1, hmmSingleP1.numStates-1] != float('-inf'):
                    p = math.log(hmmSingleP1.transitionMat[0][hmmSingleP1.numStates-1]) + beta[t+1, q+1, hmmSingleP1.numStates-1]
                    prob = self.math.logAdd(prob, p)
                
                beta[t, q, hmmSingle.numStates-1] = prob
                if beta[t, q, hmmSingle.numStates-1] > maxT:
                    maxT = beta[t, q, hmmSingle.numStates-1]
                if beta[t, q, hmmSingle.numStates-1] > maxQ[q]:
                    maxQ[q] = beta[t, q, hmmSingle.numStates-1]

            # 2 to Nq-1            
            for s in range(hmmSingle.numStates-2, 0, -1):
                prob = float('-inf')
                for j in range(hmmSingle.numStates-2):
                   if hmmSingle.transitionMat[s][j+1] != 0 and beta[t+1,q,j+1] != float('-inf'):
                       utterance.populateOutProb(t+1, q, j, hmmSingle)
                       p = math.log(hmmSingle.transitionMat[s][j+1]) + beta[t+1,q,j+1] + utterance.outProb[t+1, q, j]
                       prob = self.math.logAdd(prob, p)
                if hmmSingle.transitionMat[s][hmmSingle.numStates-1] != 0 and beta[t, q,hmmSingle.numStates-1] != float('-inf'):
                    p = math.log(hmmSingle.transitionMat[s][hmmSingle.numStates-1]) + beta[t,q,hmmSingle.numStates-1]
                    prob = self.math.logAdd(prob, p)

                beta[t, q, s] = prob
                if beta[t, q, s] > maxT:
                    maxT = beta[t, q, s]
                if beta[t, q, s] > maxQ[q]:
                    maxQ[q] = beta[t, q, s]
            #1
            prob = float('-inf')
            for s in range(hmmSingle.numStates-2):
                if beta[t, q, s+1] != float('-inf'):
                    trans = hmmSingle.transitionMat[0][s+1]
                    if trans != 0:
                        utterance.populateOutProb(t, q, s, hmmSingle)
                        p = math.log(trans)+utterance.outProb[t,q,s]+beta[t, q, s+1]
                        prob = self.math.logAdd(prob, p)

            beta[t, q, 0] = prob
            if beta[t, q, 0] > maxT:
                maxT = beta[t, q, 0]
            if beta[t, q, 0] > maxQ[q]:
                maxQ[q] = beta[t, q, 0]
            #print("q=",q)
            #print(beta[t, q, :])

        # pruning
        qLo = utterance.Qlo[t]
        while qLo <= utterance.Qhi[t]:
            if maxQ[qLo] + betaThre >= maxT:
                break
            qLo += 1

        qHi = utterance.Qhi[t]
        while qHi >= utterance.Qlo[t]:
            if maxQ[qHi] + betaThre >= maxT:
                break
            qHi -= 1
        if qLo > qHi:
            print("Pruning Error")
        else:
            if qLo < 0:
                qLo = 0
            if qHi > utterance.Q - 1:
                qHi = utterance.Q - 1
            utterance.Qlo[t] = qLo
            utterance.Qhi[t] = qHi
    
        #print("t=", t, " qLo=", utterance.Qlo[t], " qHi=", utterance.Qhi[t])
        #print("t=",t, " trueQ=", trueQList[t], " trueState=", trueStateList[t])        

        # populate outProb, store them for forward run and parameter update
        for q in range(utterance.Qhi[t], utterance.Qlo[t]-1, -1):
            hmmSingle = hmmSet.hmmList[utterance.labSeq[q]]
            for s in range(hmmSingle.numStates-2):
                utterance.populateOutProb(t, q, s, hmmSingle)

    def backwardUtt(self, hmmSet, utterance, betaThre, trueQList, trueStateList):
        self.backwardInit(hmmSet, utterance, betaThre, trueQList, trueStateList)
        for t in range(utterance.T-2, -1, -1):
            self.backwardStep(hmmSet, utterance, t, betaThre, trueQList, trueStateList)

        utterance.logProb = utterance.beta[0, 0, 0]

        return utterance.beta[0, 0, 0]
    def runBackward(self, hmmSet, utterance, trueQList, trueStateList):
        thre = utterance.pruningBackwardStart
        while thre <= utterance.pruningBackwardStop:
            logProb = self.backwardUtt(hmmSet, utterance, thre, trueQList, trueStateList)
            if logProb > -0.5E10:
                break
            thre += utterance.pruningBackwardInc

        if thre > utterance.pruningBackwardStop:
            print("Error in beta pruning, no path found")
            return -1
        return 0

class forward:
    def __init__(self):
        self.math = myMath()
        return
    def getMaxTQ(self, alpha, beta, numStates):
        maxTQ = float('-inf')
        for s in range(numStates-2):
            x = alpha[s+1]+beta[s+1]
            if x > maxTQ:
                maxTQ = x

        return maxTQ
            
    def forwardInit(self, hmmSet, utterance, thre, trueQList, trueStateList):
        alpha = utterance.alpha
        beta = utterance.beta

        qHi = utterance.Qhi[0]+1
        if qHi > utterance.Q - 1:
            qHi = utterance.Q - 1
        qLo = utterance.Qlo[0]
        if qLo < 0:
            qLo = 0

        #print("t=0 qLo=",qLo," qHi=",qHi)
        for q in range(qLo, qHi + 1, 1):
            hmmSingle = hmmSet.hmmList[utterance.labSeq[q]] 
            # 1
            if q != 0:
                hmmSingleM1 = hmmSet.hmmList[utterance.labSeq[q-1]]
                if hmmSingleM1.transitionMat[0][hmmSingleM1.numStates-1] != 0: 
                    alpha[0,q,0] = math.log(hmmSingleM1.transitionMat[0][hmmSingleM1.numStates-1]) + alpha[0, q-1, 0]
            else:
                alpha[0,q,0] = 0

            # 2 to Nq-1
            for s in range(1, hmmSingle.numStates-1, 1):
                trans = hmmSingle.transitionMat[0][s]
                if trans != 0:
                    alpha[0, q, s] = math.log(trans) + alpha[0, q, 0] 
                    lBase = alpha[0, q, s] - utterance.logProb + beta[0, q, s]
                    utterance.populateOutProb(0, q, s-1, hmmSingle)
                    alpha[0, q, s] += utterance.outProb[0,q,s-1]
                    hmmState = hmmSingle.stateList[s-1]
                    for stream in range(hmmState.numStreams):
                        hmmStream = hmmState.streamList[stream]                         
                        for m in range(hmmStream.numMixtures):
                            hmmMixture = hmmStream.mixtureList[m]
                            l = lBase + utterance.outProbFull[0,q,s-1,stream,m] + utterance.outProbCompl[0,q,s-1,stream]
                            if l > float('-inf'):
                                hmmStream.trainingWeights.acc[m] += math.exp(l)
                                # mean
                                hmmMixture.trainingMean.cnt += math.exp(l)
                                hmmMixture.trainingMean.acc += math.exp(l)*utterance.getObs(0, stream, hmmSingle)
                                # variance
                                hmmMixture.trainingVariance.cnt += math.exp(l)                       
                                hmmMixture.trainingVariance.acc_2 += math.exp(l)*np.power(utterance.getObs(0, stream, hmmSingle),2)
                                hmmMixture.trainingVariance.acc_1 += math.exp(l)*utterance.getObs(0, stream, hmmSingle)
                    

            #Nq
            prob = float('-inf')
            for s in range(hmmSingle.numStates-2):
                if alpha[0, q, s+1] != float('-inf'):
                    trans = hmmSingle.transitionMat[s+1][hmmSingle.numStates-1]
                    if trans != 0:
                        prob = math.log(trans) + alpha[0, q, s+1]
                        prob = self.math.logAdd(prob, p)
            alpha[0, q, hmmSingle.numStates-1] = prob
            #print("q=",q)
            #print(alpha[0, q, :])

    def forwardStep(self, hmmSet, utterance, t, thre, trueQList, trueStateList):
        alpha = utterance.alpha
        beta = utterance.beta
        totalProb = utterance.logProb

        # pruning
        #print("t=", t-1, " qLo=", utterance.Qlo[t-1], " qHi=", utterance.Qhi[t-1])
        #print("t=",t-1, " trueQ=", trueQList[t-1], " trueState=", trueStateList[t-1])
            
        qLo = utterance.Qlo[t-1]
        while qLo <= utterance.Qhi[t-1]:
            maxTQ = self.getMaxTQ(np.squeeze(alpha[t-1,qLo,:]), np.squeeze(utterance.beta[t-1,qLo,:]), hmmSet.hmmList[utterance.labSeq[qLo]].numStates)
            if maxTQ + thre >= totalProb:
                break
            qLo += 1

        qHi = utterance.Qhi[t-1]
        while qHi >= utterance.Qlo[t-1]:
            maxTQ = self.getMaxTQ(np.squeeze(alpha[t-1,qHi,:]), np.squeeze(utterance.beta[t-1,qHi,:]), hmmSet.hmmList[utterance.labSeq[qHi]].numStates)
            if maxTQ + thre >= totalProb:
                break
            qHi -= 1
        if qLo > qHi:
            print("Pruning Error")
        else:
            if qLo < 0:
                qLo = 0
            if qHi+1 > utterance.Q-1:
                qHi = utterance.Q-2
            utterance.Qlo[t] = qLo
            utterance.Qhi[t] = qHi+1

        # populate outProb, store them for forward run and parameter update
        for q in range(utterance.Qlo[t], utterance.Qhi[t]+1, 1):
            hmmSingle = hmmSet.hmmList[utterance.labSeq[q]]
            for s in range(hmmSingle.numStates-2):
                utterance.populateOutProb(t, q, s, hmmSingle)
        #print("t=", t, " qLo=", utterance.Qlo[t], " qHi=", utterance.Qhi[t])
        for q in range(utterance.Qlo[t], utterance.Qhi[t]+1, 1):
            hmmSingle = hmmSet.hmmList[utterance.labSeq[q]]
            nSt = hmmSingle.numStates
            # 1
            if q != 0:
                hmmSingleM1 = hmmSet.hmmList[utterance.labSeq[q-1]]
                nStM1 = hmmSingleM1.numStates
                prob = alpha[t-1,q-1,nStM1-1]

                if hmmSingleM1.transitionMat[0][nStM1-1] != 0 and alpha[t-1, q-1, 0] != float('-inf'):
                    p = math.log(hmmSingleM1.transitionMat[0][nStM1-1]) + alpha[t-1, q-1, 0]
                    prob = self.math.logAdd(prob, p)
               
                alpha[t, q, 0] = prob

            # 2 to Nq-1
            for s in range(1, nSt-1, 1):
                prob = float('-inf')
                for j in range(hmmSingle.numStates-2):
                    if hmmSingle.transitionMat[j+1][s] != 0 and alpha[t-1,q,j+1] != float('-inf'):
                       p = math.log(hmmSingle.transitionMat[j+1][s]) + alpha[t-1,q,j+1]
                       prob = self.math.logAdd(prob, p)
                if hmmSingle.transitionMat[0][s] != 0 and alpha[t,q,0] != float('-inf'):
                    p = math.log(hmmSingle.transitionMat[0][s]) + alpha[t,q,0]
                    prob = self.math.logAdd(prob, p)

                if prob != float('-inf'):
                    alpha[t, q, s] = prob + utterance.outProb[t, q, s-1]
                    lBase = prob + beta[t,q,s] - utterance.logProb
                    hmmState = hmmSingle.stateList[s-1]
                    for stream in range(hmmState.numStreams):
                        hmmStream = hmmState.streamList[stream]
                        
                        for m in range(hmmStream.numMixtures):
                            hmmMixture = hmmStream.mixtureList[m]
                            l = lBase + utterance.outProbFull[t,q,s-1,stream,m] + utterance.outProbCompl[t,q,s-1,stream]
                            if l > float('-inf'):
                                hmmStream.trainingWeights.acc[m] += math.exp(l)
                                # mean
                                hmmMixture.trainingMean.cnt += math.exp(l)
                                hmmMixture.trainingMean.acc += math.exp(l)*utterance.getObs(t, stream, hmmSingle)
                                # variance
                                hmmMixture.trainingVariance.cnt += math.exp(l)                       
                                #hmmMixture.trainingVariance.acc += math.exp(l)*np.power(utterance.getObs(t, stream, hmmSingle) - hmmMixture.meanVector,2)
                                hmmMixture.trainingVariance.acc_2 += math.exp(l)*np.power(utterance.getObs(t, stream, hmmSingle),2)
                                hmmMixture.trainingVariance.acc_1 += math.exp(l)*utterance.getObs(t, stream, hmmSingle)
            #Nq
            prob = float('-inf')
            for s in range(hmmSingle.numStates-2):
                if alpha[t, q, s+1] != float('-inf'):
                    trans = hmmSingle.transitionMat[s+1][nSt-1]
                    if trans != 0:
                        p = math.log(trans) + alpha[t, q, s+1]
                        prob = self.math.logAdd(prob, p)

            alpha[t, q, nSt-1] = prob
            #print("q=",q, "alpha:")
            #print(alpha[t, q, :])

    def accumulate_trans(self, hmmSet, utterance, t):
        alpha = utterance.alpha
        beta = utterance.beta

        for q in range(utterance.Qlo[t], utterance.Qhi[t]+1, 1):
            hmmSingle = hmmSet.hmmList[utterance.labSeq[q]]
            nSt = hmmSingle.numStates
            if alpha[t, q, 0] != float('-inf') and beta[t, q, 0] != float('-inf'):
                if q < utterance.Q-1:
                    p1 = alpha[t, q, 0]+beta[t, q, 0]
                    if hmmSingle.transitionMat[0][nSt-1] > 0:
                        p2 = alpha[t, q, 0]+beta[t, q+1, 0] + math.log(hmmSingle.transitionMat[0][nSt-1])
                        temp = self.math.logAdd(p1, p2)
                        p1 = temp
                    p1 -= utterance.logProb
                    temp = math.exp(p1)
                else:
                    temp = math.exp(alpha[t, q, 0]+beta[t, q, 0]-utterance.logProb)

                for s in range(1, hmmSingle.numStates-1, 1):
                    if hmmSingle.transitionMat[0][s] > 0:
                        pp = alpha[t,q,0]+math.log(hmmSingle.transitionMat[0][s])+utterance.outProb[t,q,s-1]+beta[t,q,s]-utterance.logProb
                        if pp != float('-inf'):
                            hmmSingle.trainingTrans.acc[0,s] += math.exp(pp)
                    hmmSingle.trainingTrans.cnt[0,s] += temp

                if hmmSingle.transitionMat[0][hmmSingle.numStates-1] > 0:
                    pp = alpha[t,q,0]+math.log(hmmSingle.transitionMat[0][hmmSingle.numStates-1])+beta[t,q+1,0]-utterance.logProb
                    if pp != float('-inf'):
                        hmmSingle.trainingTrans.acc[0,hmmSingle.numStates-1] += math.exp(pp)
                hmmSingle.trainingTrans.cnt[0,hmmSingle.numStates-1] += temp

            
            for s in range(1, hmmSingle.numStates-1, 1):
                if alpha[t, q, s] != float('-inf') and beta[t, q, s] != float('-inf'): 
                    temp = math.exp(alpha[t, q, s]+beta[t, q, s]-utterance.logProb)
                    if hmmSingle.transitionMat[s][hmmSingle.numStates-1] > 0:
                        pp = alpha[t, q, s]+math.log(hmmSingle.transitionMat[s][hmmSingle.numStates-1])+beta[t, q, hmmSingle.numStates-1]-utterance.logProb
                        if pp != float('-inf'):
                            hmmSingle.trainingTrans.acc[s,hmmSingle.numStates-1] += math.exp(pp) 
                    hmmSingle.trainingTrans.cnt[s,hmmSingle.numStates-1] += temp
                    if t < utterance.T-1:
                        for j in range(1, hmmSingle.numStates-1, 1):
                            if hmmSingle.transitionMat[s][j] > 0:
                                logProb = alpha[t, q, s]+math.log(hmmSingle.transitionMat[s][j])
                                logProb += utterance.outProb[t+1,q,j-1]+beta[t+1, q, j]-utterance.logProb
                                if logProb != float('-inf'):
                                    hmmSingle.trainingTrans.acc[s,j] += math.exp(logProb)
                            hmmSingle.trainingTrans.cnt[s,j] += temp
                            #pdb.set_trace()
                            

    def runForward(self, hmmSet, utterance, trueQList, trueStateList):

        self.forwardInit(hmmSet, utterance, utterance.pruningForward, trueQList, trueStateList)
        for t in range(1, utterance.T, 1):
            self.forwardStep(hmmSet, utterance, t, utterance.pruningForward, trueQList, trueStateList)
            self.accumulate_trans(hmmSet, utterance, t)
            

