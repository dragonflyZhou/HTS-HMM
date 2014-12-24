
import sys
import math
import numpy as np
import functools

class segment:
    def __init__(self, hmmSingle, data = None):
        self.data = data
        self.T = data.shape[1]
        # numState * (T+1) array, stores the prob of the best path up to time t
        self.prob = np.zeros([hmmSingle.numStates-2,self.T+1]) 
        self.numStreams = hmmSingle.hmmSet.gv.numStream
        # numState * (T+1) array, stores the mixture choice of each stream, to give the best path
        # nonMSD case: streamChoice is the mixID gives max prob
        # MSD case: streamChoice = -1 represents choosing discrete space
        #              streamChoice >= 0 gives mixture ID
        self.streamChoice = np.zeros([hmmSingle.numStates-2,self.T+1, self.numStreams],dtype=int)
        # numState * (T+1) array, stores the state in t-1 that leads to the best prob
        #                         used to get the path
        self.backTrace = np.zeros([hmmSingle.numStates-2,self.T+1],dtype=int)
        # T dimensional array, the viterbi path
        self.path = np.zeros([self.T],dtype=int)  
        self.logProb = -1

    def getProb(self, t, s, hmmSingle):
        prob = 0
        streamChoice = np.zeros([self.numStreams])
        obs = self.data[:,t]
        streamLen = 0
        for streamCnt in range(self.numStreams):
            hmmStream = hmmSingle.stateList[s].streamList[streamCnt]
            #fixme: one mixture supported            
            if hmmStream.isMSD:
                #fixme: some subtle difference here, how to decide if an observation belongs to the 0-dim space?
                if sum(obs[streamLen:streamLen+hmmStream.vecSize]) == 0:
                    streamChoice[streamCnt] = -1
                    prob += math.log(hmmStream.mixWeights[1])
                else:
                    prob += math.log(hmmStream.mixWeights[0])
                    hmmMixture = hmmStream.mixtureList[0]
                    prob -= 0.5*hmmMixture.gConst
                    prob -= 0.5*sum(np.power(obs[streamLen:streamLen+hmmStream.vecSize] - hmmMixture.meanVector, 2)*hmmMixture.invCov)
                    streamChoice[streamCnt] = 0
                    
            else:
                hmmMixture = hmmStream.mixtureList[0]
                prob -= 0.5*hmmMixture.gConst
                prob -= 0.5*sum(np.power(obs[streamLen:streamLen+hmmStream.vecSize] - hmmMixture.meanVector, 2)*hmmMixture.invCov)
                streamChoice[streamCnt] = 0

            streamLen += hmmStream.vecSize

        return prob, streamChoice

    def getObs(self, t, stream, hmmSingle):
        strVecLen = hmmSingle.hmmSet.gv.streamVecLen
        if stream == 0:
            return self.data[:strVecLen[0],t]
        else:
            st = sum(strVecLen[:stream])
            return self.data[st:st+strVecLen[stream],t]

class viterbi:
    def __init__(self):
        return
    def runViterbi(self, hmmSingle, segment): 

        for t in range(segment.T+1):
            for s in range(hmmSingle.numStates - 2):
                if t == 0:
                    if hmmSingle.transitionMat[0][s+1] <= 0:
                        segment.prob[s,t] = float('-inf')
                    else:
                        p, streamChoice = segment.getProb(0,s,hmmSingle)
                        p += math.log(hmmSingle.transitionMat[0][s+1])
                        segment.prob[s,t] = p
                        segment.streamChoice[s,t,:] = streamChoice
                    segment.backTrace[s, t] = -1
                
                else:
                    maxProb = float('-inf')

                    #sPrev, s starts from 0, but transitionMat includes non-emitting states
                    for sPrev in range(hmmSingle.numStates - 2):
                        prob = segment.prob[sPrev, t-1]
                        if t == segment.T:
                            if hmmSingle.transitionMat[sPrev+1][hmmSingle.numStates-1] <= 0:
                                prob += float('-inf')
                            else:
                                prob += math.log(hmmSingle.transitionMat[sPrev+1][hmmSingle.numStates-1])
                        else:
                            if hmmSingle.transitionMat[sPrev+1][s+1] <= 0:
                                prob += float('-inf')
                            else:
                                prob += math.log(hmmSingle.transitionMat[sPrev+1][s+1])
                        if prob > maxProb:
                            maxProb = prob
                            backTrace = sPrev
                            if t == segment.T:
                                segment.logProb = maxProb
                    if t != segment.T:
                        probObs, streamChoice = segment.getProb(t,s,hmmSingle)
                        maxProb += probObs
                        segment.streamChoice[s,t,:] = streamChoice

                    if maxProb == float('-inf'):
                        segment.backTrace[s,t] = -1
                    else:
                        segment.backTrace[s,t] = backTrace
                    segment.prob[s,t] = maxProb
                    
                    if t == segment.T:
                        break

    def backTrack(self, segment):
        prevID = segment.backTrace[0,segment.T]
        for t in range(segment.T - 1,0,-1):
            segment.path[t] = prevID
            prevID = segment.backTrace[prevID][t]
        segment.path[0] = prevID
    def accumulateData(self, hmmSingle, segment):
        for t in range(segment.T):
            # transition mat
            if t == 0:
                prevStateID = 0 
            else:
                prevStateID = segment.path[t-1]+1
            stateID = segment.path[t]+1
            hmmSingle.trainingTrans.acc[prevStateID, stateID] +=1
            for s in range(hmmSingle.numStates):
                hmmSingle.trainingTrans.cnt[prevStateID][s] += 1

            if t == segment.T - 1:
                hmmSingle.trainingTrans.acc[stateID, hmmSingle.numStates-1] +=1
            # weights
            stateID = segment.path[t]
            streamChoice = segment.streamChoice[stateID,t,:]
            hmmState = hmmSingle.stateList[stateID]
            for s in range(hmmState.numStreams):
                hmmStream = hmmState.streamList[s]
                hmmStream.trainingWeights.acc[streamChoice[s]] += 1
                if streamChoice[s] >= 0: # chose a non-0-dim mixture         
                    hmmMixture=hmmStream.mixtureList[streamChoice[s]]
                    # mean
                    hmmMixture.trainingMean.cnt += 1
                    hmmMixture.trainingMean.acc += segment.getObs(t, s, hmmSingle)
                    # variance
                    hmmMixture.trainingVariance.cnt += 1                        
                    hmmMixture.trainingVariance.acc_1 += np.power(segment.getObs(t, s, hmmSingle) - hmmMixture.meanVector,2)
                       
        
    def processSegment(self, hmmSingle, segment):
        self.runViterbi(hmmSingle, segment)
        self.backTrack(segment)
        self.accumulateData(hmmSingle, segment)
    def updateModel(self, hmmSingle):
        #fixme: check gv bounds
        for ii in range(hmmSingle.numStates):
            for jj in range(hmmSingle.numStates):
                hmmSingle.transitionMat[ii][jj] = 0
                if hmmSingle.trainingTrans.cnt[ii,jj] != 0:
                    hmmSingle.transitionMat[ii][jj] = hmmSingle.trainingTrans.acc[ii,jj]/hmmSingle.trainingTrans.cnt[ii,jj]
        count = sum(hmmSingle.trainingTrans.acc[:,hmmSingle.numStates-1])
        for ii in range(hmmSingle.numStates):
            hmmSingle.transitionMat[ii][hmmSingle.numStates-1] = hmmSingle.trainingTrans.acc[ii,hmmSingle.numStates-1]/count
        for ii in range(hmmSingle.numStates):
            for jj in range(hmmSingle.numStates):
                hmmSingle.trainingTrans.cnt[ii,jj] = 0
                hmmSingle.trainingTrans.acc[ii,jj] = 0

        for hmmState in hmmSingle.stateList:
            streamIdx = 0
            for hmmStream in hmmState.streamList:
                cnt = sum(hmmStream.trainingWeights.acc)
                for m in range(hmmStream.numMixtures):
                    hmmStream.mixWeights[m] = hmmStream.trainingWeights.acc[m]/cnt
                    hmmStream.trainingWeights.acc[m] = 0
                #fixme: No sufficient data for msd, only 1 mixture case supported
                if hmmStream.isMSD:
                    if hmmStream.mixWeights[0] < 0.0001:
                        hmmStream.mixWeights[0] = 0.05
                        hmmStream.mixWeights[1] = 0.95
                    if hmmStream.mixWeights[1] < 0.0001:
                        hmmStream.mixWeights[0] = 0.95
                        hmmStream.mixWeights[1] = 0.05
                        
                    
                for hmmMixture in hmmStream.mixtureList:
                    # mean
                    if len(hmmMixture.meanVector) > 0:
                        if hmmMixture.trainingMean.cnt > 0:
                            hmmMixture.meanVector = hmmMixture.trainingMean.acc/hmmMixture.trainingMean.cnt
                            hmmMixture.trainingMean.acc.fill(0)
                            hmmMixture.trainingMean.cnt = 0
                    # variance
                        if hmmMixture.trainingVariance.cnt > 1:
                            #print(hmmMixture.trainingVariance.acc)
                            invCov = hmmMixture.trainingVariance.cnt/hmmMixture.trainingVariance.acc_1
                            if sum(hmmMixture.trainingVariance.acc_1) == 0:
                                print(invCov)
                            vaFloor = hmmSingle.hmmSet.gv.covFloorMat[streamIdx]
                            hmmMixture.invCov = np.minimum(invCov, vaFloor)
                            hmmMixture.trainingVariance.acc_1.fill(0)
                            hmmMixture.trainingVariance.cnt = 0
                    # gConst
                            tempA = functools.reduce(lambda x, y: x*y, invCov)
                            tempB = math.pow((2*3.14),hmmMixture.vecSize)
                            hmmMixture.gConst=math.log(tempB)-math.log(tempA)
                streamIdx += 1
                
                
                
            
            

