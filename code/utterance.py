import numpy as np
import math
import pdb

class utterance:
    def __init__(self, data, labSeqName, numStates, numStreams, numMixtures, pruningForward, pruningBackwardStart, pruningBackwardInc, pruningBackwardStop, hmmSet):
        self.data = data
        self.T = data.shape[1]
        # list of length Q, stores the label sequence idx to the hmmSet
        self.labSeq = self.convertLabnameToIdx(labSeqName, hmmSet)
        self.Q = len(self.labSeq)

        self.numStates = numStates
        self.numStreams = numStreams
        self.numMixtures = numMixtures

        # Q * numState * T array, stores the alpha and beta arrays  
        self.alpha = np.empty([self.T, self.Q, numStates])
        self.alpha[:] = float('-inf')
        self.beta = np.empty([self.T, self.Q, numStates])
        self.beta[:] = float('-inf')
        # Proceed only if alpha(q, s, t-1)*beta(q,s,t-1) is within a distance of Pr(O|M)
        # Find the qLo and qHi based on the above, but choose qLo to qHi+1, allowing q(Nq)->q+1(1)
        self.pruningForward = pruningForward

        # Prune Qlow and Qhi for every t, based on logP max at time t, logP max at time t label q
        # note Qlow needs to -1, to allow transition from q+1 to q
        self.pruningBackwardStart = pruningBackwardStart
        self.pruningBackwardInc = pruningBackwardInc
        self.pruningBackwardStop = pruningBackwardStop

        # array of length T, stores the starting and ending q point(idx of labSeq) for time t
        self.Qlo = np.zeros([self.T],dtype=int)
        self.Qhi = np.empty([self.T],dtype=int)
        self.Qhi[:] = self.Q - 1

        # array of T*Q*S
        self.outProb = np.empty([self.T, self.Q, numStates-2])
        self.outProb[:] = float('-inf')
        
        # array of T*Q*S*Stream*M, M is the maxim for all the streams
        self.outProbFull = np.empty([self.T, self.Q, numStates-2, numStreams, numMixtures])
        self.outProbFull[:] = float('-inf')

        # array of T*Q*S*Stream, prob of emitting obs for all the streams EXCEPT Stream s
        self.outProbCompl = np.empty([self.T, self.Q, numStates-2, numStreams])
        self.outProbCompl[:] = float('-inf')
        
        self.logProb = -1

    def populateOutProb(self, t, q, s, hmmSingle):
        if self.outProb[t,q,s] > float('-inf'):
            return
        prob = 0
        obs = self.data[:,t]
        streamLen = 0
        for streamCnt in range(self.numStreams):
            hmmStream = hmmSingle.stateList[s].streamList[streamCnt]
            #fixme: one mixture supported            
            if hmmStream.isMSD:
                #fixme: some subtle difference here, how to decide if an observation belongs to the 0-dim space?
                if sum(obs[streamLen:streamLen+hmmStream.vecSize]) < (-1e+10+1):
                    temp = math.log(hmmStream.mixWeights[1])
                    prob += temp
                    self.outProbFull[t,q,s,streamCnt,1] = temp
                else:
                    temp = math.log(hmmStream.mixWeights[0])
                    hmmMixture = hmmStream.mixtureList[0]
                    temp -= 0.5*hmmMixture.gConst
                    temp -= 0.5*sum(np.power(obs[streamLen:streamLen+hmmStream.vecSize] - hmmMixture.meanVector, 2)*hmmMixture.invCov)
                    prob += temp
                    self.outProbFull[t,q,s,streamCnt,0] = temp
                    
            else:
                hmmMixture = hmmStream.mixtureList[0]
                temp = -0.5*hmmMixture.gConst
                temp -= 0.5*sum(np.power(obs[streamLen:streamLen+hmmStream.vecSize] - hmmMixture.meanVector, 2)*hmmMixture.invCov)
                prob += temp
                self.outProbFull[t,q,s,streamCnt,0] = temp

            streamLen += hmmStream.vecSize

        self.outProb[t,q,s] = prob
        streamLen = 0
        for streamCnt in range(self.numStreams):
            hmmStream = hmmSingle.stateList[s].streamList[streamCnt]
            if hmmStream.isMSD:
                if sum(obs[streamLen:streamLen+hmmStream.vecSize]) < (-1e+10+1):
                    self.outProbCompl[t,q,s,streamCnt] = prob - self.outProbFull[t,q,s,streamCnt,1] 
                else:
                    self.outProbCompl[t,q,s,streamCnt] = prob - self.outProbFull[t,q,s,streamCnt,0] 
            else:
                self.outProbCompl[t,q,s,streamCnt] = prob - self.outProbFull[t,q,s,streamCnt,0]

            streamLen += hmmStream.vecSize

    def getObs(self, t, stream, hmmSingle):
        strVecLen = hmmSingle.hmmSet.gv.streamVecLen
        if stream == 0:
            return self.data[:strVecLen[0],t]
        else:
            st = sum(strVecLen[:stream])
            return self.data[st:st+strVecLen[stream],t]

    def convertLabnameToIdx(self, labSeqName, hmmSet):
        labSeq = []
        for name in labSeqName:
            labSeq.append(hmmSet.nameIdxDict[name])
        labSeq = np.array(labSeq)
        return labSeq
        
            

