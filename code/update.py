import numpy as np
import functools
import math

import pdb

class meanUpdater:
    def __init__(self, vecSize):
        self.acc =  np.zeros([vecSize])
        self.cnt =  0

class varianceUpdater:
    def __init__(self, vecSize, isDiag):
        if isDiag:
            self.acc_1 =np.zeros([vecSize])
            self.acc_2 =np.zeros([vecSize])
        else:
            self.acc_1 =np.zeros([vecSize, vecSize])
            self.acc_2 =np.zeros([vecSize, vecSize])
        self.cnt = 0

class transUpdater:
    def __init__(self, numStates):
        #zero cnt? 
        self.acc = np.zeros([numStates, numStates])
        self.cnt = np.zeros([numStates, numStates])

class weightUpdater:  # for both mixtures and msd weights
    def __init__(self, numMixtures):
        self.acc = np.zeros([numMixtures])

class update:
    def __init__(self):
        return
    def updateModel(self, hmmSingle):
        #fixme: check gv bounds
        for ii in range(hmmSingle.numStates):
            for jj in range(hmmSingle.numStates):
                if hmmSingle.trainingTrans.cnt[ii,jj] != 0:
                    hmmSingle.transitionMat[ii][jj] = hmmSingle.trainingTrans.acc[ii,jj]/hmmSingle.trainingTrans.cnt[ii,jj]
        for ii in range(hmmSingle.numStates):
            for jj in range(hmmSingle.numStates):
                if hmmSingle.trainingTrans.cnt[ii,jj] != 0:
                    hmmSingle.trainingTrans.cnt[ii,jj] = 0
                    hmmSingle.trainingTrans.acc[ii,jj] = 0

        for hmmState in hmmSingle.stateList:
            streamIdx = 0
            for hmmStream in hmmState.streamList:
                cnt = sum(hmmStream.trainingWeights.acc)
                if cnt > 0:
                    for m in range(hmmStream.numMixtures):
                        hmmStream.mixWeights[m] = hmmStream.trainingWeights.acc[m]/cnt
                        hmmStream.trainingWeights.acc[m] = 0
                    #fixme: No sufficient data for msd, only 1 mixture case supported
                    #fixme: make the lower bound configurable
                    if hmmStream.isMSD:
                        if hmmStream.mixWeights[0] < 0.05:
                            hmmStream.mixWeights[0] = 0.05
                            hmmStream.mixWeights[1] = 0.95
                        if hmmStream.mixWeights[1] < 0.05:
                            hmmStream.mixWeights[0] = 0.95
                            hmmStream.mixWeights[1] = 0.05
                        
                    
                for hmmMixture in hmmStream.mixtureList:
                    # mean
                    if len(hmmMixture.meanVector) > 0:
                        if hmmMixture.trainingMean.cnt > 1E-5:
                            hmmMixture.meanVector = hmmMixture.trainingMean.acc/hmmMixture.trainingMean.cnt
                            hmmMixture.trainingMean.acc[:] = 0
                            hmmMixture.trainingMean.cnt = 0
                    # variance
                        if hmmMixture.trainingVariance.cnt > 1E-5:
                            #print(hmmMixture.trainingVariance.acc)                            
                            covariance = hmmMixture.trainingVariance.acc_2 - 2*hmmMixture.meanVector*hmmMixture.trainingVariance.acc_1 + hmmMixture.trainingVariance.cnt*np.power(hmmMixture.meanVector,2)
                            if sum(covariance) == 0:
                                pdb.set_trace()
                                print(covariance)
                            invCov = hmmMixture.trainingVariance.cnt/covariance
                            vaFloor = hmmSingle.hmmSet.gv.covFloorMat[streamIdx]
                            for i in range(len(invCov)):
                                #fixme: some negative value due to inprecision
                                if invCov[i] < 0:
                                    invCov[i] = vaFloor[i]
                                else:
                                    invCov[i] = min(invCov[i], vaFloor[i])
                            hmmMixture.invCov = invCov
                            hmmMixture.trainingVariance.acc_1[:] = 0
                            hmmMixture.trainingVariance.acc_2[:] = 0
                            hmmMixture.trainingVariance.cnt = 0
                    # gConst
                            tempA = [math.log(x) for x in invCov] 
                            tempB = hmmMixture.vecSize*math.log(2*3.14)
                            hmmMixture.gConst=tempB-sum(tempA)
                streamIdx += 1
