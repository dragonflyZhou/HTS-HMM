import sys
import math
import update
import numpy as np

class readStates:
    def __init__(self):
        self.READ_IDLE = 0x0000
        self.READ_HEAD = 0x0001
        self.READ_HMM = 0x0002
        self.READ_VARIANCE = 0x0003
        self.READ_STATE = 0x0004
        self.READ_TRANSITION = 0x0005
        self.READ_STREAM = 0x0006
        self.READ_MIXTURE = 0x0007
        self.READ_MEAN = 0x0008

class hmmMixture:

    def __init__(self, isDiag = None):
        self.vecSize = 0
        self.meanVector = np.array([])
        self.isDiag = isDiag
        self.invCov = np.array([])
        self.gConst = 0
        self.readStates = readStates()
        self.readState = self.readStates.READ_IDLE
        
        self.trainingMean = None
        self.trainingVariance = None
    def loadMixture(self, line):
        data = line.split(" ")
        
        if self.readState == self.readStates.READ_MEAN:
            self.meanVector = np.array([float(x) for x in data])
            self.readState = self.readStates.READ_IDLE
        elif self.readState == self.readStates.READ_VARIANCE:
            #fixme: full cov
            self.invCov = np.array([1.0/float(x) for x in data])
            self.readState = self.readStates.READ_IDLE
        if data[0] == "<MEAN>" and int(data[1]) != 0: 
            self.readState = self.readStates.READ_MEAN
        elif data[0] == "<VARIANCE>" and int(data[1]) != 0: 
            self.readState = self.readStates.READ_VARIANCE
        elif data[0] == "<GCONST>":
            self.gConst = float(data[1])
            self.vecSize = self.meanVector.shape[0]
            self.trainingMean = update.meanUpdater(self.vecSize)
            # fixme: full cov
            self.trainingVariance = update.varianceUpdater(self.vecSize, 1) 
class hmmStream:

    def __init__(self, isMSD = None):
        self.numMixtures = 0
        self.mixtureList = []
        self.mixWeights = []
        self.isMSD = isMSD       
        self.vecSize = 0
        self.hmmState = None
        self.readStates = readStates()
        self.readState = self.readStates.READ_IDLE
        
        self.trainingWeights = None
    def loadStream(self, line):
        data = line.split(" ")
        if data[0] == "<NUMMIXES>":
            self.numMixtures = int(data[1])
            self.trainingWeights = update.weightUpdater(self.numMixtures)
        elif data[0] == "<MEAN>" and self.readState == self.readStates.READ_IDLE:
            self.numMixtures = 1
            self.trainingWeights = update.weightUpdater(self.numMixtures)
            self.readState = self.readStates.READ_MIXTURE
            self.mixWeights = [1]
            self.mixtureList.append(hmmMixture())
            self.mixtureList[-1].hmmStream = self
            self.mixtureList[-1].vecSize = self.vecSize
            self.mixtureList[-1].isDiag = True
        elif data[0] == "<MIXTURE>":
            self.readState = self.readStates.READ_MIXTURE
            self.mixWeights.append(float(data[2]))
            self.mixtureList.append(hmmMixture())
            self.mixtureList[-1].hmmStream = self
            self.mixtureList[-1].vecSize = self.vecSize
            self.mixtureList[-1].isDiag = True

        if self.readState == self.readStates.READ_MIXTURE:
            self.mixtureList[-1].loadMixture(line)

class hmmState:

    def __init__(self, numStreams = 0, streamList = [], hmmSingle = None):
        self.numStreams = 0         
        self.streamList = []
        self.hmmSingle = None
        self.readStates = readStates()
        self.readState = self.readStates.READ_IDLE
    def loadHMMState(self, line):
        data = line.split(" ")
        if data[0] == "<STREAM>":
            self.readState = self.readStates.READ_STREAM
            self.streamList.append(hmmStream())
            self.numStreams = self.numStreams + 1
            self.streamList[-1].hmmState = self
            self.streamList[-1].isMSD = self.hmmSingle.hmmSet.gv.streamIsMSD[int(data[1])-1]
            self.streamList[-1].vecSize = self.hmmSingle.hmmSet.gv.streamVecLen[int(data[1])-1]
        #fixme: how to add the above to init 
        if self.readState == self.readStates.READ_STREAM:            
            self.streamList[-1].loadStream(line)

class hmmSingle:

    def __init__(self, name = None):
        self.name = name
        self.numStates = -1
        self.stateList = []
        self.transitionMat = []
        self.hmmSet = None
        self.readStates = readStates()
        self.readState = self.readStates.READ_IDLE
        self.transPCnt = 0
        
        self.trainingTrans = None
    def loadHMM(self, line):
        data = line.split(" ")        
        if data[0] == "~h":
            ss = data[1]
            self.name = ss[1:-1]
        elif data[0] == "<NUMSTATES>":
            # includes non-emitting states
            self.numStates = int(data[1])
            self.trainingTrans = update.transUpdater(self.numStates)
        elif data[0] == "<STATE>":
            hmmStateObj = hmmState()
            self.stateList.append(hmmStateObj)
            self.stateList[-1].hmmSingle = self
            self.readState = self.readStates.READ_STATE
        elif data[0] == "<TRANSP>":
            self.transPCnt = int(data[1])
            self.readState = self.readStates.READ_TRANSITION

        if self.readState == self.readStates.READ_STATE:
            self.stateList[-1].loadHMMState(line)
        elif self.readState == self.readStates.READ_TRANSITION and self.transPCnt > 0 and data[0] != "<TRANSP>":
            self.transitionMat.append([float(x) for x in data])
            self.transPCnt = self.transPCnt - 1

        if self.transPCnt == 0 and self.readState == self.readStates.READ_TRANSITION:
            self.readState == self.readStates.READ_IDLE

class globeVariable:

    def __init__(self, numStream = 0, streamVecLen = [], streamIsMSD = [], \
                 covFloorMat = []):
        self.numStream = numStream
        self.streamVecLen = streamVecLen   # list[len of stream0, len of stream1...]
        self.streamIsMSD = streamIsMSD
        self.covFloorMat = covFloorMat
        self.readStates = readStates()
        self.readState = self.readStates.READ_IDLE
        self.vIdx = -1
        
    def loadGV(self, line):
        data = line.split(" ")
        if data[0] == "<STREAMINFO>":
            self.numStream = int(data[1])
            self.covFloorMat = []
            for ii in range(self.numStream):
                self.streamVecLen.append(int(data[ii+2]))
                self.covFloorMat.append([])
        elif data[0] == "<MSDINFO>":
            for ii in range(self.numStream):
                self.streamIsMSD.append(int(data[ii+2]))
        elif data[0] == "~v":
            self.readState = self.readStates.READ_VARIANCE
            ss = data[1]
            self.vIdx = int(ss[-2])-1
            ##fixme: what if two digits idx
        elif data[0] != "<VARIANCE>" and self.readState == self.readStates.READ_VARIANCE:
            self.covFloorMat[self.vIdx]=[1/float(x) for x in data]
            self.readState == self.readStates.READ_IDLE
            
class hmmSet:
     
    def __init__(self, numHMM = 0, gv = None):
        self.numHMM = numHMM 
        self.hmmList = []
        self.nameIdxDict = {}
        self.gv = gv
        self.readStates = readStates()

    def loadHMMsetFromFile(self, fileName = None):
        if fileName is None:
            return -1
        file = open(fileName, 'r')
        readState = self.readStates.READ_IDLE
        for line in file:
            line = line.replace("\n", "")
            line = line.lstrip(" ")
            line = line.rstrip() 
            data = line.split(' ')
            if data[0] == "~o":
                if readState != self.readStates.READ_IDLE:
                    print("ERROR: Can't read header, in non idle state")
                    return None
                else:
                    readState = self.readStates.READ_HEAD
                    self.gv = globeVariable()
                    self.gv.loadGV(line)
            elif data[0] == "~h":
                self.numHMM += 1
                hmmSingleObj = hmmSingle()
                self.hmmList.append(hmmSingleObj)
                self.hmmList[-1].hmmSet = self
                readState = self.readStates.READ_HMM
                ss = data[1]
                name = ss[1:-1]
                self.nameIdxDict[name] = self.numHMM - 1
            if readState == self.readStates.READ_HEAD:
                self.gv.loadGV(line)
            elif readState == self.readStates.READ_HMM:
                self.hmmList[-1].loadHMM(line)
    def writeHMMsetToFile(self, fileName = None):
        if fileName is None:
            return -1
        file = open(fileName, 'w')

        for hmmSingle in self.hmmList:
            ss = '~h "'+hmmSingle.name+'"\n'
            file.write(ss)
            file.write('<BEGINHMM>\n')
            ss = '<NUMSTATES> '+str(hmmSingle.numStates)+'\n'
            file.write(ss)
            for st in range(hmmSingle.numStates-2):
                ss = '<STATE> '+str(st+2)+'\n'
                file.write(ss)
                hmmState = hmmSingle.stateList[st]
                streamIdx = 0
                for streamCnt in range(hmmState.numStreams):
                    ss = '<STREAM> '+str(streamCnt+1)+'\n'
                    file.write(ss)
                    hmmStream = hmmState.streamList[streamCnt]
                    if hmmStream.isMSD == 1:
                        ss = '<NUMMIXES> '+str(hmmStream.numMixtures)+'\n'
                        file.write(ss)
                        ss = '<MIXTURE> '+str(1)+' '+'{0:.6e}'.format(hmmStream.mixWeights[0])+'\n'
                        file.write(ss)
                    hmmMixture = hmmStream.mixtureList[0]
                    ss = '<MEAN> '+str(hmmMixture.vecSize)+'\n'
                    file.write(ss)
                    ss=[]
                    for item in hmmMixture.meanVector:                        
                        ss.append('{0:.6e}'.format(item))
                    ss=' '.join(ss)
                    file.write(ss)
                    file.write('\n')
                    ss = '<VARIANCE> '+str(hmmMixture.vecSize)+'\n'
                    file.write(ss)
                    ss=[]
                    for item in hmmMixture.invCov:
                        ss.append('{0:.6e}'.format(1/item))
                    ss=' '.join(ss)
                    file.write(ss)
                    file.write('\n')
                    ss = '<GCONST> '+'{0:.6e}'.format(hmmMixture.gConst)+'\n'
                    file.write(ss)
                                  
                    if hmmStream.isMSD == 1:
                        ss = '<MIXTURE> '+str(2)+' '+'{0:.6e}'.format(hmmStream.mixWeights[1])+'\n'
                        file.write(ss)
                        ss = '<MEAN> 0\n'
                        file.write(ss)
                        ss = '<VARIANCE> 0\n'
                        file.write(ss)
                        ss = '<GCONST> 0.000000e+00\n'
                        file.write(ss)

            ss = '<TRANSP> '+str(hmmSingle.numStates)+'\n'
            file.write(ss)
            ss=[]
            for ll in hmmSingle.transitionMat:
                ss=[]
                for item in ll:
                    ss.append('{0:.6e}'.format(item))
                ss=' '.join(ss)
                file.write(ss)
                file.write('\n')
                
            file.write('<ENDHMM>\n')
        file.close()






        
