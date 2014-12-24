import random
import numpy as np
import math
import copy
import sys
sys.path.append("../code")

import hmmModel
import baum_welch
import utterance
import update
import htkmfc

hmmSetProto = hmmModel.hmmSet()
hmmSetProto.loadHMMsetFromFile("monophone.mmf.nonembedded.nonBin")

hmmSetObj = copy.deepcopy(hmmSetProto)
numStates = hmmSetObj.hmmList[0].numStates
numStreams = hmmSetObj.gv.numStream

backwardObj = baum_welch.backward()
forwardObj = baum_welch.forward()

updateObj = update.update()

segmentList = []

# train.scp is from HTS ARCTIC demo data set
# a file list containing all the data and label file names

fh = open('../../test/HTS-demo_CMU-ARCTIC-SLT/data/scp/train.scp')
fileList = fh.readlines()
for i in range(len(fileList)):
	print(fileList[i])
	data=None
	htkObj=None
	htkObj=htkmfc.openHTK(fileList[i].replace('\n',''), "r")
	data=htkObj.getall()
	data = np.transpose(data)
	labelFile=fileList[i].replace('/cmp/','/labels/mono/')
	labelFile=labelFile.replace('.cmp','.lab')
	labelFile=labelFile.replace('\n','')
	fLabel=open(labelFile, 'r')
	lines=fLabel.readlines()
	fLabel.close()
	labSeqStr=[]
	for line in lines:
		tmp=line.split()
		labSeqStr.append(tmp[2])        
	uttObj = utterance.utterance(data, labSeqStr, numStates, numStreams, 2, 10, 1500, 100, 5000, hmmSetObj)
	ret = backwardObj.runBackward(hmmSetObj, uttObj, None, None)
	if ret >= 0:
		forwardObj.runForward(hmmSetObj, uttObj, None, None)

for hmmSingle in hmmSetObj.hmmList:
	updateObj.updateModel(hmmSingle)

hmmSetObj.writeHMMsetToFile("result.txt")
