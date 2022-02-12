#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg') #TkAgg
#import sys
#sys.path.append('/nfs/a2/userhome/tianjinhua/workingdir/meg/mne/')
import mne
from mne.transforms import Transform
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# basic info
rootDir = '/data/home/nummag01/workingdir/eeg1/'
# subjid = 'subj022' #subjName = ['subj016']
# 'subj009','subj011','subj012','subj013','subj017','subj018','subj020','subj021','subj022','subj023'
subjList = ['subj002','subj003']
RDMtype = ['number','item area']
newSamplingRate = 500
# sessions(12) x train data x label x test data x label x tpoints
fullData=np.zeros((len(subjList),12,2,2,2,2,int(newSamplingRate*0.8)))
for i in range(len(subjList)):
    data = np.load(pj(rootDir, 'crossDecoding12x'+str(newSamplingRate)+'hz_'+ subjList[i] +'.npy'))
    fullData[i] = data

avgdata = np.average(fullData,axis=(0,1))
#plot the data
fig = plt.figure(figsize=(9, 6), dpi=100)

plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,0,0,0,:],label='Number decoding accuracy(number task)') # color='r',color='b',
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,1,0,1,:],label='Item area decoding accuracy(number task)')
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,0,1,0,:],linestyle="--",label='Number decoding accuracy(Item area task)') 
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,1,1,1,:],linestyle="--",label='Item area decoding accuracy(Item area task)')
# plt.plot(np.arange(-100,700,1000/samplingRate),correctedSig[i,:],color=color[i])  # range(-30,tps-30)
plt.xlabel('Time points(ms)')
plt.ylabel('Decoding accuracy(%)')
# 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
plt.legend()
#plt.savefig(pj(rootDir, 'GroupRSA_multifeature'+types[j]+'.png'))
plt.show()


fig = plt.figure(figsize=(9, 6), dpi=100)

plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,0,0,0,:],label='Within magnitude decoding accuracy(number task)') 
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[0,0,1,1,:],label='Cross magnitude decoding accuracy(number task)')
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,1,1,1,:],linestyle="--",label='Within magnitude decoding accuracy(Item area task)') 
plt.plot(np.arange(-100,700,1000/newSamplingRate),avgdata[1,1,0,0,:],linestyle="--",label='Cross magnitude decoding accuracy(Item area task)')
# plt.plot(np.arange(-100,700,1000/samplingRate),correctedSig[i,:],color=color[i])  # range(-30,tps-30)
plt.xlabel('Time points(ms)')
plt.ylabel('Decoding accuracy(%)')
# 'Time course of partial Spearman correlations between MEG RDMs and model RDMs(pvalue)'
plt.title('Time course of Spearman correlations between MEG RDMs and model RDMs')
plt.legend()
#plt.savefig(pj(rootDir, 'GroupRSA_multifeature'+types[j]+'.png'))
plt.show()

print('Done')
print('Done')