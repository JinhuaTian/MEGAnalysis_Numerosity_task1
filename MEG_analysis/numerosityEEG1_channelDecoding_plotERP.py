#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:13:38 2021

@author: tianjinhua
"""

'''
# plot psd
# select the largest amplitude according to psd plot
'''
from unittest import result
import numpy as np
import os
from os.path import join as pj
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import time
# import matplotlib
# matplotlib.use('Qt5Agg') #TkAgg
#from libsvm import svmutil as sv # pip install -U libsvm-official
#import sys
#sys.path.append('/nfs/a2/userhome/tianjinhua/workingdir/meg/mne/')
import mne
from mne.transforms import Transform
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from numba import jit
jit(nopython=True,parallel=True) #nopython=True,parallel=True

from functools import reduce
from operator import add

# basic info
rootDir = '/data/home/nummag01/workingdir/eeg1/'
# subjid = 'subj022' #subjName = ['subj016']
subjList = ['subj002','subj003']
#eventMatrix = np.loadtxt('/data/home/nummag01/workingdir/eeg1/stimuli/ModelRDM_NumIsTfaDenLLF.npy',encoding='unicode_escape',dtype=int)
eventMatrix = np.loadtxt('/data/home/nummag01/workingdir/eeg1/STI2.txt')
newSamplingRate = 500
repeat = 100
kfold = 3
labelNum = 45
tpoints = int(newSamplingRate*0.8) #-0.1~0.7ms
session = 12
decodingNum = 2 #'num', 'ia'

def transLabel(label):
    label = np.array(label)
    label0,label1 = np.zeros(label.shape[0],dtype=int),np.zeros(label.shape[0],dtype=int)
    for i in range(labelNum):
        label0[label == i] = eventMatrix[:,0][eventMatrix[:,2]==i]
        label1[label == i] = eventMatrix[:,1][eventMatrix[:,2]==i]
    return label0,label1

def flattenList(listlist):
    resultList = []
    for childList in listlist:
        resultList = resultList+childList.tolist() # transfer numpy to list
    return resultList

for subjid in subjList:
    # compute MEG RDM using pairwise SVM classification
    print('subject ' + subjid +' is running.')
    savePath = pj(rootDir, subjid, 'preprocessed')
    
    # epoch and label slice
    epochList1,labelList1,epochList2,labelList2=[],[],[],[] # data slice and corresponding label
    
    epochCount1,epochCount2 = 0,0
    # load data
    epochs_list1, epochs_list2 = [],[] # real data
    # walk through subj path, concatenate single subject's data to one file
    # search for epochList(epoch number) and labelList (label number)
    for sess in range(session):
        fifpath1 = pj(savePath, 'num'+str(sess)+'.fif')
        epoch1 = mne.read_epochs(fifpath1, preload=True, verbose=True)

        fifpath2 = pj(savePath, 'ia'+str(sess)+'.fif')
        epoch2 = mne.read_epochs(fifpath2, preload=True, verbose=True)

        # select label array
        labelList1.append(epoch1.events[:, 2])        
        epochData1 = epoch1.get_data(picks = 'eeg')

        labelList2.append(epoch2.events[:, 2])        
        epochData2 = epoch2.get_data(picks = 'eeg')
        
        # select label, epoch number  
        nEpochs1, nChan, nTime = epochData1.shape
        nEpochs2, nChan, nTime = epochData2.shape

        epochList1.append(list(range(epochCount1,epochCount1+nEpochs1)))
        epochList2.append(list(range(epochCount2,epochCount2+nEpochs2)))
        
        epochCount1 = epochCount1 + nEpochs1
        epochCount2 = epochCount2 + nEpochs2
        
        epochs_list1.append(epoch1)
        epochs_list2.append(epoch2)
        del epoch1, epochData1,epoch2, epochData2
epochs_all1 = mne.concatenate_epochs(epochs_list1)
epochs_all2 = mne.concatenate_epochs(epochs_list2)

tmin = -0.1
tmax = 0.5
epochs_all1.info['bads'].extend(['CB1','CB2'])
# epochs_all1.plot_psd()
Evoked_all1=epochs_all1.average().crop(tmin=tmin,tmax=tmax,include_tmax=True)
Evoked_all1.plot()
dd1 = Evoked_all1.plot_topomap()
dd1.savefig('toponum.png')

epochs_all2.info['bads'].extend(['CB1','CB2'])
Evoked_all2=epochs_all2.average().crop(tmin=tmin,tmax=tmax,include_tmax=True)
Evoked_all2.plot()
dd2 = Evoked_all2.plot_topomap()
dd2.savefig('topois.png')

'''
epochs_all1.plot_sensors(show_names=True)
fig = epochs_all1.plot_sensors('3d')
# plot sensor location 62 channels # "FCZ"
Sixty_montage = mne.channels.make_standard_montage('standard_1020')
Sixty_montage.plot(kind='topomap', show_names=True)
'''
print('All Done')