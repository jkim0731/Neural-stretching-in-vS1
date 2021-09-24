# -*- coding: utf-8 -*-
"""
Show whole session frame-to-frame registration results.
That it does not work well.
Compare between mimg from each session and that after whole session registration.

Use results from 210802_nonrigid_registration.py (or register_to_reference.py) and 
regResult.npy (from session_selection.py which would be modified soon)

2021/08/25 JK
"""



#%% BS
import numpy as np
from matplotlib import pyplot as plt
from suite2p.registration import rigid, nonrigid, utils
import os, glob
import napari
from suite2p.io.binary import BinaryFile
import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'

mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]

def get_session_names(baseDir, mouse, planeNum):
    tempFnList = glob.glob(f'{baseDir}{mouse:03}_*_plane_{planeNum}.h5')
    tempFnList = [fn for fn in tempFnList if 'x' not in fn] # Treatment for h5 files with 'x' (testing blank edge removal)
    midNum = np.array([int(fn.split('\\')[1].split('_')[1]) for fn in tempFnList])
    trialNum = np.array([int(fn.split('\\')[1].split('_')[2][0]) for fn in tempFnList])
    spontSi = np.where( (midNum>5000) & (midNum<6000) )[0]
    piezoSi = np.where(midNum>9000)[0]
    
    if np.any(spontSi): 
        spontTrialNum = np.unique(trialNum[spontSi]) # used only for mouse > 50
    
    if np.any(piezoSi):
        piezoTrialNum = np.unique(trialNum[piezoSi])
    
    sessionNum = np.unique(midNum)
    regularSni = np.where(sessionNum < 1000)[0]
    
    sessionNames = []
    for sni in regularSni:
        sn = sessionNum[sni]
        sname = f'{mouse:03}_{sn:03}_'
        sessionNames.append(sname)
    if mouse < 50:
        for si in spontSi:
            sessionNames.append(tempFnList[si].split('\\')[1].split('.h5')[0][:-8])
    else:
        for stn in spontTrialNum:
            sn = midNum[spontSi[0]]
            sname = f'{mouse:03}_{sn}_{stn}'
            sessionNames.append(sname)
    for ptn in piezoTrialNum:
        sn = midNum[piezoSi[0]]
        sname = f'{mouse:03}_{sn}_{ptn}'
        sessionNames.append(sname)
    return sessionNames




#%%
mi = 0
pn = 3

mouse = mice[mi]
refSession = refSessions[mi]

planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
result = np.load(f'{planeDir}regResult.npy', allow_pickle = True).item()

napari.view_image(result['meanImg'])


#%%
sn=22
ops = np.load(f'{planeDir}{sn:03}/plane0/ops.npy', allow_pickle = True).item()
napari.view_image(ops['meanImg'])


#%% after nonrigid registration
reg = np.load(f'{planeDir}s2p_nr_reg.npy', allow_pickle=True).item()
napari.view_image(reg['regImgs'])
