# -*- coding: utf-8 -*-
"""
Fixing bidirectional scanning offset.

Look at individual mean images, pick those with noticeable bidirectional scanning offsets,
fix them. By trying different 'nimg_init' values.

2021/10/04 JK
"""

#%% BS
import numpy as np
from auto_pre_cur import auto_pre_cur
import napari
import matplotlib.pyplot as plt
from suite2p.io.binary import BinaryFile
from suite2p.registration import bidiphase
from suite2p.run_s2p import run_s2p, default_ops
import h5py
import time
import glob, os, shutil
import gc
gc.enable()

baseDir = 'D:/TPM/JK/h5/'
fastDir = 'C:/JK/' # This better be in SSD
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]


def get_session_names(planeDir, mouse, planeNum):
    tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{planeNum}.h5')
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
        sname = f'{mouse:03}_{sn:03}'
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

#%% Check mean images and pick horrible sessions
# In each mouse
mi = 0
mouse = mice[mi]
#%% mean images
# viewer = napari.Viewer()
bdpList = []
for pn in range(1,9):
    planeDir = f'{baseDir}{mouse:03}/plane_{pn}/'
    sessionNames = get_session_names(planeDir, mouse, pn)
    # mimgList = []
    bdpPlane = []
    for sname in sessionNames:
        sname = sname[4:]
        ops = np.load(f'{planeDir}{sname}/plane0/ops.npy', allow_pickle=True).item()
        # mimgList.append(ops['meanImg'])
        bdpPlane.append(ops['bidiphase'])
    bdpList.append(np.array(bdpPlane))
    # viewer.add_image(np.array(mimgList), visible=False, name=f'plane {pn}')


#%% In each horrible session, re-run registration with nimg_init that detects
# bidirectional offset.
# Increase by 25 from 100 until 500, until bidirecitonal offset is not 0


# JK025
# errorSessionInds = {1: [7],
#                     2: [1,24],
#                     3: [1,7,20,23],
#                     4: [1,7],
#                     5: [1,10,13,25,28,30],
#                     6: [1,2,5,13,22,25,26,28,30],
#                     7: [1,4,8,10,11,12,13,20,25,28,30,32],
#                     8: [1,16,25]}
errorSessionInds = {1: [],
                    2: [],
                    3: [7,20,23],
                    4: [1,7],
                    5: [1,10,13,25,28,30],
                    6: [1,2,5,13,22,25,26,28,30],
                    7: [1,4,8,10,11,12,13,20,25,28,30,32],
                    8: [1,16,25]}



#%% check ref images
viewer = napari.Viewer()
for pn in range(1,9):
    planeDir = f'{baseDir}{mouse:03}/plane_{pn}/'
    sessionNames = get_session_names(planeDir, mouse, pn)
    mimgList = []
    for ei in errorSessionInds[pn]:
        sname = sessionNames[ei][4:]
        ops = np.load(f'{planeDir}{sname}/plane0/ops.npy', allow_pickle=True).item()
        mimgList.append(ops['refImg'])
    viewer.add_image(np.array(mimgList), visible=False, name=f'plane {pn}')


#%%
nimginitList = np.arange(100,525,25, dtype=int)
ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
# ops['nimg_init'] = 100
ops['batch_size'] = 3000
ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True
for pn in range(1,9):
    planeDir = f'{baseDir}{mouse:03}/plane_{pn}/'
    sessionNames = get_session_names(planeDir, mouse, pn)
    h5List = glob.glob(f'{planeDir}{mouse:03}_*_plane_{pn}.h5')  
    for ei in errorSessionInds[pn]:
        foundflag = 0
        snameFull = sessionNames[ei]
        sname = snameFull[4:]
        print(f'Processing JK{mouse:03} plane {pn} session {sname}')
        
        sessionDir = f'{planeDir}{sname}/plane0/'
        prevOps = np.load(f'{sessionDir}/ops.npy', allow_pickle=True).item()
        
        tempFnList = [fn for fn in h5List if snameFull in fn]
        for nimginit in nimginitList:
            if nimginit == prevOps['nimg_init']:
                print(f'Skipping {nimginit} due to redundancy')
            else:
                print(f'nimg_init {nimginit}')
                nframeTotal = 0
                for fi, fn in enumerate(tempFnList):
                    with h5py.File(fn, 'r') as f:
                        nframe = f['data'].shape[0]
                        nframeTotal += nframe
                        if fi == 0:
                            Ly,Lx = f['data'].shape[1:]
                            data = np.zeros((nimginit, Ly, Lx), dtype=np.int16)    
                frames = np.linspace(0,nframeTotal,nimginit+1, dtype=int)[:-1]
                nframeCurr = 0
                dataCurr = 0
                for fi, fn in enumerate(tempFnList):
                    with h5py.File(fn, 'r') as f:
                        nframe = f['data'].shape[0]
                        useInd = np.where(frames < nframe)[0]
                        data[dataCurr:dataCurr+len(useInd), :, :] = \
                            f['data'][frames[useInd]-nframeCurr,:,:]
                        dataCurr += len(useInd)
                        nframeCurr += nframe
                bdphase = bidiphase.compute(data)
                if bdphase != prevOps['bidiphase']:
                    foundflag = 1
                    print(f'Found a new bidirectional offset {bdphase} at {nimginit}')
                    ops['nimg_init'] = nimginit
                    break
        
        if foundflag:
            # Before running a new suite2p, back up everything
            pathlist = os.listdir(f'{planeDir}{sname}/plane0/')
            backupDir = f'{planeDir}{sname}/plane0/backup'
            if not os.path.isdir(backupDir):
                os.mkdir(backupDir)
            
            for pathname in pathlist:
                source = os.path.join(f'{planeDir}{sname}/plane0/', pathname)
                dest = os.path.join(backupDir, pathname)
                shutil.move(source, dest)
                
            db = {'h5py': tempFnList,
                'h5py_key': ['data'],
                'data_path': [],
                'save_path0': planeDir,
                'save_folder': f'{sname}',
                'fast_disk': f'{fastDir}',
                'roidetect': False,
            }
            run_s2p(ops,db)
            rawbinFn = f'{planeDir}{sname}/plane0/data_raw.bin'
            os.remove(rawbinFn)
        else:
            print(f'Fixing offset failed for JK{mouse:03} plane {pn} {sname}.')
        


#%% Check again if the issue is fixed
viewer = napari.Viewer()
for pn in range(1,9):
    planeDir = f'{baseDir}{mouse:03}/plane_{pn}/'
    sessionNames = get_session_names(planeDir, mouse, pn)
    mimgList = []
    for ei in errorSessionInds[pn]:
        sname = sessionNames[ei][4:]
        ops = np.load(f'{planeDir}{sname}/plane0/ops.npy', allow_pickle=True).item()
        mimgList.append(ops['meanImg'])
    viewer.add_image(np.array(mimgList), visible=False, name=f'plane {pn}')

#%%
pn = 3
sname = '002'

imgs = []
ops = np.load(f'{baseDir}{mouse:03}/plane_{pn}/{sname}/plane0/backup/ops.npy', allow_pickle=True).item()
imgs.append(ops['meanImg'])
ops = np.load(f'{baseDir}{mouse:03}/plane_{pn}/{sname}/plane0/ops.npy', allow_pickle=True).item()
imgs.append(ops['meanImg'])
napari.view_image(np.array(imgs))

#%%
pn = 2
sname = '5555_103'

imgs = []
opsPrev = np.load(f'{baseDir}{mouse:03}/plane_{pn}/{sname}/plane0/backup/ops.npy', allow_pickle=True).item()
imgs.append(opsPrev['meanImg'])
opsNew = np.load(f'{baseDir}{mouse:03}/plane_{pn}/{sname}/plane0/ops.npy', allow_pickle=True).item()
imgs.append(opsNew['meanImg'])
napari.view_image(np.array(imgs))
