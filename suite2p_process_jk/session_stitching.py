# -*- coding: utf-8 -*-
"""
Make stitched data.bin and modified ops.npy files
Removing session information are at google sheets (suite2p process/Removed sessions)
After selecting sessions using 'session_selection.py'
Save to 2021 S1 S2p... HDD's (or in D:/TPM/JK/ in case of RAID tower)
Also save all ops files.
2021/09/07 JK
"""

#%% BS
removingSessions = {'025Upper': ['014', '016', '017','018','024','025','5555_001','5555_004','5555_014','5555_103','9999_1', '9999_2'],
                    '025Lower': ['011', '012', '016','025','5554_001','5554_003','5554_012','5554_013','5554_103','9998_1', '9998_2'],
                    '027Upper': [],
                    '027Lower': [],
                    '030Upper': [],
                    '030Lower': [],
                    '036Upper': [],
                    '036Lower': [],
                    '037Upper': [],
                    '037Lower': [],
                    '038Upper': [],
                    '038Lower': [],
                    '039Upper': [],
                    '039Lower': [],
                    '041Upper': [],
                    '041Lower': [],
                    '052Upper': [],
                    '052Lower': [],
                    '053Upper': [],
                    '053Lower': [],
                    '054Upper': [],
                    '054Lower': [],
                    '056Upper': [],
                    '056Lower': [],}

import numpy as np
from matplotlib import pyplot as plt
from suite2p.registration import rigid, nonrigid
import os, glob
import napari
from suite2p.io.binary import BinaryFile
from skimage import exposure
import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'
s2pDir = 'D:/TPM/JK/s2p/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]

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


#%%
mi = 0
mouse = mice[mi]
# for pn in range(1,5):
for pn in range(1,9):
    volumeName = f'{mouse:03}Upper' if pn < 5 else f'{mouse:03}Lower'
    h5planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
    s2pPlaneDir = f'{s2pDir}{mouse:03}/plane_{pn}/'
    if not os.path.exists(f'{s2pPlaneDir}plane0'):
        os.makedirs(f'{s2pPlaneDir}plane0')
    allSessionNames = get_session_names(h5planeDir, mouse, pn)
    useSessionNames = [sn for sn in allSessionNames if sn[4:] not in removingSessions[volumeName]]
    sInds = np.where(np.array([True if sn in useSessionNames else False for sn in allSessionNames]))[0]
    
    regOps = np.load(f'{h5planeDir}s2p_nr_reg.npy', allow_pickle=True).item()
    # Temporary: for first 3 mice where block sizes and maxregshiftNRs were not saved for each registration step
    if 'block_size1' not in regOps:
        regOps['block_size1'] = [128,128]
        regOps['block_size2'] = [32,32]
        regOps['maxregshiftNR1'] = 12
        regOps['maxregshiftNR2'] = 3
    # Temporary: for where session names had '_' for training sessions    
    regOps['sessionNames'] = [sn[:-1] if sn[-1]=='_' else sn for sn in regOps['sessionNames']]

    # Get each registration offsets for those in useSessionNames
    opsList = []
    writeFn = f'{s2pPlaneDir}plane0/data.bin' # plane0 is suite2p convention
    if os.path.isfile(writeFn):
        os.remove(writeFn)
    saveFn = f'{s2pPlaneDir}plane0/stitched_ops.npy' # plane0 is suite2p convention
    batchSize = 5000
    stitchedOps = {}
    stitchedOps['useSessionNames'] = useSessionNames
    stitchedOps['corrVals'] = [regOps['corrVals'][si] for si in sInds]
    stitchedOps['maxregshift'] = regOps['maxregshift']
    stitchedOps['msdVals'] = [regOps['msdVals'][si] for si in sInds]
    stitchedOps['nonrigid_offsets_y1'] = [regOps['nonrigid_offsets_1st'][0][0][si] for si in sInds]
    stitchedOps['nonrigid_offsets_x1'] = [regOps['nonrigid_offsets_1st'][0][1][si] for si in sInds]
    stitchedOps['nonrigid_offsets_y2'] = [regOps['nonrigid_offsets_2nd'][0][0][si] for si in sInds]
    stitchedOps['nonrigid_offsets_x2'] = [regOps['nonrigid_offsets_2nd'][0][1][si] for si in sInds]
    stitchedOps['rigid_offsets_y1'] = [regOps['rigid_offsets_1st'][0][0][si] for si in sInds]
    stitchedOps['rigid_offsets_x1'] = [regOps['rigid_offsets_1st'][0][1][si] for si in sInds]
    stitchedOps['rigid_offsets_y2'] = [regOps['rigid_offsets_2nd'][0][0][si] for si in sInds]
    stitchedOps['rigid_offsets_x2'] = [regOps['rigid_offsets_2nd'][0][1][si] for si in sInds]
    stitchedOps['regImgs'] = [regOps['regImgs'][si] for si in sInds]
    stitchedOps['smooth_sigma'] = regOps['smooth_sigma']
    stitchedOps['smooth_sigma_time'] = regOps['smooth_sigma_time']
    stitchedOps['snr_thresh'] = regOps['snr_thresh']
    stitchedOps['nFrames'] = []
    stitchedOps['block_size1'] = regOps['block_size1']
    stitchedOps['block_size2'] = regOps['block_size2']
    stitchedOps['maxregshiftNR1'] = regOps['maxregshiftNR1']
    stitchedOps['maxregshiftNR2'] = regOps['maxregshiftNR2']
    
    refSn = refSessions[mi]
    refSn = f'{mouse:03}_{refSn:03}'
    refSi = np.where(np.array([True if sn == refSn else False for sn in useSessionNames]))
    stitchedOps['refSi'] = refSi
    
    nSessions = len(useSessionNames)
    for i, sn in enumerate(useSessionNames):
    # for i, sn in enumerate(useSessionNames[:5]):
    # sn = useSessionNames[0]
        print(f'Processing JK{mouse:03} plane {pn} session {sn} ({i+1}/{nSessions})')
        si = np.where(np.array([tempSn == sn for tempSn in regOps['sessionNames']]))[0][0]
        (Ly, Lx) = regOps['regImgs'].shape[1:]
        rigid_y1 = regOps['rigid_offsets_1st'][0][0][si]
        rigid_x1 = regOps['rigid_offsets_1st'][0][1][si]
        nonrigid_y1 = regOps['nonrigid_offsets_1st'][0][0][si,:]
        nonrigid_x1 = regOps['nonrigid_offsets_1st'][0][1][si,:]
        rigid_y2 = regOps['rigid_offsets_2nd'][0][0][si]
        rigid_x2 = regOps['rigid_offsets_2nd'][0][1][si]
        nonrigid_y2 = regOps['nonrigid_offsets_2nd'][0][0][si,:]
        nonrigid_x2 = regOps['nonrigid_offsets_2nd'][0][1][si,:]
        # Read data.bin
        sname = sn[4:]
        readFn = f'{h5planeDir}{sname}/plane0/data.bin'
        readOps = np.load(f'{h5planeDir}{sname}/plane0/ops.npy', allow_pickle=True).item()
        opsList.append(readOps)
        stitchedOps['nFrames'].append(readOps['nframes'])
        
        with BinaryFile(Ly=Ly, Lx=Lx, read_filename=readFn, write_filename=writeFn, append=True) as f:
            for k, (_, frames) in enumerate(f.iter_frames(batch_size=batchSize)):
                # Apply 2-step registration
                # 1st rigid shift
                frames = np.roll(frames, (-rigid_y1, -rigid_x1), axis=(1,2))
                # 1st nonrigid shift
                yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=regOps['block_size1'])
                ymax1 = np.tile(nonrigid_y1, (frames.shape[0],1))
                xmax1 = np.tile(nonrigid_x1, (frames.shape[0],1))
                frames = nonrigid.transform_data(data=frames, nblocks=nblocks, 
                    xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
                # 2nd rigid shift
                frames = np.roll(frames, (-rigid_y2, -rigid_x2), axis=(1,2))
                # 2nd nonrigid shift            
                yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=regOps['block_size2'])
                ymax1 = np.tile(nonrigid_y2, (frames.shape[0],1))
                xmax1 = np.tile(nonrigid_x2, (frames.shape[0],1))
                frames = nonrigid.transform_data(data=frames, nblocks=nblocks, 
                    xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
                # Append to the writing binary file
                f.write(frames)
    stitchedOps['opsList'] = opsList
    np.save(saveFn, stitchedOps)


#%% Testing the result
# Part of '210908_data_stitching_test.py'
#(1) Testing the process itself (did it work as it was supposed to work?)

# pn = 4
# s2pPlaneDir = f'{s2pDir}{mouse:03}/plane_{pn}/'
# stitchedOpsFn = f'{s2pPlaneDir}plane0/stitched_ops.npy'
# stitchedDataFn = f'{s2pPlaneDir}plane0/data.bin'
# stitchedOps = np.load(stitchedOpsFn, allow_pickle=True).item()
# with BinaryFile(Ly=Ly, Lx=Lx, read_filename=stitchedDataFn) as f:
#     nSessions = len(stitchedOps['useSessionNames'])
#     nframes = np.array(stitchedOps['nFrames'])
#     startFrames = np.concatenate(([0],np.cumsum(nframes)[:-1]))
#     endFrames= np.cumsum(nframes)
#     meanImgs = []
#     for i in range(nSessions):
#     # for i in range(5):
#         meanImgs.append(f.ix(range(startFrames[i],endFrames[i]), is_slice=True).mean(axis=0))
        

# viewer = napari.Viewer()
# for i in range(nSessions):
# # for i in range(5):    
#     viewer.add_image(np.array([stitchedOps['regImgs'][i], meanImgs[i]]), visible=False)
   
# #%%
# #(2) Testing FOV matching again, for the last time
# napari.view_image(np.array(meanImgs))