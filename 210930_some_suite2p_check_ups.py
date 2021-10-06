# -*- coding: utf-8 -*-
"""
Some suite2p process check ups
2021/09/30 JK
(1) auto_pre_cur
(2) bidirectional alignment
(3) Register_to_reference
"""

import numpy as np
from auto_pre_cur import auto_pre_cur
import napari
import matplotlib.pyplot as plt
from suite2p.io.binary import BinaryFile
from suite2p.registration import bidiphase
from suite2p.run_s2p import run_s2p, default_ops
import h5py
import time
#%% (1) auto_pre_cur
# Why are some good-looking cells still in "not cell" part?
baseDir = 'D:/TPM/JK/h5/025/plane_2/001/plane0/th040/for_test/'
stats = np.load(f'{baseDir}stat.npy', allow_pickle=True).tolist()
F = np.load(f'{baseDir}F.npy', allow_pickle=True).tolist()
Fn = np.load(f'{baseDir}Fneu.npy', allow_pickle=True).tolist()
#%% Neuropil - soma fluorescence
testci = 1229
a = np.array(Fn[testci]) - np.array(F[testci])
print(a.mean())

#%%
stats[testci]

#%%
auto_pre_cur(baseDir)


#%% (2) bidirectional alignment
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
baseDir = 'D:/TPM/JK/h5/052/plane_6/'
nr = np.load(f'{baseDir}s2p_nr_reg.npy', allow_pickle=True).item()

napari.view_image(nr['regImgs'])

#%%
testDir = 'D:/TPM/JK/h5/025/plane_3/5555_004/plane0/'
ops = np.load(f'{testDir}ops.npy', allow_pickle=True).item()
fig, ax = plt.subplots()
ax.imshow(ops['meanImg'])



#%% Test with already-registered binary file
testFn = f'{testDir}data.bin'
with BinaryFile(Lx=ops['Lx'], Ly=ops['Ly'], read_filename=testFn) as f:
    frames = f[np.linspace(0, ops['nframes'], 1 + np.minimum(ops['nimg_init'], ops['nframes']), dtype=int)[:-1]]
    
fig, ax = plt.subplots()
ax.imshow(frames.mean(axis=0))

bdphase = bidiphase.compute(frames)

#%% Test with h5 file
baseDir = 'D:/TPM/JK/h5/025/plane_3/'
h5fn = f'{baseDir}025_5555_004_plane_3.h5'
with h5py.File(h5fn, 'r') as f:
    # frinds = np.linspace(0, ops['nframes'], 1 + np.minimum(ops['nimg_init'], ops['nframes']), dtype=int)[:-1]
    frinds = np.linspace(0, ops['nframes'], 1 + 500, dtype=int)[:-1]
    framesH5 = f['data'][frinds,:,:]
fig, ax = plt.subplots()
ax.imshow(framesH5.mean(axis=0))

bdphaseH5 = bidiphase.compute(framesH5)
'''
JK025 plane_3 5555_004
bdphase = 1
bdphaseH5 = 0
    with nimg_init = 100, bdphaseH5 = -1
ops['bidiphase'] = 0
Why wasn't it captured before?
    nimg_init is important. Too high, then the initial frames get blurred

'''
#%%
plt.plot(framesH5.mean(axis=(1,2)))

#%%
sf = 'D:/TPM/JK/h5/025/plane_3/025_5555_004_plane_3.h5'
planeDir = 'D:/TPM/JK/h5/025/plane_3/'
fastDir = 'C:/JK/'

ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 100
ops['batch_size'] = 5000
ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True

db = {'h5py': sf,
    'h5py_key': ['data'],
    'data_path': [],
    'save_path0': planeDir,
    'save_folder': f'{planeDir}5555_004_test',
    'fast_disk': f'{fastDir}',
    'roidetect': False,
}
run_s2p(ops,db)


'''
Running suite2p again captures bidiphase as -1 pixels.
'''

'''
ops['nimg_init'] determines how many frames are going to be used for bidishift,
and the result depends on this number.
Is there a sweet spot for every file?
'''
#%% 
mouse = 25
pn = 3
sname = '5555_004'

baseDir = 'D:/TPM/JK/h5/'
planeDir = f'{baseDir}{mouse:03}/plane_{pn}/'
h5fn = f'{planeDir}{mouse:03}_{sname}_plane_{pn}.h5'

nimgList = np.arange(25,500,25,dtype=int)
bdphaseList = np.zeros(len(nimgList))
with h5py.File(h5fn, 'r') as f:
    nframes = f['data'].shape[0]
    for i, n in enumerate(nimgList):
        print(f'Processing nimg_init {n}')
        t1 = time.time()
        frinds = np.linspace(0, ops['nframes'], 1 + np.minimum(n, nframes), dtype=int)[:-1]
        framesH5 = f['data'][frinds,:,:]
        bdphaseList[i] = bidiphase.compute(framesH5)
        t11 = time.time()-t1
        print(f'Took {t11:.0f} s.')
    
'''    
from np.arange(25,500,25) frame nums,
results were [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,0,-1,0,-1]
300, 400, 450 were unlucky ones.
Try other files
'''


#%%
baseDir = 'D:/TPM/JK/h5/'
mice =      [25, 25, 25, 25, 25, 25, 25, 36, 36, 36, 36, 52, 52, 52]
pnList =    [2,   3,  3,  3,  5,  5,  5,  4,  4,  4,  4, 6,   6,  6]
snameList = ['5555_004',
             '5555_004',
             '5555_014',
             '014_000',
             '014_000',
             '5554_002',
             '5554_012',
             '001_000',
             '015_000',
             '018_000',
             '5555_110',
             '003_001',
             '010_000',
             '023_000']
#%%
# nimgList = np.arange(25,500,25,dtype=int)
nimgList = np.arange(50,400,50,dtype=int)
bdphaseList = np.zeros((len(mice),len(nimgList)))

nframePerSlice = 10


for i, (mouse, pn, sname) in enumerate(zip(mice, pnList, snameList)):
# for i in range(11,len(mice)):
    # mouse = mice[i]
    # pn = pnList[i]
    # sname = snameList[i]
    print(f'JK{mouse:03} plane {pn} {sname}')
    planeDir = f'{baseDir}{mouse:03}/plane_{pn}/'
    h5fn = f'{planeDir}{mouse:03}_{sname}_plane_{pn}.h5'
    with h5py.File(h5fn, 'r') as f:
        nframes = f['data'].shape[0]
        for j, n in enumerate(nimgList):
            print(f'Processing nimg_init {n}')
            t1 = time.time()
            # frinds = np.linspace(0, nframes, 1 + np.minimum(n, nframes), dtype=int)[:-1]
            # framesH5 = f['data'][frinds,:,:]
            
            # framesH5 = np.expand_dims(f['data'][frinds,:,:].mean(axis=0), axis=0)
            
            frinds = np.linspace(0, nframes-nframePerSlice, 1 + np.minimum(n, nframes-nframePerSlice), dtype=int)[:-1]
            framesH5 = np.array([f['data'][tempi:tempi+nframePerSlice].mean(axis=0) for tempi in frinds])
                       
            bdphaseList[i,j] = bidiphase.compute(framesH5)
            t11 = time.time()-t1
            print(f'Took {t11:.0f} s.')

#%%
fig, ax = plt.subplots()
im = ax.imshow(bdphaseList)
# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.colorbar(im)
plt.show()

ax.set_xticks(range(0,7,2))
ax.set_xticklabels([nimgList[i] for i in range(0,7,2)])
ax.set_xlabel('nimg_init')

#%% Using best correlated frames
bdphaseList = np.zeros(len(mice))
nframesBestCorr = 50
for i, (mouse, pn, sname) in enumerate(zip(mice, pnList, snameList)):
# for i in range(11,len(mice)):
    # mouse = mice[i]
    # pn = pnList[i]
    # sname = snameList[i]
    print(f'JK{mouse:03} plane {pn} {sname}')
    planeDir = f'{baseDir}{mouse:03}/plane_{pn}/'
    h5fn = f'{planeDir}{mouse:03}_{sname}_plane_{pn}.h5'
    with h5py.File(h5fn, 'r') as f:
        t1 = time.time()
        
        frames = np.array(f['data'])
        nimg,Ly,Lx = frames.shape
        frames = np.reshape(frames, (nimg,-1)).astype('float32')
        frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1))
        cc = np.matmul(frames, frames.T)
        ndiag = np.sqrt(np.diag(cc))
        cc = cc / np.outer(ndiag, ndiag)
        CCsort = -np.sort(-cc, axis = 1)
        bestCC = np.mean(CCsort[:, 1:nframesBestCorr], axis=1);
        imax = np.argmax(bestCC)
        indsort = np.argsort(-cc[imax, :])[0:nframesBestCorr]
        
        framesH5 = np.array([f['data'][tempIs,:,:] for tempIs in indsort])
                    
        bdphaseList[i] = bidiphase.compute(framesH5)
        t11 = time.time()-t1
        print(f'Took {t11:.0f} s.')
                    
#%%
plt.imshow(np.expand_dims(bdphaseList,axis=1))

'''
Averaging and selecting best-correlated frames does not help.
For now, I don't know how to solve this instability.
Practically, revisit sessions with noticeable bidirectional scanning offset.
'''

#%% Example image
i = 2
mouse = mice[i]
pn = pnList[i]
sname = snameList[i]

baseDir = 'D:/TPM/JK/h5/'
planeDir = f'{baseDir}{mouse:03}/plane_{pn}/'

ops = np.load(f'{planeDir}{sname}/plane0/ops.npy', allow_pickle=True).item()
plt.imshow(ops['meanImg'], cmap='gray')



#%% Using brightest frames only

bdphaseList = np.zeros(len(mice))
nframesBright = 300
for i, (mouse, pn, sname) in enumerate(zip(mice, pnList, snameList)):
# for i in range(11,len(mice)):
    # mouse = mice[i]
    # pn = pnList[i]
    # sname = snameList[i]
    print(f'JK{mouse:03} plane {pn} {sname}')
    planeDir = f'{baseDir}{mouse:03}/plane_{pn}/'
    h5fn = f'{planeDir}{mouse:03}_{sname}_plane_{pn}.h5'
    with h5py.File(h5fn, 'r') as f:
        t1 = time.time()
        
        frames = np.array(f['data'])
        brightness = frames.mean(axis=(1,2))
        sortedInds = np.argsort(brightness)
        indsort = sortedInds[-nframesBright:]
        
        framesH5 = np.array([f['data'][tempIs,:,:] for tempIs in indsort])
                    
        bdphaseList[i] = bidiphase.compute(framesH5)
        t11 = time.time()-t1
        print(f'Took {t11:.0f} s.')
#%%
fig, ax = plt.subplots()
im = ax.imshow(np.expand_dims(bdphaseList,axis=1))
# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
# plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
plt.colorbar(im)
plt.show()


'''
Continues in 211004_bidirectional_offset_fix.py
'''