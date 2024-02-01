# -*- coding: utf-8 -*-
"""
Check the result of each session registration
Compare with the reference session mean image
Try nonrigid registration to the reference session
    and see if it degrades mean image quality
    If not, then try applying this transform to each frames
2021/07/16 JK
"""

import napari, glob
import numpy as np
from suite2p.registration import nonrigid
import h5py
from suite2p.run_s2p import run_s2p, default_ops
import os, shutil
from suite2p.io.binary import BinaryFile
from matplotlib import pyplot as plt

h5Dir = 'D:/TPM/JK/h5/'
fastDir = 'C:/JK/' # This better be in SSD
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]

mi = 0
mouse = mice[mi]
pn = 1

planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{pn}.h5')    
fnames = [fn.split('\\')[1].split('.h5')[0] for fn in tempFnList]
midNum = np.array([int(fn.split('\\')[1].split('_')[1]) for fn in tempFnList])
trialNum = np.array([int(fn.split('\\')[1].split('_')[2][0]) for fn in tempFnList])
regularSi = np.where(midNum<1000)[0]
spontSi = np.where( (midNum>5000) & (midNum<6000) )[0]
piezoSi = np.where(midNum>9000)[0]

if np.any(spontSi): 
    spontTrialNum = np.unique(trialNum[spontSi]) # used only for mouse > 50

if np.any(piezoSi):
    piezoTrialNum = np.unique(trialNum[piezoSi])

sessionNum = np.unique(midNum)
regularSni = np.where(sessionNum < 1000)[0]

sessionNames = []
sessionFiles = []

for sni in regularSni:
    sn = sessionNum[sni]
    sname = f'{mouse:03}_{sn:03}_'
    sessionNames.append(sname)
    sessionFiles.append([fn for fn in tempFnList if sname in fn])
if mouse < 50:
    for si in spontSi:
        sessionNames.append(tempFnList[si].split('\\')[1].split('.h5')[0][:-8])
        sessionFiles.append([tempFnList[si]])
else:
    for stn in spontTrialNum:
        sn = midNum[spontSi[0]]
        sname = f'{mouse:03}_{sn}_{stn}'
        sessionNames.append(sname)
        sessionFiles.append([fn for fn in tempFnList if sname in fn])
for ptn in piezoTrialNum:
    sn = midNum[piezoSi[0]]
    sname = f'{mouse:03}_{sn}_{ptn}'
    sessionNames.append(sname)
    sessionFiles.append([fn for fn in tempFnList if sname in fn])

sname = []
for i, (sn, sf) in enumerate(zip(sessionNames, sessionFiles)):
    midNum = sn.split('_')[1]
    if int(midNum) < 1000:
        sname.append(midNum)
        if int(midNum) == refSessions[mi]:
            refSi = i
            opsFn = f'{planeDir}{midNum}/plane0/ops.npy'
            ops = np.load(opsFn, allow_pickle=True).item()
            refImg = np.uint16(ops['meanImg'])
    else: # spont and piezo sessions
        trialNum = sn.split('_')[2]
        sname.append(f'{midNum}_{trialNum}')

numSessions = len(sname)
meanImgs = np.zeros((numSessions, ops['Ly'], ops['Lx']))
for i, sn in enumerate(sname):
    opsFn = f'{planeDir}{sn}/plane0/ops.npy'
    ops = np.load(opsFn, allow_pickle=True).item()
    meanImgs[i,:,:] = ops['meanImg']

#%% Visual inspection using napari
napari.view_image(meanImgs)


#%% Right mean images as a .h5 file and do nonrigid registration using suite2p
# See the result, and compare the level of blurring with single-frame registration
wfn = f'{planeDir}eachSessionMmimgRegTest.h5'
# with h5py.File(wfn, 'w') as wf:
    # wf.create_dataset('data', data=meanImgs, dtype='uint16')

#%%
ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = False
ops['nimg_init'] = 100
ops['batch_size'] = 5000
ops['two_step_registration'] = False
ops['keep_movie_raw'] = False
ops['smooth_sigma_time'] = 0
ops['move_bin'] = True
ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']
    
ops['maxregshift'] = 0.3
# ops['maxregshiftNR'] = 20
ops['smooth_sigma'] = 0.7

ops['force_refImg'] = True
ops['refImg'] = refImg.astype(np.uint16)
ops['do_registration']=2


bslist = [64,128,160]


for bs in bslist:
    ops['block_size'] = [bs, bs]
    saveFolderName = f'{planeDir}bstest{bs}/'
    if os.path.isdir(saveFolderName):
        shutil.rmtree(saveFolderName)
    # checkOpsFn = f'{saveFolderName}plane0/ops.npy'
    # checkBinFn = f'{saveFolderName}plane0/data.bin'
    
    db = {'h5py': wfn,
        'h5py_key': ['data'],
        'data_path': [],
        'save_path0': planeDir,
        'save_folder': f'{saveFolderName}',
        'fast_disk': f'{fastDir}',
        'roidetect': False,
    }
    run_s2p(ops,db)
    if ops['two_step_registration'] and ops['keep_movie_raw']:
        rawbinFn = f'{saveFolderName}/plane0/data_raw.bin'
        os.remove(rawbinFn)

# Check the result
viewer = napari.Viewer()

for bs in bslist:
    saveFolderName = f'{planeDir}bstest{bs}/'
    
    opsFn = f'{saveFolderName}plane0/ops.npy'
    ops = np.load(opsFn, allow_pickle=True).item()
    binFn = f'{saveFolderName}plane0/data.bin'
    with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binFn) as f:
        data = f.data.astype(np.float32)
    viewer.add_image(data, name=f'[{bs}, {bs}]')

'''
The result is awful (too many weirdly distorted image). Why??
- Too small block size
- Other factors: maxregshift (larger the better?), maxregshiftNR (larger the better?), smooth_sigma (lower the better?), smooth_sigma_time (should be 0)
- Non-matched FOVs show very weird transformations (which actually maybe good)
For well-matched FOVs, there's no more blurry subregions (which is better than the previous method of registering each frame to the reference image)

Just by eyes, I can't be sure if other nonrigid registration methods will work better.
Maybe try a couple other public methods.

Also, I can't be sure this method will be better than running suite2p in each session and THEN aligning detected cells.
It is because there are some visible (though minor) differences in mean images.
What should be the criteria? # of detected (& curated) ROIs?
What is the tolerance? About 1% of missing ROIs - is this OK? What if 2%?
'''
#%%
# refImgs = np.zeros((len(bslist)+1,refImg.shape[0],refImg.shape[1]))
# refImgs[0,:,:] = refImg
# for i, bs in enumerate(bslist):
#     saveFolderName = f'{planeDir}bstest{bs}/'
#     opsFn = f'{saveFolderName}plane0/ops.npy'
#     ops = np.load(opsFn, allow_pickle=True).item()
#     refImgs[i+1,:,:] = ops['refImg']
# napari.view_image(refImgs)

#%% error with refImg? (didn't use .copy(), just assigned, so it could have been referenced)

plt.imshow(refImg.astype(np.uint16))
