# -*- coding: utf-8 -*-
"""
Test StackReg bilinear for session-to-session image registration

2022/01/20 JK
"""

import numpy as np
from pystackreg import StackReg
import napari
import os, glob

import gc
gc.enable()

def clahe_each(img: np.float64, kernel_size = None, clip_limit = 0.01, nbins = 2**16):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    newimg = exposure.equalize_adapthist(newimg, kernel_size = kernel_size, clip_limit = clip_limit, nbins=nbins)    
    return newimg

def imblend_for_napari(refImg, testImg):
    if (len(refImg.shape) != 2) or (len(testImg.shape) != 2):
        raise('Both images should have 2 dims.')
    if any(np.array(refImg.shape)-np.array(testImg.shape)):
        raise('Both images should have matching dims')
    refImg = img_norm(refImg.copy())
    testImg = img_norm(testImg.copy())
    refRGB = np.moveaxis(np.tile(refImg,(3,1,1)), 0, -1)
    testRGB = np.moveaxis(np.tile(testImg,(3,1,1)), 0, -1)
    blended = imblend(refImg, testImg)
    return np.array([refRGB, testRGB, blended])

def imblend(refImg, testImg):
    if (len(refImg.shape) != 2) or (len(testImg.shape) != 2):
        raise('Both images should have 2 dims.')
    if any(np.array(refImg.shape)-np.array(testImg.shape)):
        raise('Both images should have matching dims')
    Ly,Lx = refImg.shape
    blended = np.zeros((Ly,Lx,3))
    blended[:,:,0] = refImg
    blended[:,:,2] = testImg
    blended[:,:,1] = refImg
    return blended

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

def img_norm(img):
    if len(img.shape) == 2:
        return (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    elif len(img.shape) == 3:
        rimg = np.zeros(img.shape)
        for i in range(img.shape[0]):
            rimg[i,:,:] = (img[i,:,:] - np.amin(img[i,:,:])) / (np.amax(img[i,:,:]) - np.amin(img[i,:,:]))
        return rimg
    else:
        raise('Input image should have either 2 or 3 dims')

h5Dir = 'D:/TPM/JK/h5/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]


#%%
mi = 0
mouse = mice[mi]
pn = 1
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
sessionNames = [sn[4:] for sn in get_session_names(planeDir, mouse, pn)]
#%%
refImg = []
srBi = StackReg(StackReg.BILINEAR)
tform = []
out = []
for sn in sessionNames:
    opsFn = f'{planeDir}{sn}/plane0/ops.npy'
    ops = np.load(opsFn, allow_pickle=True).item()
    tmpMimg = ops['meanImg']
    if len(refImg)==0:
        refImg = tmpMimg
    tform.append(srBi.register(refImg, tmpMimg))
    out.append(srBi.transform(tmpMimg, tmat=tform[-1]))
#%%
viewer = napari.Viewer()
viewer.add_image(np.array(out), name=f'plane {pn}')

#%%
refImg = []
srAff = StackReg(StackReg.AFFINE)
tform = []
out = []
for sn in sessionNames:
    opsFn = f'{planeDir}{sn}/plane0/ops.npy'
    ops = np.load(opsFn, allow_pickle=True).item()
    tmpMimg = ops['meanImg']
    if len(refImg)==0:
        refImg = tmpMimg
    tform.append(srAff.register(refImg, tmpMimg))
    out.append(srAff.transform(tmpMimg, tmat=tform[-1]))

viewer = napari.Viewer()
viewer.add_image(np.array(out), name=f'plane {pn}')


'''
It's not good as suite2p nonlinear 2-step registration.
'''