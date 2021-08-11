# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 12:04:40 2021

@author: jkim
"""
#%% Visual inspection to decide FOV proportion selection
import h5py
import numpy as np
from suite2p.run_s2p import run_s2p, default_ops
import os, glob, shutil
import matplotlib.pyplot as plt
from skimage import exposure
import napari
import ffmpeg
import cv2
from suite2p.io.binary import BinaryFile
import time
from functools import reduce

def clahe_each(img):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img)) * (2**16-1)
    newimg = exposure.equalize_adapthist(newimg.astype(np.uint16))
    return newimg

def phase_corr(a,b):
    if a.shape != b.shape:
        raise('Dimensions must match')
    R = np.fft.fft2(a) * np.fft.fft2(b).conj()
    R /= np.absolute(R)
    r = np.absolute(np.fft.ifft2(R))
    ymax, xmax = np.unravel_index(np.argmax(r), r.shape)
    cmax = np.amax(r)
    center = r[0, 0]    
    return ymax, xmax, cmax, center, r

h5Dir = 'D:/TPM/JK/h5/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]

#%%
mi = 8
mouse = mice[mi]
planes = range(1,9)
# planes = [7]

mimgPlanes = []
viewer = napari.Viewer()
for planei in planes:
    mouseDir = f'{h5Dir}{mouse:03}/'
    planeDir = f'{mouseDir}plane_{planei}/'
        
    paramSetInd = 0
    testDn = f'test{paramSetInd:02}'
    dataDir = f'{planeDir}/{testDn}/plane0/'
    binfn = f'{dataDir}data.bin'
    opsfn = f'{dataDir}ops.npy'
    
    ops = np.load(opsfn, allow_pickle = True).item()
    framesPerSession = ops['framesPerSession']
    Ly = ops['Ly']
    Lx = ops['Lx']
    nframes = ops['nframes']
    framesPerSession = ops['framesPerSession']
    numSessions = len(ops['testSessionNames'])
    meanImgs = np.zeros((numSessions,Ly,Lx))
    with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
        for i in range(numSessions):
            inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
            frames = f.ix(indices=inds).astype(np.float32)
            tempMimg = clahe_each(frames.mean(axis=0))
            meanImgs[i,:,:]= tempMimg
            # viewer.add_image(tempMimg, name = ops['testSessionNames'][i])
    viewer.add_image(meanImgs, name = f'plane#{planei}')
    mimgPlanes.append(meanImgs)
        
#%% Set top bottom left right margin in "pixel number" and test the result
topMargin = 40
bottomMargin = 350
leftMargin = 80
rightMargin = 590
viewer = napari.Viewer()
for pi, planei in enumerate(planes):
    viewer.add_image(mimgPlanes[pi][:,topMargin:bottomMargin, leftMargin:rightMargin], name = f'plane#{planei}')


#%% Compare with the ref session, collect and save all reg params and results
Ly = bottomMargin - topMargin
Lx = rightMargin - leftMargin

for planei in planes:
    # Just compare with the ref session
    # Also, collect all reg params
    refSession = refSessions[mi]
    refSname = f'{mouse:03}_{refSession:03}_'
    refSi = [i for i,fn in enumerate(ops['testSessionNames']) if refSname in fn]
    tic = time.time()
    planeDir = f'{mouseDir}plane_{planei}/'
    refPC = np.zeros(111)
    refIC = np.zeros(111)
    nonrigidVals = np.zeros(111, dtype=bool)
    maxregshiftVals = np.zeros(111, dtype=float)
    maxregshiftNRVals = np.zeros(111, dtype=np.uint8)
    blocksizeVals = np.zeros(111, dtype=np.uint8)
    snrthreshVals = np.zeros(111, dtype=float)
    for testn in range(111):
        testDir = f'{planeDir}/test{testn:02}/plane0/'
        ops = np.load(f'{testDir}ops.npy', allow_pickle=True).item()
        
        nonrigidVals[testn] = ops['nonrigid']
        maxregshiftVals[testn] = ops['maxregshift']
        maxregshiftNRVals[testn] = ops['maxregshiftNR']
        blocksizeVals[testn] = ops['block_size'][0]
        snrthreshVals[testn] = ops['snr_thresh']
        
        fpsession = ops['framesPerSession']
        numSessions = len(ops['testFileList'])
        binfn = f'{testDir}data.bin'
        perSessionMeanImg = np.zeros((Ly,Lx,numSessions))
        with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
            for i in range(numSessions):
                inds = np.arange(i*fpsession,(i+1)*fpsession)
                frames = f.ix(indices=inds).astype(np.float32)
                perSessionMeanImg[:,:,i] = clahe_each(frames.mean(axis=0))[topMargin:bottomMargin, leftMargin:rightMargin]
        
        phaseCorr = np.zeros(numSessions)
        imgCorr = np.zeros(numSessions)
        refImg = perSessionMeanImg[:,:,refSi[0]]
        for i in range(numSessions):
            img1 = perSessionMeanImg[:,:,i]
            _, _, _, phaseCorr[i], _ = phase_corr(img1, refImg)
            imgCorr[i] = np.corrcoef(img1.flatten(), refImg.flatten())[0,1]
    
        refPC[testn] = np.mean(phaseCorr[phaseCorr!=0])
        refIC[testn] = np.mean(imgCorr[imgCorr!=0])
    toc= time.time()
    timeInMin = np.round((toc-tic)/60)
    print(f'{timeInMin} minutes for plane#{planei}.')
    
    # Save the result
    savefn = f'{planeDir}regParamTestResult.npy'
    
    result = {}
    result['nonrigidVals'] = nonrigidVals
    result['maxregshiftVals'] = maxregshiftVals
    result['maxregshiftNRVals'] = maxregshiftNRVals
    result['blocksizeVals'] = blocksizeVals
    result['snrthreshVals'] = snrthreshVals
    result['refPC'] = refPC
    result['refIC'] = refIC
    result['topMargin'] = topMargin
    result['bottomMargin'] = bottomMargin
    result['leftMargin'] = leftMargin
    result['rightMargin'] = rightMargin
    
    np.save(savefn, result)

    f, ax = plt.subplots()
    ax.plot(refPC, color='b')
    axr = ax.twinx()
    axr.plot(refIC, color='y')
    ax.set_ylabel('Phase correlation', color='b')
    ax.tick_params(axis= 'y', colors='b')
    axr.set_ylabel('Image correlation', color='y')
    axr.tick_params(axis= 'y', colors='y')
    ax.set_xlabel('Param #')
    ax.set_title(f'Plane #{planei}')

    # Finding the best condition
    maxregshiftList = [0.1, 0.2, 0.3]
    maxregshiftNRList = [5, 10, 20]
    block_sizeList = [[128,128], [96, 96], [64, 64]]
    snr_threshList = [1, 1.1, 1.2, 1.3]
    
    clist = ['k','b','c','y']
    
    
    f, ax = plt.subplots(2,4)
    nrInd = np.where(nonrigidVals)[0]
    # Group by maxregshiftList
    for i, val in enumerate(maxregshiftList):
        valInd = np.where(maxregshiftVals == val)[0]
        tempInd = np.intersect1d(nrInd, valInd)
        ax[0][0].plot(refPC[tempInd], color = clist[i])
        ax[1][0].plot(refIC[tempInd], color = clist[i])
    ax[0][0].legend(maxregshiftList)
    ax[0][0].set_title('maxregshift')
    ax[0][0].set_ylabel('Phase correlation')
    ax[1][0].set_ylabel('Pixel correlation')
    # Group by maxregshiftNRList
    for i, val in enumerate(maxregshiftNRList):
        valInd = np.where(maxregshiftNRVals == val)[0]
        tempInd = np.intersect1d(nrInd, valInd)
        ax[0][1].plot(refPC[tempInd], color = clist[i])
        ax[1][1].plot(refIC[tempInd], color = clist[i])
    ax[0][1].legend(maxregshiftNRList)
    ax[0][1].set_title('maxregshiftNR')
    # Group by block_sizeList
    for i, val in enumerate(block_sizeList):
        valInd = np.where(blocksizeVals == val[0])[0]
        tempInd = np.intersect1d(nrInd, valInd)
        ax[0][2].plot(refPC[tempInd], color = clist[i])
        ax[1][2].plot(refIC[tempInd], color = clist[i])
    ax[0][2].legend(block_sizeList)
    ax[0][2].set_title('block_size')
    # Group by snr_threshList
    for i, val in enumerate(snr_threshList):
        valInd = np.where(snrthreshVals == val)[0]
        tempInd = np.intersect1d(nrInd, valInd)
        ax[0][3].plot(refPC[tempInd], color = clist[i])
        ax[1][3].plot(refIC[tempInd], color = clist[i])
    ax[0][3].legend(snr_threshList)
    ax[0][3].set_title('snr_thresh')
    f.tight_layout()
    f.suptitle(f'JK{mouse:03} plane#{planei}')


#%%
result = np.load('D:/TPM/JK/h5/052/plane_3/regParamTestResult.npy', allow_pickle = True).item()
f, ax = plt.subplots()
ax.plot(result['refIC'], color='b')
axr = ax.twinx()
axr.plot(result['refPC'], color='y')
#%% Visual inspection
mouse = 25
testPi = 5

# meanImgs = mimgPlanes[testPi]
# viewer = napari.view_image(meanImgs)
    
paramSetInd = 5

mouseDir = f'{h5Dir}{mouse:03}/'
planeDir = f'{mouseDir}plane_{testPi}/'
testDn = f'test{paramSetInd:02}'
dataDir = f'{planeDir}/{testDn}/plane0/'
binfn = f'{dataDir}data.bin'
opsfn = f'{dataDir}ops.npy'


ops = np.load(opsfn, allow_pickle = True).item()
framesPerSession = ops['framesPerSession']
Ly = ops['Ly']
Lx = ops['Lx']
nframes = ops['nframes']
framesPerSession = ops['framesPerSession']
numSessions = len(ops['testSessionNames'])
meanImgs = np.zeros((numSessions,Ly,Lx))
with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
        frames = f.ix(indices=inds).astype(np.float32)
        tempMimg = clahe_each(frames.mean(axis=0))
        meanImgs[i,:,:]= tempMimg
        # viewer.add_image(tempMimg, name = ops['testSessionNames'][i])
napari.view_image(meanImgs, name = f'plane#{planei}')


#%% Selecting best parameter combination for all test sessions
miList = [0, 3, 8]
planes = range(1,9)
pcmaxReg = np.zeros((len(miList),len(planes)))
pcmaxRegnr = np.zeros((len(miList),len(planes)))
pcmaxBsize = np.zeros((len(miList),len(planes)))
pcmaxSnr = np.zeros((len(miList),len(planes)))

icmaxReg = np.zeros((len(miList),len(planes)))
icmaxRegnr = np.zeros((len(miList),len(planes)))
icmaxBsize = np.zeros((len(miList),len(planes)))
icmaxSnr = np.zeros((len(miList),len(planes)))

for i, mi in enumerate(miList):
    mouse = mice[mi]
    for j, pi in enumerate(planes):
        resultfn = f'D:/TPM/JK/h5/{mouse:03}/plane_{pi}/regParamTestResult.npy'
        result = np.load(resultfn, allow_pickle = True).item()
        
        # pcmaxInd = np.argmax(result['refPC'])
        # icmaxInd = np.argmax(result['refIC'])
        
        pcmaxInd = np.where(result['refPC'] == np.amax(result['refPC']))[0][-1]
        icmaxInd = np.where(result['refIC'] == np.amax(result['refIC']))[0][-1]
        
        pcmaxReg[i, j] = result['maxregshiftVals'][pcmaxInd]
        pcmaxRegnr[i, j] = result['maxregshiftNRVals'][pcmaxInd]
        pcmaxBsize[i, j] = result['blocksizeVals'][pcmaxInd]
        pcmaxSnr[i, j] = result['snrthreshVals'][pcmaxInd]
        
        icmaxReg[i, j] = result['maxregshiftVals'][icmaxInd]
        icmaxRegnr[i, j] = result['maxregshiftNRVals'][icmaxInd]
        icmaxBsize[i, j] = result['blocksizeVals'][icmaxInd]
        icmaxSnr[i, j] = result['snrthreshVals'][icmaxInd]

f,ax = plt.subplots(4,2)
ax[0,0].imshow(pcmaxReg, vmin = 0.1, vmax = 0.3)
ax[0,0].set_title('Phase correlation\nMax reg shift')
a = ax[0,1].imshow(icmaxReg, vmin = 0.1, vmax = 0.3)
f.colorbar(a, ax=ax[0,1])
ax[0,1].set_title('Pixel correlation\nMax reg shift')
ax[1,0].imshow(pcmaxRegnr, vmin = 5, vmax = 20)
ax[1,0].set_title('Max reg shift non-rigid')
a = ax[1,1].imshow(icmaxRegnr, vmin = 5, vmax = 20)
f.colorbar(a, ax=ax[1,1])
ax[1,1].set_title('Max reg shift non-rigid')
ax[2,0].imshow(pcmaxBsize, vmin = 64, vmax = 128)
ax[2,0].set_title('Block size')
a = ax[2,1].imshow(icmaxBsize, vmin = 64, vmax = 128)
f.colorbar(a, ax=ax[2,1])
ax[2,1].set_title('Block size')
ax[3,0].imshow(pcmaxSnr, vmin = 1, vmax = 1.3)
ax[3,0].set_title('SNR threshold')
a = ax[3,1].imshow(icmaxSnr, vmin = 1, vmax = 1.3)
f.colorbar(a, ax=ax[3,1])
ax[3,1].set_title('SNR threshold')
for i in range(4):
    for j in range(2):
        ax[i,j].set_xticklabels('')
        ax[i,j].set_yticklabels('')
    ax[i,0].set_ylabel('Mouse')
for j in range(2):
    ax[i,j].set_xlabel('Plane')
f.tight_layout()


#%%
from functools import reduce
'''
maxregshift 0.3 is very close to max value in all cases
maxregshiftNR 20 is rarely the best, and could be replaced with either 5 or 10  (JK025 plane 1)
snrthresh 1.2 and 1.3 are better than 1 and 1.1
Quantify the effect of replacing these values 
normalized by maxPC - default PC
'''

miList = [0,0,0,0,3,3,3,3,8]
piList = [3,4,6,7,2,3,4,5,4]
defaultInd = 5
absDiff = np.zeros(len(miList))
normDiff = np.zeros(len(miList))
for i, [mi, pi] in enumerate(zip(miList,piList)):
    mouse = mice[mi]
    resultfn = f'D:/TPM/JK/h5/{mouse:03}/plane_{pi}/regParamTestResult.npy'
    result = np.load(resultfn, allow_pickle = True).item()
    # pcmaxInd = np.where(result['refPC'] == np.amax(result['refPC']))[0][-1]
    # regnr = result['maxregshiftNRVals'][pcmaxInd]
    # bsize = result['blocksizeVals'][pcmaxInd]
    # snr = result['snrthreshVals'][pcmaxInd]
    
    # regnrInds = np.where(result['maxregshiftNRVals']==regnr)[0]
    # bsizeInds = np.where(result['blocksizeVals']==bsize)[0]
    # snrInds = np.where(result['snrthreshVals']==snr)[0]
    mrsInds = np.where(result['maxregshiftVals']==0.3)[0]
    
    # testInd = reduce(np.intersect1d, (regnrInds, bsizeInds, snrInds, mrsInds))
    # if len(testInd) != 1:
    #     raise('test index should be in length of 1')
    
    defaultPC = result['refPC'][5]
    maxPC = np.amax(result['refPC'])
    # testPC = result['refPC'][testInd]
    testPC = np.amax(result['refPC'][mrsInds])
    
    absDiff[i] = (maxPC-testPC)
    normDiff[i] = (maxPC-testPC) / (maxPC-defaultPC)
    
print(np.amax(normDiff))
mdiff = np.mean(normDiff)
sdiff = np.std(normDiff)
print(f'{mdiff} \u00B1 {sdiff}')

#%% Forcing maxregshiftNR to be 5 or 10
mi = 0
pi = 1
mouse = mice[mi]
resultfn = f'D:/TPM/JK/h5/{mouse:03}/plane_{pi}/regParamTestResult.npy'
result = np.load(resultfn, allow_pickle = True).item()
nrInds = np.where(result['maxregshiftNRVals']<20)
defaultPC = result['refPC'][5]
maxPC = np.amax(result['refPC'])
testPC = np.amax(result['refPC'][nrInds])
normDiff = (maxPC-testPC) / (maxPC-defaultPC)

#%% Forcing snrthresh to be 1.2 or 1.3
miList = [0,0,8]
piList = [4,8,8]
absDiff = np.zeros(len(miList))
normDiff = np.zeros(len(miList))
for i, [mi, pi] in enumerate(zip(miList,piList)):
    mouse = mice[mi]
    resultfn = f'D:/TPM/JK/h5/{mouse:03}/plane_{pi}/regParamTestResult.npy'
    result = np.load(resultfn, allow_pickle = True).item()
    snrInds = np.where(result['snrthreshVals']>1.1)[0]
    
    defaultPC = result['refPC'][5]
    maxPC = np.amax(result['refPC'])
    testPC = np.amax(result['refPC'][snrInds])
    
    absDiff[i] = (maxPC-testPC)
    normDiff[i] = (maxPC-testPC) / (maxPC-defaultPC)
    
print(np.amax(normDiff))
mdiff = np.mean(normDiff)
sdiff = np.std(normDiff)
print(f'{mdiff} \u00B1 {sdiff}')

#%% Increase in registration quality
miList = [0,3,8]
piList = range(1,9)
rqIncrease = np.zeros((len(miList), len(piList)))
for i, mi in enumerate(miList):
    mouse = mice[mi]
    for j, pi in enumerate(piList):
        resultfn = f'D:/TPM/JK/h5/{mouse:03}/plane_{pi}/regParamTestResult.npy'
        result = np.load(resultfn, allow_pickle = True).item()
        defaultPC = result['refPC'][5]
        maxPC = np.amax(result['refPC'])
        
        rqIncrease[i,j] = (maxPC - defaultPC)/defaultPC * 100

#%%
np.amax(rqIncrease)
np.mean(rqIncrease)
np.std(rqIncrease)


#%%
#%% Compare with the ref session, collect and save all reg params and results
#%% Including test #200-223
testNums = np.arange(200,223, dtype=np.uint16)
for mi in [0,3,8]:
    mouse = mice[mi]
    mouseDir = f'{h5Dir}{mouse:03}/'
    for pi in range(1,9):
        prevResultFn = f'D:/TPM/JK/h5/{mouse:03}/plane_{pi}/regParamTestResult.npy'
        result = np.load(prevResultFn, allow_pickle = True).item()
        
        topMargin = result['topMargin']
        bottomMargin = result['bottomMargin']
        leftMargin = result['leftMargin']
        rightMargin = result['rightMargin']
            
        Ly = bottomMargin - topMargin
        Lx = rightMargin - leftMargin

        nonrigidVals = result['nonrigidVals']
        maxregshiftVals = result['maxregshiftVals']
        maxregshiftNRVals = result['maxregshiftNRVals']
        blocksizeVals = result['blocksizeVals']
        snrthreshVals = result['snrthreshVals']
        refPC = result['refPC']
        refIC = result['refIC']
        
        # Just compare with the ref session
        # Also, collect all reg params
        planeDir = f'{mouseDir}plane_{pi}/'
        testDir = f'{planeDir}/test200/plane0/'
        ops = np.load(f'{testDir}ops.npy', allow_pickle=True).item()
        refSession = refSessions[mi]
        refSname = f'{mouse:03}_{refSession:03}_'
        refSi = [i for i,fn in enumerate(ops['testSessionNames']) if refSname in fn]
        tic = time.time()
        
        for testn in testNums:
            testDir = f'{planeDir}/test{testn:02}/plane0/'
            ops = np.load(f'{testDir}ops.npy', allow_pickle=True).item()
            
            nonrigidVals = np.append(nonrigidVals, ops['nonrigid'])
            maxregshiftVals = np.append(maxregshiftVals, ops['maxregshift'])
            maxregshiftNRVals = np.append(maxregshiftNRVals, ops['maxregshiftNR'])
            blocksizeVals = np.append(blocksizeVals, ops['block_size'][0])
            snrthreshVals = np.append(snrthreshVals, ops['snr_thresh'])
            
            fpsession = ops['framesPerSession']
            numSessions = len(ops['testFileList'])
            binfn = f'{testDir}data.bin'
            perSessionMeanImg = np.zeros((Ly,Lx,numSessions))
            with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
                for i in range(numSessions):
                    inds = np.arange(i*fpsession,(i+1)*fpsession)
                    frames = f.ix(indices=inds).astype(np.float32)
                    perSessionMeanImg[:,:,i] = clahe_each(frames.mean(axis=0))[topMargin:bottomMargin, leftMargin:rightMargin]
            
            phaseCorr = np.zeros(numSessions)
            imgCorr = np.zeros(numSessions)
            refImg = perSessionMeanImg[:,:,refSi[0]]
            for i in range(numSessions):
                img1 = perSessionMeanImg[:,:,i]
                _, _, _, phaseCorr[i], _ = phase_corr(img1, refImg)
                imgCorr[i] = np.corrcoef(img1.flatten(), refImg.flatten())[0,1]
        
            refPC = np.append(refPC, np.mean(phaseCorr[phaseCorr!=0]))
            refIC = np.append(refIC, np.mean(imgCorr[imgCorr!=0]))
        toc= time.time()
        timeInMin = np.round((toc-tic)/60)
        print(f'{timeInMin} minutes for JK{mouse:03} plane#{pi}.')
        
        # Save the result
        savefn = f'{planeDir}regParamTestResult2.npy'
        
        result['nonrigidVals'] = nonrigidVals
        result['maxregshiftVals'] = maxregshiftVals
        result['maxregshiftNRVals'] = maxregshiftNRVals
        result['blocksizeVals'] = blocksizeVals
        result['snrthreshVals'] = snrthreshVals
        result['refPC'] = refPC
        result['refIC'] = refIC
        
        np.save(savefn, result)

#%% 210707 Finalize parameters to test for the rest of mice
# First, compare block_size between 64, 48, and 32
# with the same snr_thresh (1.2 and 1.3)
from functools import reduce
clist = ['k','b','c']

for mi in [0,3,8]:

    mouse = mice[mi]
    mouseDir = f'{h5Dir}{mouse:03}/'
    
    f, ax = plt.subplots(2,4, sharex=True)
    for pi in range(1,9):
        axi = pi-1
        yi = int(np.floor(axi/4))
        xi = np.mod(axi,4)
        planeDir = f'{mouseDir}plane_{pi}/'
        # for pi in range(1,9):
        mrs = 0.3
        mrsNR = [5, 10]
        snrThresh = [1.2, 1.3]
        bsList = [64, 48, 32]
        
        ResultFn = f'D:/TPM/JK/h5/{mouse:03}/plane_{pi}/regParamTestResult2.npy'
        result = np.load(ResultFn, allow_pickle = True).item()
        maxregshiftVals = result['maxregshiftVals'][110:]
        maxregshiftNRVals = result['maxregshiftNRVals'][110:]
        block_sizeVals = result['blocksizeVals'][110:]
        snrthreshVals = result['snrthreshVals'][110:]
        refPC = result['refPC'][110:]
        
        mrsInds = np.where(maxregshiftVals == mrs)[0]
        mrsNRInds = np.array([], dtype=np.uint16)
        for val in mrsNR:
            mrsNRInds = np.append(mrsNRInds, np.where(maxregshiftNRVals == val)[0])
        snrInds = np.array([], dtype=np.uint16)
        for val in snrThresh:
            snrInds = np.append(snrInds, np.where(snrthreshVals == val)[0])
        bsInds = np.array([], dtype=np.uint16)
        for val in bsList:
            bsInds = np.append(bsInds, np.where(block_sizeVals == val)[0])
        testInd = reduce(np.intersect1d, (mrsInds, mrsNRInds, snrInds, bsInds))
        
        mrsNRVals = maxregshiftNRVals[testInd]
        snrThreshVals = snrthreshVals[testInd]
        blockSizeVals = block_sizeVals[testInd]
        pcVals = refPC[testInd]
        
        for i, val in enumerate(bsList):
            tempind = np.where(blockSizeVals==val)[0]
            ax[yi][xi].plot(pcVals[tempind], color = clist[i])
            ax[yi][xi].set_title(f'plane #{pi}')
        if (yi == 1) & (xi == 0):
            ax[yi][xi].set_xlabel('Param #')
            ax[yi][xi].set_ylabel('Phase correlation')
        if (yi == 0) & (xi == 0):
            ax[yi][xi].legend(bsList)
    f.suptitle(f'JK{mouse:03}')
    f.tight_layout()
    
#%% SNR threshold test
# with block size 48 and 32
clist = ['k','b','c', 'g']

for mi in [0,3,8]:

    mouse = mice[mi]
    mouseDir = f'{h5Dir}{mouse:03}/'
    
    f, ax = plt.subplots(2,4, sharex=True)
    for pi in range(1,9):
        axi = pi-1
        yi = int(np.floor(axi/4))
        xi = np.mod(axi,4)
        planeDir = f'{mouseDir}plane_{pi}/'
        # for pi in range(1,9):
        mrs = 0.3
        mrsNR = [5, 10]
        snrThresh = [1.2, 1.3, 1.4, 1.5]
        bsList = [48, 32]
        
        ResultFn = f'D:/TPM/JK/h5/{mouse:03}/plane_{pi}/regParamTestResult2.npy'
        result = np.load(ResultFn, allow_pickle = True).item()
        maxregshiftVals = result['maxregshiftVals'][110:]
        maxregshiftNRVals = result['maxregshiftNRVals'][110:]
        block_sizeVals = result['blocksizeVals'][110:]
        snrthreshVals = result['snrthreshVals'][110:]
        refPC = result['refPC'][110:]
        
        mrsInds = np.where(maxregshiftVals == mrs)[0]
        mrsNRInds = np.array([], dtype=np.uint16)
        for val in mrsNR:
            mrsNRInds = np.append(mrsNRInds, np.where(maxregshiftNRVals == val)[0])
        snrInds = np.array([], dtype=np.uint16)
        for val in snrThresh:
            snrInds = np.append(snrInds, np.where(snrthreshVals == val)[0])
        bsInds = np.array([], dtype=np.uint16)
        for val in bsList:
            bsInds = np.append(bsInds, np.where(block_sizeVals == val)[0])
        testInd = reduce(np.intersect1d, (mrsInds, mrsNRInds, snrInds, bsInds))
        
        mrsNRVals = maxregshiftNRVals[testInd]
        snrThreshVals = snrthreshVals[testInd]
        blockSizeVals = block_sizeVals[testInd]
        pcVals = refPC[testInd]
        
        for i, val in enumerate(snrThresh):
            tempind = np.where(snrThreshVals==val)[0]
            ax[yi][xi].plot(pcVals[tempind], color = clist[i])
            ax[yi][xi].set_title(f'plane #{pi}')
        if (yi == 1) & (xi == 0):
            ax[yi][xi].set_xlabel('Param #')
            ax[yi][xi].set_ylabel('Phase correlation')
        if (yi == 0) & (xi == 0):
            ax[yi][xi].legend(snrThresh)
    f.suptitle(f'JK{mouse:03}')
    f.tight_layout()

#%% Visual inspection
# NR 5 and bs 32 (snr 1.2, mrs 0.3)
# Compare with bs 64
testNums = [*range(111), *range(200,224)]

mi = 0
mouse = mice[mi]
mouseDir = f'{h5Dir}{mouse:03}/'    
pi = 1
ResultFn = f'{mouseDir}plane_{pi}/regParamTestResult2.npy'
result = np.load(ResultFn, allow_pickle = True).item()

topMargin = result['topMargin']
bottomMargin = result['bottomMargin']
leftMargin = result['leftMargin']
rightMargin = result['rightMargin']

maxregshiftVals = result['maxregshiftVals']
maxregshiftNRVals = result['maxregshiftNRVals']
block_sizeVals = result['blocksizeVals']
snrthreshVals = result['snrthreshVals']

mrsInds = np.where(maxregshiftVals == 0.3)[0]
nrInds = np.where(maxregshiftNRVals == 5)[0]
snrInds = np.where(snrthreshVals == 1.2)[0]
bsInds = np.where(block_sizeVals == 32)

bestInd = reduce(np.intersect1d, (mrsInds, nrInds, snrInds, bsInds))[0]
testNum = testNums[bestInd]
bestDir = f'{mouseDir}plane_{pi}/test{testNum}/plane0/'
opsfn = f'{bestDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()

fpsession = ops['framesPerSession']
numSessions = len(ops['testFileList'])
binfn = f'{bestDir}data.bin'
Ly = bottomMargin - topMargin
Lx = rightMargin - leftMargin
perSessionMeanImg = np.zeros((numSessions, Ly,Lx))
with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*fpsession,(i+1)*fpsession)
        frames = f.ix(indices=inds).astype(np.float32)
        perSessionMeanImg[i,:,:] = clahe_each(frames.mean(axis=0))[topMargin:bottomMargin, leftMargin:rightMargin]

napari.view_image(perSessionMeanImg)

#%% bs 64
testNums = [*range(111), *range(200,224)]

mi = 0
mouse = mice[mi]
mouseDir = f'{h5Dir}{mouse:03}/'    
pi = 1
ResultFn = f'{mouseDir}plane_{pi}/regParamTestResult2.npy'
result = np.load(ResultFn, allow_pickle = True).item()

topMargin = result['topMargin']
bottomMargin = result['bottomMargin']
leftMargin = result['leftMargin']
rightMargin = result['rightMargin']

maxregshiftVals = result['maxregshiftVals']
maxregshiftNRVals = result['maxregshiftNRVals']
block_sizeVals = result['blocksizeVals']
snrthreshVals = result['snrthreshVals']

mrsInds = np.where(maxregshiftVals == 0.3)[0]
nrInds = np.where(maxregshiftNRVals == 5)[0]
snrInds = np.where(snrthreshVals == 1.2)[0]
bsInds = np.where(block_sizeVals == 64)[0]

testInd = reduce(np.intersect1d, (mrsInds, nrInds, snrInds, bsInds))[0]
testNum = testNums[testInd]
bestDir = f'{mouseDir}plane_{pi}/test{testNum}/plane0/'
opsfn = f'{bestDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()

fpsession = ops['framesPerSession']
numSessions = len(ops['testFileList'])
binfn = f'{bestDir}data.bin'
Ly = bottomMargin - topMargin
Lx = rightMargin - leftMargin
perSessionMeanImg = np.zeros((numSessions, Ly,Lx))
with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*fpsession,(i+1)*fpsession)
        frames = f.ix(indices=inds).astype(np.float32)
        perSessionMeanImg[i,:,:] = clahe_each(frames.mean(axis=0))[topMargin:bottomMargin, leftMargin:rightMargin]

napari.view_image(perSessionMeanImg)

#%% Compare between 32 and 64
mi = 8
mouse = mice[mi]
mouseDir = f'{h5Dir}{mouse:03}/'    
pi = 2
ResultFn = f'{mouseDir}plane_{pi}/regParamTestResult2.npy'
result = np.load(ResultFn, allow_pickle = True).item()

topMargin = result['topMargin']
bottomMargin = result['bottomMargin']
leftMargin = result['leftMargin']
rightMargin = result['rightMargin']

maxregshiftVals = result['maxregshiftVals']
maxregshiftNRVals = result['maxregshiftNRVals']
block_sizeVals = result['blocksizeVals']
snrthreshVals = result['snrthreshVals']

mrsInds = np.where(maxregshiftVals == 0.3)[0]
nrInds = np.where(maxregshiftNRVals == 5)[0]
snrInds = np.where(snrthreshVals == 1.2)[0]
bsInds = np.where(block_sizeVals == 32)[0]

bestInd = reduce(np.intersect1d, (mrsInds, nrInds, snrInds, bsInds))[0]
testNum = testNums[bestInd]
bestDir = f'{mouseDir}plane_{pi}/test{testNum}/plane0/'
opsfn = f'{bestDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()

fpsession = ops['framesPerSession']
numSessions = len(ops['testFileList'])
binfn = f'{bestDir}data.bin'
Ly = bottomMargin - topMargin
Lx = rightMargin - leftMargin
perSessionMeanImg = np.zeros((numSessions, Ly,Lx,3))
with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*fpsession,(i+1)*fpsession)
        frames = f.ix(indices=inds).astype(np.float32)
        perSessionMeanImg[i,:,:,0] = clahe_each(frames.mean(axis=0))[topMargin:bottomMargin, leftMargin:rightMargin]
        
bsInds = np.where(block_sizeVals == 64)[0]

testInd = reduce(np.intersect1d, (mrsInds, nrInds, snrInds, bsInds))[0]
testNum = testNums[testInd]
bestDir = f'{mouseDir}plane_{pi}/test{testNum}/plane0/'
opsfn = f'{bestDir}ops.npy'
ops = np.load(opsfn, allow_pickle = True).item()

fpsession = ops['framesPerSession']
numSessions = len(ops['testFileList'])
binfn = f'{bestDir}data.bin'
Ly = bottomMargin - topMargin
Lx = rightMargin - leftMargin
# perSessionMeanImg64 = np.zeros((numSessions, Ly,Lx))
with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*fpsession,(i+1)*fpsession)
        frames = f.ix(indices=inds).astype(np.float32)
        perSessionMeanImg[i,:,:,1] = clahe_each(frames.mean(axis=0))[topMargin:bottomMargin, leftMargin:rightMargin]

viewer = napari.Viewer()
viewer.add_image(perSessionMeanImg[:,:,:,0], blending='additive', colormap = 'magenta', name = 'bs [32, 32]')
viewer.add_image(perSessionMeanImg[:,:,:,1], blending='additive', colormap = 'cyan', name = 'bs [64, 64]')
#%%
'''
Not so much improvement from well-matched sessions, but non-matched sessions seem to be quite different between block sizes
So I need to select some well-matched and non-matched sessions and compare the effect of block sizes within these sessions
'''
#%% Select well-matched and non-matched from correlation values and visual inspection using default parameters (block size [128, 128])
# Do this volumewise
# I already have JK025
from functools import reduce
testNums = [*range(111), *range(200,224)]

mi = 0

mouse = mice[mi]
mouseDir = f'{h5Dir}{mouse:03}/' 
viewer =napari.Viewer()
f, ax = plt.subplots()

errorPlanes = [3,4,5]
errorInds = [1,4] # remove these indices from error planes 3,4,5, if mi == 8
piList = range(1,5)
# piList = range(5,9)
for pi in piList:
    ResultFn = f'{mouseDir}plane_{pi}/regParamTestResult2.npy'
    result = np.load(ResultFn, allow_pickle = True).item()
    
    nrVals = result['nonrigidVals']
    maxregshiftVals = result['maxregshiftVals']
    maxregshiftNRVals = result['maxregshiftNRVals']
    block_sizeVals = result['blocksizeVals']
    snrthreshVals = result['snrthreshVals']
    
    nrInds = np.where(nrVals==True)[0]
    mrsInds = np.where(maxregshiftVals == 0.3)[0]
    mrsnrInds = np.where(maxregshiftNRVals == 5)[0]
    snrInds = np.where(snrthreshVals == 1.2)[0]
    bsInds = np.where(block_sizeVals == 128)[0]
    
    defaultInd = reduce(np.intersect1d, (nrInds, mrsInds, mrsnrInds, snrInds, bsInds))[0]
    defaultNum = testNums[defaultInd]
    defaultDir = f'{mouseDir}plane_{pi}/test{defaultNum}/plane0/'
    opsfn = f'{defaultDir}ops.npy'
    ops = np.load(opsfn, allow_pickle = True).item()
    
    fpsession = ops['framesPerSession']
    numSessions = len(ops['testFileList'])
    binfn = f'{defaultDir}data.bin'
    
    topMargin = result['topMargin']
    bottomMargin = result['bottomMargin']
    leftMargin = result['leftMargin']
    rightMargin = result['rightMargin']
    Ly = bottomMargin - topMargin
    Lx = rightMargin - leftMargin
    perSessionMeanImg = np.zeros((numSessions, Ly,Lx))
    with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
        for i in range(numSessions):
            inds = np.arange(i*fpsession,(i+1)*fpsession)
            frames = f.ix(indices=inds).astype(np.float32)
            perSessionMeanImg[i,:,:] = clahe_each(frames.mean(axis=0))[topMargin:bottomMargin, leftMargin:rightMargin]
    if (mi == 8) & (pi in errorPlanes):
        perSessionMeanImg = np.delete(perSessionMeanImg, errorInds, axis=0)
    numSessions = perSessionMeanImg.shape[0]
        
    phaseCorr = np.zeros(numSessions)
    imgCorr = np.zeros(numSessions)
    refSession = refSessions[mi]
    refSname = f'{mouse:03}_{refSession:03}_'
    if (mi == 8) & (pi in errorPlanes):
        testSessionNames = [tsn for i, tsn in enumerate(ops['testSessionNames']) if i not in errorInds]
    else:
        testSessionNames = ops['testSessionNames']
    refSi = [i for i,fn in enumerate(testSessionNames) if refSname in fn]
    refImg = perSessionMeanImg[refSi[0],:,:]
    for i in range(numSessions):
        img1 = perSessionMeanImg[i,:,:]
        _, _, _, phaseCorr[i], _ = phase_corr(img1, refImg)
    ax.plot(phaseCorr)
    compareMimg = np.zeros((numSessions, Ly, Lx,3))
    for i in range(numSessions):
        compareMimg[i,:,:,0] = refImg
        compareMimg[i,:,:,2] = refImg
    compareMimg[:,:,:,1] = perSessionMeanImg
    viewer.add_image(compareMimg, rgb=True, name = f'plane {pi}')
ax.legend([f'plane {pi}' for pi in piList])
ax.set_title(f'JK{mouse:03}')
ax.set_ylabel('Phase correlation value')
ax.set_xlabel('Session index')

#%% Applying well-matched, non-matched sessions
# Compare between different block sizes
# 
wellMatchedList = np.array([[1,5,19,21,22], [0,6,11,18,23], [1,2,5,22,23], [2,8,11,22], [2,6,16,27,29], [15,18,20,28,29]])
nonMatchedList = np.array([[9,17,24,25], [16,22,24,25,26], [3,12,21,27], [10,19,23,27], [22,30,32], [19,22,30,32]])
mouseList = np.array([25,25,36,36,52,52])
planesList = np.array([[1,2,3,4], [5,6,7,8], [1,2,3,4], [5,6,7,8], [1,2,3,4], [5,6,7,8]])

from functools import reduce
testNums = [*range(111), *range(200,224)]

mrs = 0.3
mrsNR = 5
snrThresh = 1.2
bsList = [128,96,64,48,32]
clist = ['k', 'b', 'c', 'g', 'y']

errorPlanes = [3,4,5]
errorSessions = [2,5] # remove these sessions from error planes 3,4,5, if mi == 8

for mi in [8]:
    mouse = mice[mi]
    mouseDir = f'{h5Dir}{mouse:03}/'
    for pi in range(3,9):
        ResultFn = f'{mouseDir}plane_{pi}/regParamTestResult2.npy'
        result = np.load(ResultFn, allow_pickle = True).item()
        
        topMargin = result['topMargin']
        bottomMargin = result['bottomMargin']
        leftMargin = result['leftMargin']
        rightMargin = result['rightMargin']
        Ly = bottomMargin - topMargin
        Lx = rightMargin - leftMargin
        
        nrVals = result['nonrigidVals']
        maxregshiftVals = result['maxregshiftVals']
        maxregshiftNRVals = result['maxregshiftNRVals']
        block_sizeVals = result['blocksizeVals']
        snrthreshVals = result['snrthreshVals']
        
        nrInds = np.where(nrVals==True)[0]
        mrsInds = np.where(maxregshiftVals == 0.3)[0]
        mrsnrInds = np.where(maxregshiftNRVals == 5)[0]
        snrInds = np.where(snrthreshVals == 1.2)[0]
        
        defaultInds = reduce(np.intersect1d, (nrInds, mrsInds, mrsnrInds, snrInds))
        wm = wellMatchedList[np.intersect1d(np.where(mouseList == mouse)[0], [i for i, pl in enumerate(planesList) if pi in pl])[0]]
        nm = nonMatchedList[np.intersect1d(np.where(mouseList == mouse)[0], [i for i, pl in enumerate(planesList) if pi in pl])[0]]
        fig, ax = plt.subplots(1,3, sharey=True, figsize=(10,5))
        for j, bs in enumerate(bsList):
            bsInds = np.where(block_sizeVals == bs)[0]
            testInd = np.intersect1d(defaultInds, bsInds)[0]
            testNum = testNums[testInd]
            defaultDir = f'{mouseDir}plane_{pi}/test{testNum}/plane0/'
            opsfn = f'{defaultDir}ops.npy'
            ops = np.load(opsfn, allow_pickle = True).item()
            
            fpsession = ops['framesPerSession']
            numSessions = len(ops['testFileList'])
            binfn = f'{defaultDir}data.bin'
            
            perSessionMeanImg = np.zeros((numSessions, Ly,Lx))
            with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
                for i in range(numSessions):
                    inds = np.arange(i*fpsession,(i+1)*fpsession)
                    frames = f.ix(indices=inds).astype(np.float32)
                    perSessionMeanImg[i,:,:] = clahe_each(frames.mean(axis=0))[topMargin:bottomMargin, leftMargin:rightMargin]
            if (mi == 8) & (pi in errorPlanes):
                sessionNums = [int(s.split('_')[1]) for s in ops['testSessionNames']]
                errorInds = [si for si, sn in enumerate(sessionNums) if sn in errorSessions]
                if len(errorInds) > 0:
                    perSessionMeanImg = np.delete(perSessionMeanImg, errorInds, axis=0)
                    numSessions = perSessionMeanImg.shape[0]
                
            phaseCorr = np.zeros(numSessions)
            imgCorr = np.zeros(numSessions)
            refSession = refSessions[mi]
            refSname = f'{mouse:03}_{refSession:03}_'
            if (mi == 8) & (pi in errorPlanes):
                sessionNums = [int(s.split('_')[1]) for s in ops['testSessionNames']]
                errorInds = [si for si, sn in enumerate(sessionNums) if sn in errorSessions]
                if len(errorInds) > 0:
                    testSessionNames = [tsn for i, tsn in enumerate(ops['testSessionNames']) if i not in errorInds]
                else:
                    testSessionNames = ops['testSessionNames']
            else:
                testSessionNames = ops['testSessionNames']
            refSi = [i for i,fn in enumerate(testSessionNames) if refSname in fn]
            refImg = perSessionMeanImg[refSi[0],:,:]
            for i in range(numSessions):
                img1 = perSessionMeanImg[i,:,:]
                _, _, _, phaseCorr[i], _ = phase_corr(img1, refImg)
            ax[0].plot(np.delete(phaseCorr, refSi[0]), color=clist[j])
            ax[1].plot(phaseCorr[wm], color=clist[j])
            ax[2].plot(phaseCorr[nm], color=clist[j])
        ax[2].legend([f'Block size [{bs},{bs}]' for bs in bsList] )
        ax[0].set_title('All sessions')
        ax[1].set_title('Well-matched sessions')
        ax[2].set_title('Non-matched sessions')
        ax[0].set_ylabel('Phase correlation')
        ax[1].set_xlabel('Session Index')
        fig.suptitle(f'JK{mouse:03} plane {pi}')
        fig.tight_layout()
        
'''
Having smaller block size is always better.
But to what limit?
What about time?
'''
#%% Registration time duration
from functools import reduce
testNums = [*range(111), *range(200,224)]
mrs = 0.3
mrsNR = 5
snrThresh = 1.2
bsList = [128,96,64,48,32]
clist = ['k', 'b', 'c', 'g', 'y']
regTimes = np.zeros((3,8,5))

for i, mi in enumerate([0,3,8]):
    mouse = mice[mi]
    mouseDir = f'{h5Dir}{mouse:03}/'
    for j, pi in enumerate(range(1,9)):
        ResultFn = f'{mouseDir}plane_{pi}/regParamTestResult2.npy'
        result = np.load(ResultFn, allow_pickle = True).item()
               
        nrVals = result['nonrigidVals']
        maxregshiftVals = result['maxregshiftVals']
        maxregshiftNRVals = result['maxregshiftNRVals']
        block_sizeVals = result['blocksizeVals']
        snrthreshVals = result['snrthreshVals']
        
        nrInds = np.where(nrVals==True)[0]
        mrsInds = np.where(maxregshiftVals == 0.3)[0]
        mrsnrInds = np.where(maxregshiftNRVals == 5)[0]
        snrInds = np.where(snrthreshVals == 1.2)[0]
        
        defaultInds = reduce(np.intersect1d, (nrInds, mrsInds, mrsnrInds, snrInds))
        for k, bs in enumerate(bsList):
            bsInds = np.where(block_sizeVals == bs)[0]
            testInd = np.intersect1d(defaultInds, bsInds)[0]
            testNum = testNums[testInd]
            defaultDir = f'{mouseDir}plane_{pi}/test{testNum}/plane0/'
            opsfn = f'{defaultDir}ops.npy'
            ops = np.load(opsfn, allow_pickle = True).item()
            
            if not ((mi==8) & ((pi >= 3) & (pi <= 5))):
                regTimes[i,j,k] = ops['timing']['total_plane_runtime']
#%%
f, ax = plt.subplots(1,3, figsize=(10,5))
for i in range(2):
    for j in range(8):
        ax[i].plot(regTimes[i,j,:])
    ax[i].set_xticks(range(5))
    ax[i].set_xticklabels(bsList)
    ax[i].set_title(f'Mouse #{i}')
i = 2
for j in range(5,8):
    ax[i].plot(regTimes[i,j,:])
    ax[i].set_xticks(range(5))
    ax[i].set_xticklabels(bsList)
    ax[i].set_title(f'Mouse #{i}')
ax[0].legend([f'plane {pi}' for pi in range(1,9)])
ax[2].legend([f'plane {pi}' for pi in range(6,9)])
ax[1].set_xlabel('Block sizes')
ax[0].set_ylabel('Runtime (s)')
#%%
normRT = np.zeros((3,8,5))
for i in range(3):
    for j in range(8):
        normRT[i,j,:] = regTimes[i,j,:]/regTimes[i,j,0]
f, ax = plt.subplots(1,3, figsize=(10,5), sharey=True)
for i in range(2):
    for j in range(8):
        ax[i].plot(normRT[i,j,:])
    ax[i].set_xticks(range(5))
    ax[i].set_xticklabels(bsList)
    ax[i].set_title(f'Mouse #{i}')
i = 2
for j in range(5,8):
    ax[i].plot(normRT[i,j,:])
    ax[i].set_xticks(range(5))
    ax[i].set_xticklabels(bsList)
    ax[i].set_title(f'Mouse #{i}')
ax[0].legend([f'plane {pi}' for pi in range(1,9)])
ax[2].legend([f'plane {pi}' for pi in range(6,9)])
ax[1].set_xlabel('Block sizes')
ax[0].set_ylabel('Runtime (normalized)')

#%%
temp = np.reshape(normRT, (-1,5))
errorInds = [16,17,18,19,20]
temp = np.delete(temp, errorInds, axis=0)
err = np.std(temp, axis=0)
mean = np.mean(temp, axis=0)
f, ax = plt.subplots() 
ax.errorbar(range(5), mean, err)
ax.set_xticks(range(5))
ax.set_xticklabels(bsList)
ax.set_ylabel('Normalized runtime')
ax.set_xlabel('Block size')
ax.fill_between(range(5), mean+err, mean-err, color='gray', edgecolor='none', alpha=0.2)



#%% How do these smaller block sizes look like? Do they really match well?
# Compare between different block sizes per plane
mi = 0
pn = 5
matchedInds = [0,3,6,11,18,23]
testNums = [*range(111), *range(200,224)]
block_sizes = [128,96,64,48,32]
mrs = 0.3
snrThresh = 1.2
mnr = 5
mouse = mice[mi]
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
resultFn = f'{planeDir}regParamTestResult2.npy'
result = np.load(resultFn, allow_pickle=True).item()

nrInds = np.where(result['nonrigidVals'] == True)
mrsInds = np.where(result['maxregshiftVals'] == mrs)
snrInds = np.where(result['snrthreshVals'] == snrThresh)
mnrInds = np.where(result['maxregshiftNRVals'] == mnr)
testInds = reduce(np.intersect1d, (nrInds, mrsInds, snrInds, mnrInds))

viewer = napari.Viewer()

for bs in block_sizes:
    tempInd = np.intersect1d(np.where(result['blocksizeVals']==bs), testInds)[0]
    tn = testNums[tempInd]
    testDir = f'{planeDir}test{tn}/plane0/'
    ops = np.load(f'{testDir}ops.npy', allow_pickle=True).item()
    refSession = refSessions[mi]
    refSname = f'{mouse:03}_{refSession:03}_'
    refSi = [i for i,fn in enumerate(ops['testSessionNames']) if refSname in fn]
    fpsession = ops['framesPerSession']
    numSessions = len(ops['testFileList'])
    perSessionMeanImg = np.zeros((numSessions,ops['Ly'], ops['Lx']))
    normImg = np.zeros((numSessions,ops['Ly'], ops['Lx'])) # normalized mean images
    compImg = np.zeros((numSessions,ops['Ly'], ops['Lx'], 3)) # composite normImgs, with the ref session normImg
    binfn = f'{testDir}data.bin'
    with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
        for i in range(numSessions):
            inds = np.arange(i*fpsession,(i+1)*fpsession)
            frames = f.ix(indices=inds).astype(np.float32)
            mimg = frames.mean(axis=0)
            perSessionMeanImg[i,:,:] = mimg
            mimg1 = np.percentile(mimg, 1)
            mimg99 = np.percentile(mimg, 99)
            mimg = (mimg - mimg1) / (mimg99 - mimg1)
            mimg = np.maximum(0, np.minimum(1, mimg))
            normImg[i,:,:] = mimg
    # viewer.add_image(normImg[matchedInds,:,:], name=f'block size [{bs}, {bs}]')
    refNormImg = normImg[refSi[0],:,:]
    for i in range(numSessions):
        compImg[i,:,:,0] = refNormImg
        compImg[i,:,:,2] = refNormImg
        compImg[i,:,:,1] = normImg[i,:,:]
    viewer.add_image(compImg[matchedInds,:,:,:], rgb = True, name=f'block size [{bs}, {bs}]')
'''
Composite image is not a good way to visually confirm FOV match
Also, little vasculature changes might be OK as long as cell bodies match well
It might be better to watch transition from the reference image to each session (moving images)
'''
#%% Visual insepction with moving image
# Has to be done in each condition
# Each napari layer is for each session mean image
# Just 2 time points to easily move back and forth between the ref image and each session mean image
mi = 3
pn = 2
# matchedInds = [0,3,6,11,18,23]

bs = 128 # 128, 96, 64, 48, 32
# block_sizes = [128,96,64,48,32]
testNums = [*range(111), *range(200,224)]
mrs = 0.3
snrThresh = 1.2
mnr = 5
mouse = mice[mi]
planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
resultFn = f'{planeDir}regParamTestResult2.npy'
result = np.load(resultFn, allow_pickle=True).item()

nrInds = np.where(result['nonrigidVals'] == True)
mrsInds = np.where(result['maxregshiftVals'] == mrs)
snrInds = np.where(result['snrthreshVals'] == snrThresh)
mnrInds = np.where(result['maxregshiftNRVals'] == mnr)
testInds = reduce(np.intersect1d, (nrInds, mrsInds, snrInds, mnrInds))

tempInd = np.intersect1d(np.where(result['blocksizeVals']==bs), testInds)[0]
tn = testNums[tempInd]
testDir = f'{planeDir}test{tn}/plane0/'
ops = np.load(f'{testDir}ops.npy', allow_pickle=True).item()
refSession = refSessions[mi]
refSname = f'{mouse:03}_{refSession:03}_'
refSi = [i for i,fn in enumerate(ops['testSessionNames']) if refSname in fn]
fpsession = ops['framesPerSession']
numSessions = len(ops['testFileList'])
perSessionMeanImg = np.zeros((numSessions,ops['Ly'], ops['Lx']))
normImg = np.zeros((numSessions,ops['Ly'], ops['Lx'])) # normalized mean images
claheImg = np.zeros((numSessions,ops['Ly'], ops['Lx'])) # normalized mean images
# compImg = np.zeros((numSessions,ops['Ly'], ops['Lx'], 3)) # composite normImgs, with the ref session normImg
binfn = f'{testDir}data.bin'
with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
    for i in range(numSessions):
        inds = np.arange(i*fpsession,(i+1)*fpsession)
        frames = f.ix(indices=inds).astype(np.float32)
        mimg = frames.mean(axis=0)
        perSessionMeanImg[i,:,:] = mimg
        # mimg1 = np.percentile(mimg, 1)
        # mimg99 = np.percentile(mimg, 99)
        # mimg = (mimg - mimg1) / (mimg99 - mimg1)
        # mimg = np.maximum(0, np.minimum(1, mimg))
        # normImg[i,:,:] = mimg
        claheImg[i,:,:] = clahe_each(mimg)

# viewer.add_image(normImg[matchedInds,:,:], name=f'block size [{bs}, {bs}]')
compareRefEachSession = np.zeros((numSessions,2,ops['Ly'],ops['Lx']))
compareRefEachSessionClahe = np.zeros((numSessions,2,ops['Ly'],ops['Lx']))
# compareRefEachSessionNorm = np.zeros((numSessions,2,ops['Ly'],ops['Lx']))
refImg = perSessionMeanImg[refSi[0],:,:]
refClaheImg = claheImg[refSi[0],:,:]
# refNormImg = normImg[refSi[0],:,:]

refPhaseCorr = np.zeros(numSessions)
refPhaseCorrClahe = np.zeros(numSessions)
refPixCorr = np.zeros(numSessions)
refPixCorrClahe = np.zeros(numSessions)
for i in range(numSessions):
    # tempImg = np.zeros((2, ops['Ly'], ops['Lx']))
    # tempImg[0,:,:] = refNormImg
    # tempImg[1,:,:] = normImg[i,:,:]
    # sessionName = ops['testSessionNames'][i]
    # viewer.add_image(tempImg, name=f'session Ind {i}')
    compareRefEachSession[i,0,:,:] = refImg
    tempImg = perSessionMeanImg[i,:,:]
    compareRefEachSession[i,1,:,:] = tempImg
    
    compareRefEachSessionClahe[i,0,:,:] = refClaheImg
    tempImgClahe = claheImg[i,:,:]
    compareRefEachSessionClahe[i,1,:,:] = tempImgClahe
    # compareRefEachSessionNorm[i,0,:,:] = refNormImg
    # compareRefEachSessionNorm[i,1,:,:] = normImg[i,:,:]
    
    _, _, _, refPhaseCorr[i], _ = phase_corr(refImg, tempImg)
    refPixCorr[i] = np.corrcoef(refImg.flatten(), tempImg.flatten())[0,1]
    _, _, _, refPhaseCorrClahe[i], _ = phase_corr(refClaheImg, tempImgClahe)
    refPixCorrClahe[i] = np.corrcoef(refClaheImg.flatten(), tempImgClahe.flatten())[0,1]

viewer = napari.Viewer()
viewer.add_image(compareRefEachSession)
viewer.add_image(compareRefEachSessionClahe)

#%%
f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(12,6))
ax0.plot(refPhaseCorrClahe, 'o-')
tempMax = np.max([corval for corval in refPhaseCorrClahe if corval < 1-1e-10]) * 1.1
tempMin = np.min(refPhaseCorrClahe) * 0.9
ax0.set_ylim(tempMin, tempMax)
ax0.set_xticks(np.arange(0,numSessions,3))
ax0.set_ylabel('Contrast adjusted')
ax0.set_title('Phase correlation')

ax1.plot(refPixCorrClahe, 'o-')
tempMax = np.max([corval for corval in refPixCorrClahe if corval < 1-1e-10]) * 1.1
tempMin = np.min(refPixCorrClahe) * 0.9
ax1.set_ylim(tempMin, tempMax)
ax1.set_xticks(np.arange(0,numSessions,3))
ax1.set_title('Pixel correlation')

ax2.plot(refPhaseCorr, 'o-')
tempMax = np.max([corval for corval in refPhaseCorr if corval < 1-1e-10]) * 1.1
tempMin = np.min(refPhaseCorr) * 0.9
ax2.set_ylim(tempMin, tempMax)
ax2.set_xticks(np.arange(0,numSessions,3))
ax2.set_ylabel('Raw mean image')
ax2.set_xlabel('Session index')

ax3.plot(refPixCorr, 'o-')
tempMax = np.max([corval for corval in refPixCorr if corval < 1-1e-10]) * 1.1
tempMin = np.min(refPixCorr) * 0.9
ax3.set_ylim(tempMin, tempMax)
ax3.set_xticks(np.arange(0,numSessions,3))

f.suptitle(f'JK{mouse:03} plane {pn}')
f.tight_layout()

'''
Blurring in each session
Maybe registering each frame to the reference session image is not good.
'''