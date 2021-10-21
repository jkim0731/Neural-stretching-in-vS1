"""
Estimate depth compared to z-stack imaging
After registration

Check within-session drift

Check across-session drift

Using zstackReg .mat file

2021/10/12 JK
"""

#%% BS

import scipy.io
import numpy as np
import napari
import os, glob
from skimage import exposure
import matplotlib.pyplot as plt

from pystackreg import StackReg

from skimage.registration import phase_cross_correlation
from skimage.transform import rotate, warp_polar
from skimage.filters import difference_of_gaussians
# from scipy.fftpack import fft2, fftshift
from scipy.ndimage import fourier_shift

def phase_corr(fixed, moving, transLim = 0):
    # apply np.roll(moving, (ymax, xmax), axis=(0,1)) to match moving to fixed
    # or, np.roll(fixed, (-ymax, -xmax), axis=(0,1))
    # if fixed.shape != moving.shape:
    #     raise('Dimensions must match')
    R = np.fft.fft2(fixed) * np.fft.fft2(moving).conj()
    R /= np.absolute(R)
    r = np.absolute(np.fft.ifft2(R))
    if transLim > 0:
        r = np.block([[r[-transLim:,-transLim:], r[-transLim:, :transLim+1]],
                     [r[:transLim+1,-transLim:], r[:transLim+1, :transLim+1]]]
            )
        ymax, xmax = np.unravel_index(np.argmax(r), (transLim*2 + 1, transLim*2 + 1))
        ymax, xmax = ymax - transLim, xmax - transLim
        cmax = np.amax(r)
        center = r[transLim, transLim]
    else:
        ymax, xmax = np.unravel_index(np.argmax(r), r.shape)
        cmax = np.amax(r)
        center = r[0, 0]
    return ymax, xmax, cmax, center, r

# ymax, xmax,_,_,_ = phase_corr(sessionImgsDiv[0], beforeImg)
# beforeReg = np.roll(beforeImg, (ymax, xmax), axis=(0,1))

def clahe_each(img: np.float64, kernel_size = None, clip_limit = 0.01, nbins = 2**16):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    newimg = exposure.equalize_adapthist(newimg, kernel_size = kernel_size, clip_limit = clip_limit, nbins=nbins)    
    return newimg

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

def rigid_reg(fixed, moving, upsample_factor=100):
    fixed = np.clip(fixed, np.percentile(fixed, 1), np.percentile(fixed, 99))
    moving = np.clip(moving, np.percentile(moving, 1), np.percentile(moving, 99))
    radius = fixed.shape[0] //2

    warp_fixed = warp_polar(fixed, radius = radius)
    warp_moving = warp_polar(moving, radius = radius)
    shiftsRotate, _, _ = phase_cross_correlation(warp_fixed[:,:180], warp_moving[:,:180], upsample_factor=upsample_factor)
    regImgRot = rotate(moving, shiftsRotate[0])

    shiftsTrans, _, _ = phase_cross_correlation(fixed, regImgRot, upsample_factor=upsample_factor)
    regImgRotReg = np.fft.ifftn(fourier_shift(np.fft.fftn(regImgRot), shiftsTrans)).real

    return regImgRotReg, shiftsRotate, shiftsTrans

h5Dir = 'I:/'
zstackDir = 'C:/JK/2020 Neural stretching in S1/Data/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]

#%%
mi = 1
mouse = mice[mi]
pn = 4
refSession = refSessions[mi]

roughEstDepth = [100, 200]
#%% Load z-stack file - already registered and contrast-adjusted
zregFnList = glob.glob(f'{zstackDir}zstackReg_{mouse:03}_*.mat')
# zregFn = 'C:/JK/2020 Neural stretching in S1/Data/zstackReg_027_1000.mat'
zregFn = zregFnList[0]
mat = scipy.io.loadmat(zregFn)

zstackReg = np.moveaxis(mat['zstackReg'], -1, 0)
zstackDepths = mat['zstackDepths']

#%%
# napari.view_image(zstackReg)

#%% Loading registered mean images
sessionReg = np.load(f'{h5Dir}/{mouse}/plane_{pn}/s2p_nr_reg.npy', allow_pickle=True).item()

#%% Find the best reg parameters for the reference image and the best-matching depth
refSi = [i for i, sn in enumerate(sessionReg['sessionNames']) if f'{refSession:03}' in sn[4:7]][0]
refMeanImg = sessionReg['regImgs'][refSi,:,:]
refmimgClahe = clahe_each(refMeanImg)
refmimgNorm = np.clip(refMeanImg, np.percentile(refMeanImg,1), np.percentile(refMeanImg,99))
refmimgNorm = (refmimgNorm-np.amin(refmimgNorm)) / (np.amax(refmimgNorm) - np.amin(refmimgNorm))
#%%
plt.imshow(refmimgNorm, cmap='gray')
#%%
roughEstInds = np.arange(np.where(zstackDepths>max(roughEstDepth))[0][-1],
                         np.where(zstackDepths<min(roughEstDepth))[0][0],
                         dtype=int)
roughMean = zstackReg[roughEstInds,:,:].mean(axis=0)
yPixStart = max((roughMean.shape[0] - refMeanImg.shape[0])//2, 0)
xPixStart = max((roughMean.shape[1] - refMeanImg.shape[1])//2, 0)
roughMeanTrim = roughMean[yPixStart:yPixStart + refMeanImg.shape[0],
                          xPixStart:xPixStart + refMeanImg.shape[1]]
# ymax, xmax, _, _, _ = phase_corr(refmimgClahe, roughMeanTrim, 100)
# roughMeanTrimMoved = np.roll(roughMeanTrim, (ymax,xmax), axis=(0,1))
# fig, ax = plt.subplots(2,2)
# ax[0,0].imshow(refmimgClahe, cmap='gray')
# ax[0,1].imshow(roughMeanTrim, cmap='gray')
# ax[1,0].imshow(roughMeanTrimMoved, cmap='gray')
# ax[1,1].imshow(imblend(refmimgClahe, roughMeanTrimMoved))

ymax, xmax, _, _, _ = phase_corr(refmimgNorm, roughMeanTrim, 150)
roughMeanTrimMoved = np.roll(roughMeanTrim, (ymax,xmax), axis=(0,1))
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(refmimgNorm, cmap='gray')
ax[0,1].imshow(roughMeanTrim, cmap='gray')
ax[1,0].imshow(roughMeanTrimMoved, cmap='gray')
ax[1,1].imshow(imblend(refmimgNorm, roughMeanTrimMoved))


#%%

yPixStart = max((zstackReg.shape[1] - refMeanImg.shape[0])//2, 0)
xPixStart = max((zstackReg.shape[2] - refMeanImg.shape[1])//2, 0)
zstackRegTrim = zstackReg[:, yPixStart:yPixStart + refMeanImg.shape[0],
                          xPixStart:xPixStart + refMeanImg.shape[1]]

#%%
ymaxAll = np.zeros(len(zstackDepths))
xmaxAll = np.zeros(len(zstackDepths))
cmaxAll = np.zeros(len(zstackDepths))
for zi in range(len(zstackDepths)):
    ymaxAll[zi], xmaxAll[zi], cmaxAll[zi], _, _ = phase_corr(zstackRegTrim[zi,:,:], refmimgClahe)


#%%
smoothWindow = 10
cmaxSmooth = np.zeros(len(cmaxAll))
for ci in range(len(cmaxAll)):
    tempInds = np.arange(max(0,ci-smoothWindow//2), min(len(cmaxAll), ci+smoothWindow//2), dtype=int)
    cmaxSmooth[ci] = cmaxAll[tempInds].mean()

#%%
fig, ax = plt.subplots()
ax.plot(cmaxAll)
ax.plot(cmaxSmooth)





#%% Load z-stack file - nonregistered, non-contrast-adjusted
zregFnList = glob.glob(f'{zstackDir}zstack_{mouse:03}_*.mat')
zregFn = zregFnList[0]
mat = scipy.io.loadmat(zregFn)

zstack = np.moveaxis(mat['zstack'], -1, 0)

# napari.view_image(zstack)


#%% Straight-ahead registration with reference mean image
yPixStart = max((zstack.shape[1] - refMeanImg.shape[0])//2, 0)
xPixStart = max((zstack.shape[2] - refMeanImg.shape[1])//2, 0)
zstackTrim = zstack[:, yPixStart:yPixStart + refMeanImg.shape[0],
                          xPixStart:xPixStart + refMeanImg.shape[1]]

ymaxAll = np.zeros(len(zstackDepths))
xmaxAll = np.zeros(len(zstackDepths))
cmaxAll = np.zeros(len(zstackDepths))
phaseCorrAll = np.zeros(len(zstackDepths))
for zi in range(len(zstackDepths)):
    # ymaxAll[zi], xmaxAll[zi], cmaxAll[zi], _, _ = phase_corr(zstackTrim[zi,:,:], refMeanImg)
    ymax, xmax, cmaxAll[zi], _, _ = phase_corr(zstackTrim[zi,:,:], refMeanImg)
    ymaxAll[zi] = ymax
    xmaxAll[zi] = xmax
    refReg = np.roll(refMeanImg, (ymax, xmax), axis=(0,1))



smoothWindow = 10
cmaxSmooth = np.zeros(len(cmaxAll))
for ci in range(len(cmaxAll)):
    tempInds = np.arange(max(0,ci-smoothWindow//2), min(len(cmaxAll), ci+smoothWindow//2), dtype=int)
    cmaxSmooth[ci] = cmaxAll[tempInds].mean()
fig, ax = plt.subplots()
ax.plot(cmaxAll)
ax.plot(cmaxSmooth)


#%%


napari.view_image(np.array(sessionReg['regImgs']))



#%% Register z-stack imaging, without contrast adjustment
nstack = zstack.shape[0]
zstackRegNew = np.zeros(zstack.shape)
zstackRegNew[nstack-1,:,:] = zstack[nstack-1,:,:].copy()
for zi in np.arange(nstack-2, -1, -1, dtype=int):
    ymax, xmax, _, _, _ = phase_corr(zstackRegNew[zi+1,:,:], zstack[zi,:,:])
    zstackRegNew[zi,:,:] = np.roll(zstack[zi,:,:].copy(), (ymax, xmax), axis=(0,1))

napari.view_image(zstackRegNew)






#%% load non-registered mean images
sessionNames = get_session_names(f'{h5Dir}{mouse:03}/plane_{pn}/', mouse, pn)
trainingSessionNames = [sn[4:] for sn in sessionNames if len(sn)==7]
meanImgList = []
for sn in trainingSessionNames:
    ops = np.load(f'{h5Dir}{mouse:03}/plane_{pn}/{sn}/plane0/ops.npy', allow_pickle=True).item()
    meanImgList.append(ops['meanImg'])

# napari.view_image(np.array(meanImgList))
#%%
topRingingPix = 80
meanImgTrimList = [mimg[topRingingPix:,:] for mimg in meanImgList]

napari.view_image(np.array(meanImgTrimList))


#%%

mimgRegList = []
fixed = meanImgTrimList[0].copy()
for img in meanImgTrimList:
    moving = img.copy()
    moved, _, _ = rigid_reg(clahe_each(fixed), clahe_each(moving), 100)
    mimgRegList.append(moved)
napari.view_image(np.array(mimgRegList))



#%%
sr = StackReg(StackReg.RIGID_BODY)
mimgRegList = []
fixed = meanImgTrimList[0].copy()
for img in meanImgTrimList:
    moving = img.copy()
    moved = sr.register_transform(fixed, moving)
    mimgRegList.append(moved)
napari.view_image(np.array(mimgRegList))





#%% Set a loose range from visual inspection

napari.view_image(sr.register_transform_stack(np.array(meanImgTrimList)))


'''
StackReg works great.
Try using this for the registration between z-stack and the reference mean image
'''


# 211013
#%% Register reference mean image to the z-stack using StackReg
# Use rigid body (do not need to be perfect)
# Quantify intensity correlation as a measure of the match.

#%% First, start with some visually-matching depth

refImg = meanImgTrimList[2]

#%% Register z-stack imaging, using StackReg
zregFnList = glob.glob(f'{zstackDir}zstack_{mouse:03}_*.mat')
zregFn = zregFnList[0]
mat = scipy.io.loadmat(zregFn)

zstack = np.moveaxis(mat['zstack'], -1, 0)

zstackRegNew = sr.register_transform_stack(zstack, reference='previous')

napari.view_image(zstackRegNew)

'''
It works great.
'''
#%%


#%% Register the reference image to a plane from the registered z-stack
# Image size should match
# (1) Padding 0's to the reference image -> Doesn't work.
# Seems like 0's matter too much (boundaries are closely aligned)
# (2) Padding NaN's to both images where 0 -> Doesnt' work
# Results in just nan's all over the pixels
# (3) Pick subregion of the sample plane (manually) with the same size as the reference image
# -> Doesn't work. Just terrible matching
# (4) In addition to (3), adjust contrast before matching
# Better matching. But can't be sure if this is really matching (or... can it be matched at all?)

#%% (2)
# tempImg = zstackRegNew[156,:,:]
# tempImg[tempImg==0] = np.nan
# tempMov = np.empty(tempImg.shape)
# tempMov[tempMov==0] = np.nan
# ystartPix = (tempImg.shape[0] - refImg.shape[0]) // 2
# xstartPix = (tempImg.shape[1] - refImg.shape[1]) // 2
# tempMov[ystartPix:ystartPix+refImg.shape[0], xstartPix:xstartPix+refImg.shape[1]] = refImg.copy()
# out = sr.register_transform(tempImg, tempMov)

#%% (3)
tempImg = zstackRegNew[120,200:200+refImg.shape[0],100:100+refImg.shape[1]].copy()
tempMov = refImg.copy()

# (4)
tempImg = clahe_each(tempImg)
tempMovClahe = clahe_each(tempMov)

out = sr.register_transform(tempImg, tempMov)

blended = imblend(tempImg/np.nanmax(tempImg), out/np.nanmax(out))

blended[np.isnan(blended)] = 0
tempImg[np.isnan(tempImg)] = 0
out[np.isnan(out)] = 0

viewer = napari.Viewer()
viewer.add_image(blended)
viewer.add_image(tempImg)
viewer.add_image(out)

#%% (4) -1 apply to the whole stack and find out the best-matched plane
def img_norm(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))
intCorr = np.zeros(zstackRegNew.shape[0])
zstackBlended = np.zeros((zstackRegNew.shape[0], refImg.shape[0], refImg.shape[1], 3))
tempMov = refImg.copy()
tempMovClahe = clahe_each(tempMov)
for i in range(zstackRegNew.shape[0]):
    tempImg = zstackRegNew[i,200:200+refImg.shape[0], 100:100+refImg.shape[1]].copy()
    tempImgClahe = clahe_each(tempImg)
    tmat = sr.register(tempImgClahe, tempMovClahe)
    out = sr.transform(tempMov, tmat)
    intCorr[i] = np.corrcoef(tempImg.flatten(), out.flatten())[0,1]
    zstackBlended[i,:,:,:] = imblend(img_norm(tempImg), img_norm(out))

fig, ax = plt.subplots()
ax.plot(intCorr, 'k-')
# ax.plot(intCorrValid, 'b-')
napari.view_image(zstackBlended)
#%%

'''
Matching with selected region does not work well.
With or without Clahe.
If correlation is calculated without clahe, then lower planes have high correlation values.
'''

#%% (4) -1 Apply to the whole stack
# by averaging a stack of z-stack (10 planes?)
avgNumPlanes = 11
intCorr = np.zeros(zstackRegNew.shape[0])
zstackBlended = np.zeros((zstackRegNew.shape[0], refImg.shape[0], refImg.shape[1], 3))
zstackAvg = np.zeros(zstackRegNew.shape)
nplanes = zstackRegNew.shape[0]
tempMov = refImg.copy()
for i in range(nplanes):
    startPlaneNum = max(0, i - avgNumPlanes//2)
    endPlaneNum = min(nplanes, i + avgNumPlanes//2)
    zstackAvg[i,:,:] = zstackRegNew[startPlaneNum:endPlaneNum, :, :].mean(axis=0)

    tempImg = zstackAvg[i,200:200+refImg.shape[0], 100:100+refImg.shape[1]].copy()
    # out = sr.register_transform(tempImg, tempMov)

    tempImgClahe = clahe_each(tempImg)
    tempMovClahe = clahe_each(tempMov)
    tmat = sr.register(tempImgClahe, tempMovClahe)
    out = sr.transform(tempMovClahe, tmat)

    intCorr[i] = np.corrcoef(tempImgClahe.flatten(), out.flatten())[0,1]
    zstackBlended[i,:,:,:] = imblend(img_norm(tempImg), img_norm(out))

fig, ax = plt.subplots()
ax.plot(intCorr, 'k-')
napari.view_image(zstackBlended)

# zstackAvg[np.isnan(zstackAvg)] = 0
# napari.view_image(zstackAvg)

'''
Averaging looks better for matching, but does not help much with registration.
As before, raw mean images make huge translation and rotation,
and clahe makes them better positioned but key vasculatures do not match.
'''

#%% Other options
# i) Use different types of registration. Affine or bilinear.
# ii) Use nonrigid registrations - suite2p,
# ii) Emphasize vasculature, neurons, or even binarize the map.
# iii) Use cell-pose cell map

#%% i) Affine and bilinear registrations

sra = StackReg(StackReg.AFFINE)
srb = StackReg(StackReg.BILINEAR)

#%% Using raw mean images
avgNumPlanes = 11
intCorrAffine = np.zeros(zstackRegNew.shape[0])
intCorrBilinear = np.zeros(zstackRegNew.shape[0])
zstackBlendedAffine = np.zeros((zstackRegNew.shape[0], refImg.shape[0], refImg.shape[1], 3))
zstackBlendedBilinear = np.zeros((zstackRegNew.shape[0], refImg.shape[0], refImg.shape[1], 3))
zstackAvg = np.zeros(zstackRegNew.shape)
nplanes = zstackRegNew.shape[0]
tempMov = refImg.copy()
for i in range(nplanes):
    startPlaneNum = max(0, i - avgNumPlanes//2)
    endPlaneNum = min(nplanes, i + avgNumPlanes//2)
    zstackAvg[i,:,:] = np.nanmean(zstackRegNew[startPlaneNum:endPlaneNum, :, :], axis=0)

    tempImg = zstackAvg[i,200:200+refImg.shape[0], 100:100+refImg.shape[1]].copy()
    outAffine = sra.register_transform(tempImg, tempMov)
    outBilinear = srb.register_transform(tempImg, tempMov)

    intCorrAffine[i] = np.corrcoef(tempImgClahe.flatten(), outAffine.flatten())[0,1]
    zstackBlendedAffine[i,:,:,:] = imblend(img_norm(tempImg), img_norm(outAffine))

    intCorrBilinear[i] = np.corrcoef(tempImgClahe.flatten(), outBilinear.flatten())[0,1]
    zstackBlendedBilinear[i,:,:,:] = imblend(img_norm(tempImg), img_norm(outBilinear))

fig, ax = plt.subplots()
ax.plot(intCorrAffine, 'k-', label='Affine')
ax.plot(intCorrBilinear, 'b-', label='Bilinear')
ax.legend()
ax.set_title('Raw mean images')
viewer = napari.Viewer()
viewer.add_image(zstackBlendedAffine, name='Affine')
viewer.add_image(zstackBlendedBilinear, name='Bilinear')

'''
Terrible registration
'''

#%% Using clahe mean images
avgNumPlanes = 11
intCorrAffine = np.zeros(zstackRegNew.shape[0])
intCorrBilinear = np.zeros(zstackRegNew.shape[0])
zstackBlendedAffine = np.zeros((zstackRegNew.shape[0], refImg.shape[0], refImg.shape[1], 3))
zstackBlendedBilinear = np.zeros((zstackRegNew.shape[0], refImg.shape[0], refImg.shape[1], 3))
zstackAvg = np.zeros(zstackRegNew.shape)
nplanes = zstackRegNew.shape[0]
tempMov = refImg.copy()
tempMovClahe = clahe_each(tempMov)

for i in range(nplanes):
    startPlaneNum = max(0, i - avgNumPlanes//2)
    endPlaneNum = min(nplanes, i + avgNumPlanes//2)
    zstackAvg[i,:,:] = np.nanmean(zstackRegNew[startPlaneNum:endPlaneNum, :, :], axis=0)

    tempImg = zstackAvg[i,200:200+refImg.shape[0], 100:100+refImg.shape[1]].copy()
    tempImgClahe = clahe_each(tempImg)
    tmat = sra.register(tempImgClahe, tempMovClahe)
    outAffine = sra.transform(tempMovClahe, tmat)

    tmat = srb.register(tempImgClahe, tempMovClahe)
    outBilinear = srb.transform(tempMovClahe, tmat)

    intCorrAffine[i] = np.corrcoef(tempImgClahe.flatten(), outAffine.flatten())[0,1]
    zstackBlendedAffine[i,:,:,:] = imblend(img_norm(tempImg), img_norm(outAffine))

    intCorrBilinear[i] = np.corrcoef(tempImgClahe.flatten(), outBilinear.flatten())[0,1]
    zstackBlendedBilinear[i,:,:,:] = imblend(img_norm(tempImg), img_norm(outBilinear))

fig, ax = plt.subplots()
ax.plot(intCorrAffine, 'k-', label='Affine')
ax.plot(intCorrBilinear, 'b-', label='Bilinear')
ax.legend()
ax.set_title('Raw mean images')
viewer = napari.Viewer()
viewer.add_image(zstackBlendedAffine, name='Affine')
viewer.add_image(zstackBlendedBilinear, name='Bilinear')

'''
Better than raw, but not matched.
It's curious... What is this trying to match? Vasculatures are definitely off.
'''












''' CHANGE OF STRATEGY
2021/10/18
Found out best-matching depth by eye (~145)
First, to include all matching FOV, register z-stack from top-down (not bottom-up like before)
Then, try best method to match 11-plane averaged z-stack image with the reference mean image
TurboReg, simpleITK, suite2p, using raw, clahe (adjust kernel), and cellpose image

Start a new py file.

'''











