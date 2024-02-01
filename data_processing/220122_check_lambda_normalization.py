# -*- coding: utf-8 -*-
"""
Test lambda
Lambda (pixel weights) is supposed to be normalized, but found not normalized
in some sessions.
- Check what could have made this difference.
- Check the effect of lambda in F, Fneu, and spks

2022/01/22 JK
"""

import numpy as np
from matplotlib import pyplot as plt
import napari
import copy
import time

from suite2p import extraction


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

#%% Test set loading
h5Dir = 'D:/TPM/JK/h5/' 
egMouse = 25
egPn = 7
egSn = 7
egDir = f'{h5Dir}{egMouse:03}/plane_{egPn}/{egSn:03}/plane0/'

stat = np.load(f'{egDir}stat.npy', allow_pickle=True)
iscell = np.load(f'{egDir}iscell.npy')
ops = np.load(f'{egDir}ops.npy', allow_pickle=True).item()
ops['reg_file'] = f'{egDir}data.bin'

spks = np.load(f'{egDir}spks.npy')
F = np.load(f'{egDir}F.npy')
Fneu = np.load(f'{egDir}Fneu.npy')

meanLam = np.mean([np.sum(s['lam']) for s in stat])
print(f'Mean lambda = {meanLam}')

#%% For those not-normalized lambda, normalize them and recalculate F, Fneu, and spks
# Compare with the original data

# First, run with the original stat
_, teststat, testF, testFneu, _, _ = extraction.create_masks_and_extract(ops, stat)

meanTestLam = np.mean([np.sum(s['lam']) for s in teststat])

Fdifftest = np.any(F - testF)
Fneudifftest = np.any(Fneu - testFneu)

print(f'Is F and testF different? {Fdifftest}')
print(f'Is Fneu and testFneu different? {Fneudifftest}')

#%% Then, run with normalized lambda
normStat = copy.deepcopy(stat)
for s in normStat:
    s['lam'] = s['lam']/np.sum(s['lam'])

meanNormLam = np.mean([np.sum(s['lam']) for s in normStat])
print(f'Mean normalized lambda = {meanNormLam}')

_, normstat, normF, normFneu, _, _ = extraction.create_masks_and_extract(ops, normStat)

meanNormResultLam = np.mean([np.sum(s['lam']) for s in normstat])

Fdiffnorm = np.any(F - normF)
Fneudiffnorm = np.any(Fneu - normFneu)

print(f'Is F and normF different? {Fdiffnorm}')
print(f'Is Fneu and normFneu different? {Fneudiffnorm}')

'''
Normalizing lambda affects F and Fneu
(Why Fneu?)
'''

#%% Why is Fneu changing by changing weights of the cell?

# Is mask changing?
_, FneuMaskTest = extraction.masks.create_masks(ops, stat)
_, FneuMaskNew = extraction.masks.create_masks(ops, normStat)

FneuMaskDiff = [ np.sum(np.abs(FneuMaskNew[i] - FneuMaskTest[i])) for i in range(len(FneuMaskTest))]
diffAnswer = np.any(FneuMaskDiff)
print(f'Do FneuTest and FneuNorm have different masks? {diffAnswer}')

'''
Masks are the same.
'''

#%% Because of lam_percentile?
'''
Yes, I think so.
No need to test this.
'''


#%% Look at the effect of making lambda even
# Compare between normalized and unnormalized

h5Dir = 'D:/TPM/JK/h5/' 
egMouse = 25
egPn = 3
egSn = 1
egDir = f'{h5Dir}{egMouse:03}/plane_{egPn}/{egSn:03}/plane0/'

stat = np.load(f'{egDir}stat.npy', allow_pickle=True)
iscell = np.load(f'{egDir}iscell.npy')
cinds = np.where(iscell[:,0]==1)[0]
ops = np.load(f'{egDir}ops.npy', allow_pickle=True).item()
ops['reg_file'] = f'{egDir}data.bin'

spks = np.load(f'{egDir}spks.npy')
F = np.load(f'{egDir}F.npy')
Fneu = np.load(f'{egDir}Fneu.npy')

# Calculate mean labmda
meanLam = np.mean([np.sum(s['lam']) for s in stat])
print(f'Mean lambda = {meanLam}')
#%%
# Even out lambda
evenStat = copy.deepcopy(stat)
for s in evenStat:
    s['lam'] = np.ones_like(s['lam'])/len(s['lam'])

# Run extraction
_, evenstat, evenF, evenFneu, _, _ = extraction.create_masks_and_extract(ops, evenStat)

# Run deconvolution
t11=time.time()
dF = evenF.copy() - ops['neucoeff']*evenFneu
dF = extraction.preprocess(
    F=dF,
    baseline=ops['baseline'],
    win_baseline=ops['win_baseline'],
    sig_baseline=ops['sig_baseline'],
    fs=ops['fs'],
    prctile_baseline=ops['prctile_baseline']
)
evenSpks = extraction.oasis(F=dF, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])
dcnvTime = time.time()-t11
print('Deconvolution took %0.2f sec.' % dcnvTime)

#%% Draw plots
ci = 40
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(spks[cinds[ci]], label='Original')
ax.plot(evenSpks[cinds[ci]], label='Even')
ax.legend()
ax.set_xlabel('Frames')
ax.set_ylabel('Inferred spikes (AU)')
ax.set_title(f'JK{egMouse:03} plane {egPn} session {egSn:03}\n mean lambda {meanLam:.2f}\n ROI index {cinds[ci]}')

