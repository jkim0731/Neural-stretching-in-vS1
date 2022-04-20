# -*- coding: utf-8 -*-
"""
What the hell is going on when ROIs merged?
First of all, enhanced mean image gets messed up.
Then, baseline inferred spikes seem much more active.
Check if it is true, and try reversing them if necessary.

!! When merging, suite2p just averages F and Fneu, and then run oasis for spks

2022/01/20 JK
"""

import numpy as np
from matplotlib import pyplot as plt
from suite2p.gui import drawroi
from suite2p import extraction
from suite2p.extraction import preprocess
from suite2p.detection.stats import roi_stats
from suite2p.io import BinaryFile
from scipy import stats
import napari
import glob, os, shutil
import time

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

def jk_masks_and_traces(ops, stat_manual, stat_orig, sessionDir):
    ''' main extraction function
        inputs: ops and stat
        creates cell and neuropil masks and extracts traces
        returns: F (ROIs x time), Fneu (ROIs x time), F_chan2, Fneu_chan2, ops, stat
        F_chan2 and Fneu_chan2 will be empty if no second channel
        
        From suite2p.gui.drawroi.py
    '''
    if 'aspect' in ops:
        dy, dx = int(ops['aspect'] * 10), 10
    else:
        d0 = ops['diameter']
        dy, dx = (d0, d0) if isinstance(d0, int) else d0
    t0 = time.time()
    # Concatenate stat so a good neuropil function can be formed
    stat_all = stat_manual.copy()
    for n in range(len(stat_orig)):
        stat_all.append(stat_orig[n])
    stat_all = roi_stats(stat_all, dy, dx, ops['Ly'], ops['Lx'])
    cell_masks = [
        extraction.masks.create_cell_mask(stat, Ly=ops['Ly'], Lx=ops['Lx'], allow_overlap=ops['allow_overlap']) for stat in stat_all
    ]
    cell_pix = extraction.masks.create_cell_pix(stat_all, Ly=ops['Ly'], Lx=ops['Lx'])
    manual_roi_stats = stat_all[:len(stat_manual)]
    manual_cell_masks = cell_masks[:len(stat_manual)]
    manual_neuropil_masks = extraction.masks.create_neuropil_masks(
        ypixs=[stat['ypix'] for stat in manual_roi_stats],
        xpixs=[stat['xpix'] for stat in manual_roi_stats],
        cell_pix=cell_pix,
        inner_neuropil_radius=ops['inner_neuropil_radius'],
        min_neuropil_pixels=ops['min_neuropil_pixels'],
    )
    print('Masks made in %0.2f sec.' % (time.time() - t0))

    F, Fneu = jk_extract_traces_from_masks(ops, manual_cell_masks, manual_neuropil_masks, sessionDir)

    # compute activity statistics for classifier
    npix = np.array([stat_orig[n]['npix'] for n in range(len(stat_orig))]).astype('float32')
    for n in range(len(manual_roi_stats)):
        manual_roi_stats[n]['npix_norm'] = manual_roi_stats[n]['npix'] / np.mean(npix[:100])  # What if there are less than 100 cells?
        manual_roi_stats[n]['compact'] = 1
        manual_roi_stats[n]['footprint'] = 2
        manual_roi_stats[n]['manual'] = 1  # Add manual key

    # subtract neuropil and compute skew, std from F
    dF = F - ops['neucoeff'] * Fneu
    sk = stats.skew(dF, axis=1)
    sd = np.std(dF, axis=1)

    for n in range(F.shape[0]):
        manual_roi_stats[n]['skew'] = sk[n]
        manual_roi_stats[n]['std'] = sd[n]
        manual_roi_stats[n]['med'] = [np.mean(manual_roi_stats[n]['ypix']), np.mean(manual_roi_stats[n]['xpix'])]

    # dF = F - ops['neucoeff'] * Fneu
    # 2021/09/23 JK for proper spike inference
    dF = preprocess(
                F=dF,
                baseline=ops['baseline'],
                win_baseline=ops['win_baseline'],
                sig_baseline=ops['sig_baseline'],
                fs=ops['fs'],
                prctile_baseline=ops['prctile_baseline']
            )
    spks = extraction.dcnv.oasis(F=dF, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])

    return F, Fneu, spks, ops, manual_roi_stats

def jk_extract_traces_from_masks(ops, cell_masks, neuropil_masks, sessionDir):
    """ extract fluorescence from both channels 
    
    also used in drawroi.py
    
    from suite2p.extraction.extract
    
    """
    F_chan2, Fneu_chan2 = [], []
    with BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'],
                    read_filename=f'{sessionDir}data.bin') as f:    
        F, Fneu, ops = extraction.extract.extract_traces(ops, cell_masks, neuropil_masks, f)
    return F, Fneu

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

h5Dir = 'D:/TPM/JK/h5/' 
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]

#%% Test set loading
h5Dir = 'D:/TPM/JK/h5/' 
egMouse = 25
egPn = 2
egSn = 2
egDir = f'{h5Dir}{egMouse:03}/plane_{egPn}/{egSn:03}/plane0/'

spks = np.load(f'{egDir}spks.npy')
stat = np.load(f'{egDir}stat.npy', allow_pickle=True)
iscell = np.load(f'{egDir}iscell.npy')
ops = np.load(f'{egDir}ops.npy', allow_pickle=True).item()
#%% Find the results of merges

imergeLen = np.array([len(s['imerge']) for s in stat])
mergedList = np.where(imergeLen>0)[0]
#%% Select an example merged ROI
mgi = 0
mergei = mergedList[mgi]
mergedFrom = stat[mergei]['imerge']

#%% Plot inferred spikes
fig, ax = plt.subplots()
ax.plot(spks[mergei,:], 'k-', linewidth=3, label='Merged')
ax.plot(spks[mergedFrom[0],:], label='Before merge ROI #1')
ax.plot(spks[mergedFrom[1],:], label='Before merge ROI#2')
ax.legend()

ax.set_xlabel('Frame')
ax.set_ylabel('Event (AU)')
ax.set_title('Inferred spikes before and after merging')

#%% Draw ROIs
mergedFromMap = np.zeros((ops['Ly'],ops['Lx']),'uint8')
for i in range(len(mergedFrom)):
    mergedFromMap[stat[mergedFrom[i]]['ypix'],stat[mergedFrom[i]]['xpix']] += 1
mergedMap = np.zeros((ops['Ly'],ops['Lx']),'uint8')
mergedMap[stat[mergei]['ypix'],stat[mergei]['xpix']] += 1

blended = imblend_for_napari(mergedFromMap, mergedMap)
napari.view_image(blended)

#%%




#%% Test having even lambda across the merged pixels
# Using 'drawroi.npy' in suite2p package
# Do this only in "test" folder. Because it might change the file

testDir = f'{egDir}test/'
spks = np.load(f'{testDir}spks.npy')
stat = np.load(f'{testDir}stat.npy', allow_pickle=True)
iscell = np.load(f'{testDir}iscell.npy')
ops = np.load(f'{testDir}ops.npy', allow_pickle=True).item()

imergeLen = np.array([len(s['imerge']) for s in stat])
mergedList = np.where(imergeLen>0)[0]
mergedFrom = stat[mergei]['imerge']

if len(mergedList)>0:
    stat0 = []
    for ci in mergedList:
        
        ypix = stat[ci]['ypix']
        xpix = stat[ci]['xpix']
        lam = np.ones(ypix.shape)
        # stat0.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size, 'med': med})
        stat0.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size})
    newstat = np.delete(stat, np.hstack((mergedFrom, mergedList)))
    
    # It takes about 5 min for 223 rois with 116k frames
    F, Fneu, F_chan2, Fneu_chan2, newspks, newops, addstat = drawroi.masks_and_traces(ops, stat0, newstat)


#%% Compare bewteen new spks with previous merged spks
fig, ax = plt.subplots()
ax.plot(spks[mergei,:], 'k-', label='Merged')
ax.plot(newspks[0,:], 'c-', label='re-calculated')
ax.legend()

ax.set_xlabel('Frame')
ax.set_ylabel('Event (AU)')
ax.set_title('Inferred spikes before and after re-calculating merged spks')



#%% Compare between new spks and previous before merging spks
fig, ax = plt.subplots()
ax.plot(newspks[0,:], 'k-', linewidth=3, label='Merged')
ax.plot(spks[mergedFrom[0],:], label='Before merge ROI #1')
ax.plot(spks[mergedFrom[1],:], label='Before merge ROI#2')
ax.legend()

ax.set_xlabel('Frame')
ax.set_ylabel('Event (AU)')
ax.set_title('Inferred spikes before merge vs new spks')




#%% Check enhanced mean image, if the weird one is saved in ops.npy
buDir = f'{egDir}backup/'
oldOps = np.load(f'{buDir}ops.npy', allow_pickle=True).item()
fig, ax = plt.subplots()
ax.imshow(oldOps['meanImgE'])


'''
It's not that meanImgE is saved in the ops.npy!!
It's the way that gui2p is loading it!!
'''


#%% From JK025 to 030
# Fix problems with "merge" sessions
# (1) Exchange spks with new calculation using even lambda
# (2) Remove before-merging from iscell
# (3) Leave a note in ops.npy
# Have a backup to be safe

mergeApplied = []

for mi in range(0,3):
# for mi in [0]:
    mouse = mice[mi]
    for pn in range(1,9):
    # for pn in [2]:
        planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
        snames = [sn[4:] for sn in get_session_names(planeDir, mouse, pn)]
        for si in range(len(snames)):
        # for si in [1]:
            sn = snames[si]
            # First, see if this session has a merged rois
            sessionDir = f'{planeDir}{sn}/plane0/'
            stat = np.load(f'{sessionDir}stat.npy', allow_pickle=True)
            ops = np.load(f'{sessionDir}ops.npy', allow_pickle=True).item()
            if 'imerge' in stat[0].keys():
                imergeLen = np.array([len(s['imerge']) for s in stat])
                mergedList = np.where(imergeLen>0)[0]
                if len(mergedList)>0: # when there are merges
                    if 'mergeCorrected' not in ops.keys() or ops['mergeCorrected']==False:
                        print(f'Processing merged ROIs from JK{mouse} plane {pn} session {sn}.')
                        currName = f'JK{mouse:03} plane {pn} session {sn}'
                        # First of all, make a backup for npy files
                        buDir = f'{sessionDir}backup/'
                        if not os.path.isdir(buDir):
                            os.mkdir(buDir)
                        fnlist = glob.glob(f'{sessionDir}*.npy')
                        for fn in fnlist:
                            shutil.copy2(fn, buDir) # using copy2 instead of copy to preserve metadata (e.g., "Date modified")
                        
                        allMergedFrom = np.concatenate([stat[ci]['imerge'] for ci in mergedList])
                        stat0 = []
                        for ci in mergedList:
                            ypix = stat[ci]['ypix']
                            xpix = stat[ci]['xpix']
                            lam = np.ones(ypix.shape)
                            imerge = stat[ci]['imerge']
                            # stat0.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size, 'med': med})
                            stat0.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size, 'inmerge': 0, 'imerge': imerge})
                        tempstat = np.delete(stat, np.concatenate((allMergedFrom, mergedList)))
                        
                        # It takes about 5 min for 223 rois with 116k frames
                        mgF, mgFneu, mgspks, _, mgstat = jk_masks_and_traces(ops, stat0, tempstat, sessionDir)
                        
                        # Swap data for the merged ROIs
                        # Load npy files
                        print('Loading npy files.')
                        F = np.load(f'{sessionDir}F.npy')
                        Fneu = np.load(f'{sessionDir}Fneu.npy')
                        spks = np.load(f'{sessionDir}spks.npy')
                        iscell = np.load(f'{sessionDir}iscell.npy')
                        
                        F[mergedList,:] = mgF
                        Fneu[mergedList,:] = mgFneu
                        spks[mergedList,:] = mgspks
                        stat[mergedList] = mgstat
                        
                        # Set before merge ROIs as not-cell
                        iscell[allMergedFrom,0] = 0
                        print('Data swapped.')
                        
                        # # Correct enhanced mean image
                        ops['mergeCorrected'] = True
                        # ops = extraction.enhanced_mean_image(ops)
                        # print('Enhanced mean image corrected.')
                        
                        # Save npy files
                        np.save(f'{sessionDir}F.npy', F)
                        np.save(f'{sessionDir}Fneu.npy', Fneu)
                        np.save(f'{sessionDir}spks.npy', spks)
                        np.save(f'{sessionDir}stat.npy', stat)
                        np.save(f'{sessionDir}iscell.npy', iscell)
                        np.save(f'{sessionDir}ops.npy', ops)
                        print('npy files saved.')
                        mergeApplied.append(currName)
                    else:
                        print(f'Merged ROIs already corrected in JK{mouse} plane {pn} session {sn}.')
                else:
                    print(f'JK{mouse} plane {pn} session {sn} has no merged ROIs.')
            else:
                print(f'JK{mouse} plane {pn} session {sn} has no merged ROIs.')
    





#%% lam error
# (Maybe) depending on the suite2p version (0.9.3 vs 0.10.1)
# lam values are different in the order of 10**4
# In later version, lam does not seem to add up to 1, but rather to ~10,000

'''
See 220122_check_lambda_normalization.py

I will normalize and make lambda the same across all pixels within an ROI.
So, there is no reason to care about spk values yet in this process.
Just add the merged ROI.
'''


#%% For error correction
# Copy backups back to the directory.
# To reverse results done with error (e.g., mergedFrom instead of allMergedFrom)
buReversed = []
for mi in range(0,3):
    mouse = mice[mi]
    for pn in range(1,9):
        planeDir = f'{h5Dir}{mouse:03}/plane_{pn}/'
        snames = [sn[4:] for sn in get_session_names(planeDir, mouse, pn)]
        for si in range(len(snames)):
            sn = snames[si]
            sessionDir = f'{planeDir}{sn}/plane0/'
            # If there is a backup directory, it means it has been altered,
            # so change it back by force-copying backup to the session directory
            buDir = f'{sessionDir}backup/'
            if os.path.isdir(buDir):
                currName = f'JK{mouse:03} plane {pn} session {sn}'
                # print(f'Reversing {currName}')
                # fnlist = glob.glob(f'{buDir}*.npy')
                # for fn in fnlist:
                #     shutil.copy2(fn, sessionDir)
                buReversed.append(currName)
print('Done.')
