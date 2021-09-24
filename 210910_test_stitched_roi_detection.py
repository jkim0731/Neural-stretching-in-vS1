# -*- coding: utf-8 -*-
"""
ROI detection from stitched data.bin
Need to make an ops.npy file 
2021/09/10 JK
"""
import numpy as np
from matplotlib import pyplot as plt
from suite2p.run_s2p import run_s2p
# from auto_pre_cur import auto_pre_cur
import os, glob, shutil
from suite2p.io.binary import BinaryFile
from suite2p.registration.register import enhanced_mean_image
from suite2p.registration import rigid, nonrigid
import napari
import gc
gc.enable()

h5Dir = 'D:/TPM/JK/h5/'
s2pDir = 'D:/TPM/JK/s2p/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]

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

def twostep_register(img, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                     rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2):
    frames = img.copy()
    if len(frames.shape) == 2:
        frames = np.expand_dims(frames, axis=0)
    elif len(frames.shape) < 2:
        raise('Dimension of the frames should be at least 2')
    elif len(frames.shape) > 3:
        raise('Dimension of the frames should be at most 3')
    (Ly, Lx) = frames.shape[1:]
    # 1st rigid shift
    frames = np.roll(frames, (-rigid_y1, -rigid_x1), axis=(1,2))
    # 1st nonrigid shift
    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size1)
    ymax1 = np.tile(nonrigid_y1, (frames.shape[0],1))
    xmax1 = np.tile(nonrigid_x1, (frames.shape[0],1))
    frames = nonrigid.transform_data(data=frames, nblocks=nblocks, 
        xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
    # 2nd rigid shift
    frames = np.roll(frames, (-rigid_y2, -rigid_x2), axis=(1,2))
    # 2nd nonrigid shift            
    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size2)
    ymax1 = np.tile(nonrigid_y2, (frames.shape[0],1))
    xmax1 = np.tile(nonrigid_x2, (frames.shape[0],1))
    frames = nonrigid.transform_data(data=frames, nblocks=nblocks, 
        xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
    return frames
    

def auto_pre_cur(dataFolder, minRadiusThreUm = 4, maxRadiusThreUm = 20, arThre = 1.5, crThre = 1.4):
    # Load necessary files
    ops = np.load(f'{dataFolder}ops.npy', allow_pickle=True).item()
    f = np.load(f'{dataFolder}F.npy')
    fneu = np.load(f'{dataFolder}Fneu.npy')
    iscell = np.load(f'{dataFolder}iscell.npy')
    stat = np.load(f'{dataFolder}stat.npy', allow_pickle=True)
    
    # Make a list of all ROIs that have lower intensitiy compared to the surroundings
    # - Likely to be blood vessels
    # by selecting the ROIs that have higher average neuropil signal compared to "soma" fluorescence
    # throughout the whole imaging sessions
    
    frameSelect = []
    for i in range(0,len(ops['nframes_per_folder'])):
        tempRange = [*range(sum(ops['nframes_per_folder'][0:i]), sum(ops['nframes_per_folder'][0:i+1]))]
        frameSelect.append(tempRange)
    
    a = fneu - f
    b = np.zeros((np.shape(a)[0],len(frameSelect)))
    for i in range(len(frameSelect)):
        b[:,i] = a[:,frameSelect[i]].mean(axis=1) >= 0
    c = b.sum(axis=1)
    vesselList = list(*np.where(c==len(frameSelect)))

    # Make a list of all ROIs that "touches" the edge of the image
    xmin = ops['xrange'][0]
    xmax = ops['xrange'][1]-1
    ymin = ops['yrange'][0]
    ymax = ops['yrange'][1]-1
    
    edgeList = []
    for i in range(len(stat)):
        if any(stat[i]['xpix']==xmin) or any(stat[i]['xpix']==xmax) or any(stat[i]['ypix']==ymin) or any(stat[i]['ypix']==ymax):
            edgeList.append(i)
            
    # Settings for ROI removal based on size, aspect ratio, and compact
    if 'umPerPix' not in ops:
        mouse = int(ops['ops_path'].split('/plane_')[0][-3:])
        if mouse > 31:
            ops['zoom'] = 1.7
        else:
            ops['zoom'] = 2.0
        ops['umPerPix'] = 1.4/ops['zoom']
        np.save(f'{dataFolder}ops.npy', ops)
    minNpixThre = (minRadiusThreUm/ops['umPerPix']) ** 2 * np.pi
    maxNpixThre = (maxRadiusThreUm/ops['umPerPix']) ** 2 * np.pi
    if dataFolder[-1] != os.path.sep:
        dataFolder = f'{dataFolder}/'
        
    # Curate ROIs
    for i in range(0,len(stat)):        
        if (i in edgeList) or (i in vesselList):
            iscell[i][0] = 0
        else:
            if stat[i]['npix'] <= minNpixThre:
                iscell[i][0] = 0
            if stat[i]['npix'] > minNpixThre:
                if (stat[i]['aspect_ratio'] < arThre) & (stat[i]['compact'] < crThre):
                    iscell[i][0] = 1
                else:
                    iscell[i][0] = 0    
            if stat[i]['npix'] > maxNpixThre:
                iscell[i][0] = 0
    
    # Save to "iscell.npy"
    np.save(f'{dataFolder}iscell.npy', iscell)
    numCell = sum(iscell[:,0])
    numNotCell = len(iscell[:,0]) - numCell
    return numCell, numNotCell

def roi_detection(ops, db):
    '''
    Runs roi detection and saves the results in a sub-directory
    (db includes the threshold)
    '''
    opsPath = os.path.join(db['save_path0'], db['save_folder'])
    threshold = int(np.round(db['threshold_scaling']*100))
    subDir = f'{opsPath}/plane0/th{threshold:03}/'
    opsfn = f'{subDir}ops.npy'
    iscellfn = f'{subDir}iscell.npy'
    if not os.path.isdir(subDir) or not os.path.isfile(opsfn) or not os.path.isfile(iscellfn):
       # raise (f'Path already exists for threshold {threshold}')
    # else:
        if not os.path.isdir(subDir):
           os.mkdir(subDir)
        run_s2p(ops,db)
        flist = [f.name for f in os.scandir(f'{opsPath}/plane0') if f.is_file()]
        for fn in flist:
           if fn[-4:] != '.bin':
               shutil.copy(f'{opsPath}/plane0/{fn}', f'{subDir}{fn}')
    numCell, numNotCell = auto_pre_cur(subDir)
    return numCell, numNotCell

#%% Making a new ops.npy 
# With necessary keys for roi detection and spike deconvolution
# Need to compare calculating dF/F from each session?
mi = 0
mouse = mice[mi]

for pn in range(6,9):
    s2pMouseDir = f'{s2pDir}{mouse:03}/'
    s2pPlaneDir = f'{s2pMouseDir}plane_{pn}/plane0/'
    stitchedOpsFn = f'{s2pPlaneDir}stitched_ops.npy'
    stitchedDataFn = f'{s2pPlaneDir}data.bin'
    stitchedOps = np.load(stitchedOpsFn, allow_pickle=True).item()
    
    thFolderLen = 9 # to test done folders
    fnresults = ['F.npy', 'Fneu.npy', 'iscell.npy', 'spks.npy', 'stat.npy', 'data.bin', 'ops.npy',
                 'stitched_ops.npy'] # to test done folders
    
    ops = stitchedOps['opsList'][0]
    
    ops['h5list'] = []
    ops['filelist'] = []
    ops['corrXY'] = []
    ops['corrXY1'] = []
    ops['NRsm'] = []
    ops['regDX'] = []
    ops['regPC'] = []
    ops['tPC'] = []
    ops['Vcorr'] = []
    ops['Vmap'] = []
    ops['Vmax'] = []
    ops['Vsplit'] = []
    ops['xoff'] = []
    ops['xoff1'] = []
    ops['yoff'] = []
    ops['yoff1'] = []
    ops['threshold_scaling'] = []
    ops['data_path'] = []
    
    with BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'], read_filename=stitchedDataFn) as f:
        ops['meanImg'] = f.data.mean(axis=0)
        ops = enhanced_mean_image(ops)
    ops['refImg'] = []
    ops['max_proj'] = []
    ops['ops_path'] = f'{s2pPlaneDir}ops.npy'
    
    
    ops['fs'] = freq[mi]
    ops['zoom'] = zoom[mi]
    ops['umPerPix'] = 1.4/ops['zoom']
    
    ops['save_path0'] = s2pMouseDir # Will be override by db, but just in case...
    ops['save_folder'] = f'plane_{pn}' # Will be override by db, but just in case...
    ops['save_path'] = s2pPlaneDir # Will be override by suite2p using db, but just in case...
    ops['nframes'] = sum(stitchedOps['nFrames'])
    ops['nframes_per_folder'] = np.array(stitchedOps['nFrames'])
    ops['badframes'] = np.concatenate([op['badframes'] for op in stitchedOps['opsList']])
    ops['soma_crop'] = True # Introduced in v0.10.0?
    ops['raw_file'] = [] # just in case
    ops['fast_disk'] = [] # just in case
    
    ops['tau'] = 1.5
    ops['batch_size'] = 5000
    
    ops['nbinned'] = 100000  # max number of binned frames for cell detection        
    ops['max_overlap'] = 0.75  # cells with more overlap than this get removed during triage, before refinement
    ops['allow_overlap'] = False
    
    np.save(f'{s2pPlaneDir}ops.npy', ops)
    
    ls = os.listdir(s2pPlaneDir)
    tempThLen = sum([os.path.isdir(os.path.join(s2pPlaneDir, lstr)) for lstr in ls])
    tempResultLen = len([b for b in fnresults if b in ls])
    if not ((tempThLen == thFolderLen) & (tempResultLen == len(fnresults))): # this folder is not done yet
        # Detect ROI's - 1st round. 0.2 resolution
        numroi = {} # dict, paired between threshold and numCell, numNotCell
        thresholdList = [int(th*10)/10 for th in np.linspace(0,0.8,5)]
        for threshold in thresholdList:
            db = {'data_path': [],
                'do_registration': 0, # Forcing to not run registration
                'save_path0': s2pMouseDir,
                'save_folder': f'plane_{pn}',
                'rerun_jk': 1, # Only for JK modification for suite2p v0.09
                'threshold_scaling': threshold
            }
            numCell, numNotCell = roi_detection(ops, db)
            numroi[threshold] = [numCell, numNotCell] 
        
        # Pick the best, and then run the 2nd round. 0.1 resolution
        numroiArray = np.array(list(numroi.values()))[:,0]
        maxind = np.argmax(numroiArray)
        maxthreshold = int(list(numroi.keys())[maxind]*10)/10
        
        thresholdList = [maxthreshold-0.1, maxthreshold+0.1]
        for threshold in thresholdList:
            db = {'data_path': [],
                'do_registration': 0, # Forcing to not run registration
                'save_path0': s2pMouseDir,
                'save_folder': f'plane_{pn}',
                'rerun_jk': 1, # Only for JK modification for suite2p v0.09
                'threshold_scaling': threshold
            }
            numCell, numNotCell = roi_detection(ops, db)
            numroi[threshold] = [numCell, numNotCell]
        
        # Pick the best, and then run the 3rd round. 0.05 resolution
        numroiArray = np.array(list(numroi.values()))[:,0]
        maxind = np.argmax(numroiArray)
        maxthreshold = int(list(numroi.keys())[maxind]*10)/10
        
        thresholdList = [maxthreshold-0.05, maxthreshold+0.05]
        for threshold in thresholdList:
            db = {'data_path': [],
                'do_registration': 0, # Forcing to not run registration
                'save_path0': s2pMouseDir,
                'save_folder': f'plane_{pn}',
                'rerun_jk': 1, # Only for JK modification for suite2p v0.09
                'threshold_scaling': threshold
            }
            numCell, numNotCell = roi_detection(ops, db)
            numroi[threshold] = [numCell, numNotCell]
        
        # Pick the best, and copy the result to the plane0 directory
        # And draw a plot for numCells and numNotCells against thresholds
        numroiArray = np.array(list(numroi.values()))[:,0]
        maxind = np.argmax(numroiArray)
        maxthreshold = list(numroi.keys())[maxind]
        mts = int(np.round(maxthreshold*100))
    
        bestDir = f'{s2pPlaneDir}/th{mts:03}/'
        flist = [f.name for f in os.scandir(bestDir) if f.is_file()]
        for fn in flist:
            shutil.copy(f'{bestDir}{fn}', f'{s2pPlaneDir}/{fn}')


#%% check mean images
# 2021/09/13

# a = np.load('D:/TPM/JK/s2p/025/plane_3/plane0/ops.npy', allow_pickle=True).item()
# m1 = a['meanImg']
# Ly = a['Ly']
# Lx = a['Lx']
# with BinaryFile(Ly=Ly, Lx=Lx, read_filename='D:/TPM/JK/s2p/025/plane_3/plane0/data.bin') as f:
#     m2 = f.data.mean(axis=0)
    
# #%%
# import napari
# napari.view_image(np.array([m1,m2]))

# #%%
# m3 = a['max_proj']
# napari.view_image(np.array([m1,m2,m3,m1]))

# #%%
# run_s2p(ops, db)

#%% Check ops
# 2021/09/14
# op = np.load('D:/TPM/JK/s2p/025/plane_3/plane0/ops.npy', allow_pickle=True).item()
# op = np.load(f'{planeDir}/ops.npy', allow_pickle=True).item()
## Correct ops['meanImg']

# baseDir = 'D:/TPM/JK/s2p/025/'
# for pn in range(1,5):
#     planeDir = f'{baseDir}plane_{pn}/plane0/'
#     dirList = [f.name for f in os.scandir(planeDir) if f.is_dir()]
#     for tempDir in dirList:
#         tempPath = os.path.join(planeDir, tempDir)
#         ops = np.load(f'{tempPath}/ops.npy', allow_pickle=True).item()
#         ops = enhanced_mean_image(ops)
#         np.save(f'{tempPath}/ops.npy', ops)
#     ops = np.load(f'{planeDir}ops.npy', allow_pickle=True).item()
#     ops = enhanced_mean_image(ops)
#     np.save(f'{planeDir}/ops.npy', ops)

#%% Compare ROI retention
# (1) For each session, check how much of the ROIs are retained in the session-stitched ROI
#     Should apply two-step registration parameters.
#     Check activity correlation with the closest ROI. Set a threshold.
# (2) Check if the ROIs from session-stitching has activities in each session.
#     First, look at cell map to see obvious missing ROIs from each session.
#     Then look at activities in sessions where the ROI is not extracted.


#%%

pn = 3
s2pPlaneDir = f'{s2pDir}{mouse:03}/plane_{pn}/plane0/'
h5PlaneDir = f'{h5Dir}{mouse:03}/plane_{pn}/'

stitchedOps = np.load(f'{s2pPlaneDir}stitched_ops.npy', allow_pickle=True).item()
# Temporary: for first 3 mice where block sizes and maxregshiftNRs were not saved for each registration step
if 'block_size1' not in stitchedOps:
    stitchedOps['block_size1'] = [128,128]
    stitchedOps['block_size2'] = [32,32]
    stitchedOps['maxregshiftNR1'] = 12
    stitchedOps['maxregshiftNR2'] = 3

allOps = np.load(f'{s2pPlaneDir}ops.npy', allow_pickle=True).item()
allStat = np.load(f'{s2pPlaneDir}stat.npy', allow_pickle=True).tolist()
allIscell = np.load(f'{s2pPlaneDir}iscell.npy', allow_pickle=True).tolist()
allCellInds = np.where(np.array([ic[0] for ic in allIscell]))[0]
Ly, Lx = allOps['Ly'], allOps['Lx']
allCellMap = np.zeros((Ly,Lx), dtype=np.int16)
for i, ci in enumerate(allCellInds):
    allCellMap[allStat[ci]['ypix'], allStat[ci]['xpix']] = 1

    
allBlended = []
nSessions = len(stitchedOps['useSessionNames'])
for si in range(len(stitchedOps['useSessionNames'])):
    print(f'Processing session index {si}/{nSessions}')
    sn = stitchedOps['useSessionNames'][si][4:]
    
    sessionOps = np.load(f'{h5PlaneDir}{sn}/plane0/ops.npy', allow_pickle=True).item()
    sessionStat = np.load(f'{h5PlaneDir}{sn}/plane0/stat.npy', allow_pickle=True).tolist()
    sessionIscell = np.load(f'{h5PlaneDir}{sn}/plane0/iscell.npy', allow_pickle=True).tolist()
    sessionCellInds = np.where(np.array([ic[0] for ic in sessionIscell]))[0]
    
    rigid_y1 = stitchedOps['rigid_offsets_y1'][si]
    rigid_x1 = stitchedOps['rigid_offsets_x1'][si]
    nonrigid_y1 = stitchedOps['nonrigid_offsets_y1'][si]
    nonrigid_x1 = stitchedOps['nonrigid_offsets_x1'][si]
    block_size1 = stitchedOps['block_size1']
    rigid_y2 = stitchedOps['rigid_offsets_y2'][si]
    rigid_x2 = stitchedOps['rigid_offsets_x2'][si]
    nonrigid_y2 = stitchedOps['nonrigid_offsets_y2'][si]
    nonrigid_x2 = stitchedOps['nonrigid_offsets_x2'][si]
    block_size2 = stitchedOps['block_size2']
    
    # Try registering all cell ID
    # See how much it differs from the original cell ID
    Ly, Lx = sessionOps['Ly'], sessionOps['Lx']
    sessionCellMap = np.zeros((Ly,Lx), dtype=np.int16)
    tempCellMap = np.zeros((len(sessionCellInds),Ly,Lx), dtype=np.int16)
    for i, ci in enumerate(sessionCellInds):
        tempCellMap[i,sessionStat[ci]['ypix'], sessionStat[ci]['xpix']] = 1
    regCellMap = twostep_register(tempCellMap, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                         rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2)    
    
    # Just rounding makes the result area too small when transformation is large
    thresh = 0.33
    regCellMap[np.where(regCellMap>thresh)] = 1
    regCellMap = np.round(regCellMap)
    
    # Visual inspection
    # boxlen = 20
    # tempCi = 105
    # fig, ax = plt.subplots(1,2)
    # tempCellRaw = tempCellMap[tempCi,:,:]
    
    # rawY, rawX = np.where(tempCellRaw)
    # rawYmed = np.median(rawY)
    # rawXmed = np.median(rawX)
    # rawYrange = (int(max(0,np.round(rawYmed-boxlen/2))), int(min(Ly,np.round(rawYmed+boxlen/2))))
    # rawXrange = (int(max(0,np.round(rawXmed-boxlen/2))), int(min(Lx,np.round(rawXmed+boxlen/2))))
    # ax[0].imshow(tempCellRaw[rawYrange[0]:rawYrange[1], rawXrange[0]:rawXrange[1]])
    
    # tempCellReg = regCellMap[tempCi,:,:] 
    # regY, regX = np.where(tempCellReg)
    # regYmed = np.median(regY)
    # regXmed = np.median(regX)
    # regYrange = (int(max(0,np.round(regYmed-boxlen/2))), int(min(Ly,np.round(regYmed+boxlen/2))))
    # regXrange = (int(max(0,np.round(regXmed-boxlen/2))), int(min(Lx,np.round(regXmed+boxlen/2))))
    # ax[1].imshow(tempCellReg[regYrange[0]:regYrange[1], regXrange[0]:regXrange[1]])
    
    # Visual comparison with stitched ROI detection
    
    regCellMapBin = np.sum(regCellMap,axis=0)
    regCellMapBin[regCellMapBin>1] = 1
    
    blended = imblend(allCellMap, regCellMapBin)
    allBlended.append(blended)

viewer = napari.Viewer()
viewer.add_image(allOps['meanImg'])
viewer.add_image(np.array(allBlended))

'''
Quite many ROIs are missing from the session stitching.
Those cannot be salvaged by manual ROI addition, since it is not obvious from the map.
Many of these are repeating from multiple sessions, meaning they are not just very sparse neurons.
(ROI curation from session stitching itself has some problems.)
'''
#%% Quantification
# Use neurons that are within the FOV of the stitched (i.e. reference image FOV)
#   - Remove the ones that touch the edge of the FOV.
# Assess matching by having more than 50% of the ROI being within an ROI of the stitched.
#    - Not the best way, but good enough for a rough estimation at this stage.
# Also calculate total # of ROIs missing from the entire each session curation.

pn = 3
s2pPlaneDir = f'{s2pDir}{mouse:03}/plane_{pn}/plane0/'
h5PlaneDir = f'{h5Dir}{mouse:03}/plane_{pn}/'

stitchedOps = np.load(f'{s2pPlaneDir}stitched_ops.npy', allow_pickle=True).item()
# Temporary: for first 3 mice where block sizes and maxregshiftNRs were not saved for each registration step
if 'block_size1' not in stitchedOps:
    stitchedOps['block_size1'] = [128,128]
    stitchedOps['block_size2'] = [32,32]
    stitchedOps['maxregshiftNR1'] = 12
    stitchedOps['maxregshiftNR2'] = 3

allOps = np.load(f'{s2pPlaneDir}ops.npy', allow_pickle=True).item()
allStat = np.load(f'{s2pPlaneDir}stat.npy', allow_pickle=True).tolist()
allIscell = np.load(f'{s2pPlaneDir}iscell.npy', allow_pickle=True).tolist()
allCellInds = np.where(np.array([ic[0] for ic in allIscell]))[0]
allCellMap = np.zeros((len(allCellInds), Ly, Lx), dtype=bool)
for i, ci in enumerate(allCellInds):
    allCellMap[i,allStat[ci]['ypix'], allStat[ci]['xpix']] = 1

eachTotal = []
eachMatch = []
sumCellMap = []
pixValThresh = 0.33
numpixThresh = 10
nSessions = len(stitchedOps['useSessionNames'])
for si in range(len(stitchedOps['useSessionNames'])):
    print(f'Processing session index {si}/{nSessions}')
    sn = stitchedOps['useSessionNames'][si][4:]
    
    sessionOps = np.load(f'{h5PlaneDir}{sn}/plane0/ops.npy', allow_pickle=True).item()
    sessionStat = np.load(f'{h5PlaneDir}{sn}/plane0/stat.npy', allow_pickle=True).tolist()
    sessionIscell = np.load(f'{h5PlaneDir}{sn}/plane0/iscell.npy', allow_pickle=True).tolist()
    sessionCellInds = np.where(np.array([ic[0] for ic in sessionIscell]))[0]
    
    rigid_y1 = stitchedOps['rigid_offsets_y1'][si]
    rigid_x1 = stitchedOps['rigid_offsets_x1'][si]
    nonrigid_y1 = stitchedOps['nonrigid_offsets_y1'][si]
    nonrigid_x1 = stitchedOps['nonrigid_offsets_x1'][si]
    block_size1 = stitchedOps['block_size1']
    rigid_y2 = stitchedOps['rigid_offsets_y2'][si]
    rigid_x2 = stitchedOps['rigid_offsets_x2'][si]
    nonrigid_y2 = stitchedOps['nonrigid_offsets_y2'][si]
    nonrigid_x2 = stitchedOps['nonrigid_offsets_x2'][si]
    block_size2 = stitchedOps['block_size2']
    
    # Try registering all cell ID
    # See how much it differs from the original cell ID
    Ly, Lx = sessionOps['Ly'], sessionOps['Lx']
    tempCellMap = np.zeros((len(sessionCellInds),Ly,Lx), dtype=np.int16)
    for i, ci in enumerate(sessionCellInds):
        tempCellMap[i,sessionStat[ci]['ypix'], sessionStat[ci]['xpix']] = 1
    regCellMap = twostep_register(tempCellMap, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                         rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2)    
    
    # Just rounding makes the result area too small when transformation is large
    regCellMap[np.where(regCellMap>pixValThresh)] = 1
    regCellMap = np.round(regCellMap).astype(bool)

    # Just use the ROIs within the FOV
    withinFOVind = np.where(np.sum(regCellMap, axis=(1,2))>=numpixThresh)[0]
    numTotalSession = len(withinFOVind)
    numMatchSession = 0
    
    for i in withinFOVind:
        # Remove ROIs out of FOV
        coordinate = np.where(regCellMap[i,:,:])
        ypix, xpix = coordinate[0], coordinate[1]
        # Just use the ROIs that does not touch the edge
        if 0 in ypix or 0 in xpix or Ly in ypix or Lx in xpix:
            # print(f'ypix = {ypix}')
            # print(f'xpix = {xpix}')
            
            numTotalSession -= 1
            # print(f'numTotal = {numTotalSession}')
        # If it is within the FOV,
        else:
            # Find matching
            numPix = np.sum(regCellMap[i,:,:])
            tempMap = np.tile(regCellMap[i,:,:],(len(allCellInds),1,1)).astype(bool)
            multMap = tempMap * allCellMap
            overlapPix = np.sum(multMap, axis=(1,2))
            if np.amax(overlapPix) > numPix/2:
                numMatchSession += 1
            #     print(f'numMatch = {numMatchSession}')
            # else:
            #     print(f'Nonmatch')
    
    eachTotal.append(numTotalSession)
    eachMatch.append(numMatchSession)

    if len(sumCellMap)==0:
        sumCellMap = regCellMap[withinFOVind,:,:] 
    else:
        tempSumMap = sumCellMap.copy()
        for i in withinFOVind:
            # Remove ROIs out of FOV
            coordinate = np.where(regCellMap[i,:,:])
            ypix, xpix = coordinate[0], coordinate[1]
            # Just use the ROIs that does not touch the edge
            if 0 in ypix or 0 in xpix or Ly in ypix or Lx in xpix:
                pass
            # If it is within the FOV,
            else:
                # Find matching
                numPix = np.sum(regCellMap[i,:,:])
                tempMap = np.tile(regCellMap[i,:,:],(sumCellMap.shape[0],1,1)).astype(bool)
                multMap = tempMap * sumCellMap
                overlapPix = np.sum(multMap, axis=(1,2))
                if np.amax(overlapPix) <= numPix/2: # It's a non-match. Add
                    tempSumMap = np.concatenate((tempSumMap, regCellMap[i:i+1,:,:]))
        sumCellMap = tempSumMap.copy()

#%% Calcualte from all each sessions
sumRoiTotal = sumCellMap.shape[0]
sumRoiMatch = 0
for i in range(sumRoiTotal):
    numPix = np.sum(sumCellMap[i,:,:])
    tempMap = np.tile(sumCellMap[i,:,:],(len(allCellInds),1,1)).astype(bool)
    multMap = tempMap * allCellMap
    overlapPix = np.sum(multMap, axis=(1,2))
    if np.amax(overlapPix) > numPix/2:
        sumRoiMatch += 1

stitchedRoiTotal = allCellInds.shape[0]
stitchedMatchToSum = 0
for i in range(stitchedRoiTotal):
    numPix = np.sum(allCellMap[i,:,:])
    tempMap = np.tile(allCellMap[i,:,:],(sumRoiTotal,1,1)).astype(bool)
    multMap = tempMap * sumCellMap
    overlapPix = np.sum(multMap, axis=(1,2))
    if np.amax(overlapPix) > numPix/2:
        stitchedMatchToSum += 1
    
    
#%%
viewer = napari.Viewer()
viewer.add_image(allOps['meanImg'])
viewer.add_image(imblend(np.squeeze(np.sum(allCellMap, axis=0))>0.5, np.squeeze(np.sum(sumCellMap, axis=0))>0.5))

#%%
fig, ax = plt.subplots()
ax.bar(range(len(eachTotal)), eachTotal, label='Total', color='b')
ax.bar(range(len(eachTotal)), eachMatch, label='Match', color='y')
ax.legend()
ax.set_xlabel('Session index')
ax.set_ylabel('Cell number')
ax2 = ax.twinx()
ax2.plot(range(len(eachTotal)), (np.array(eachTotal)-np.array(eachMatch))/np.array(eachTotal), 'ro')
ax2.set_ylabel('Miss rate', color='r')
ax2.set_ylim([0,0.5])
ax2.set_yticklabels(np.linspace(0,5,6)/10, color='r')

#%%
print(sumRoiTotal)
print(sumRoiMatch)
print(sumRoiMatch / sumRoiTotal)
print(stitchedRoiTotal)
print(stitchedMatchToSum)
print(stitchedMatchToSum / stitchedRoiTotal)



'''
Conclusion: Too much loss from the stitched.
Each session curation is necessary
'''



#%% ROI matching
# What is the best method in matching ROIs??
# Also, which neurons get lost in the stitched, and combined?

# Starting from the stitched ROIs. Add missing ROIs from each session, chronologically.
# Before doing this, see the distribution of Jaccard index, centroid difference, and signal correlation.
# Look at 3 closest neurons. (even the ones without any overlap)

mi = 0
mouse = mice[mi]
pn = 3
s2pPlaneDir = f'{s2pDir}{mouse:03}/plane_{pn}/plane0/'
h5PlaneDir = f'{h5Dir}{mouse:03}/plane_{pn}/'

stitchedOps = np.load(f'{s2pPlaneDir}stitched_ops.npy', allow_pickle=True).item()
Ly = stitchedOps['opsList'][0]['Ly']
Lx = stitchedOps['opsList'][0]['Lx']
umPerPix = stitchedOps['opsList'][0]['umPerPix']

allSpk = np.load(f'{s2pPlaneDir}spks.npy', allow_pickle=True).tolist()
allStat = np.load(f'{s2pPlaneDir}stat.npy', allow_pickle=True).tolist()
allIscell = np.load(f'{s2pPlaneDir}iscell.npy', allow_pickle=True).tolist()

allCellInds = np.where(np.array([ic[0] for ic in allIscell]))[0]
allSpk = [np.array(allSpk[i]) for i in allCellInds]
allStat = [allStat[i] for i in allCellInds]
allCellMap = np.zeros((len(allCellInds), Ly, Lx), dtype=bool)
for i, _ in enumerate(allCellInds):
    allCellMap[i,allStat[i]['ypix'], allStat[i]['xpix']] = 1
allMed = np.array([stat['med'] for stat in allStat])
frameNums = np.concatenate(([0], np.array(np.cumsum(stitchedOps['nFrames']))))

jaccardInd = [] # Jaccard index (intersection / union)
centDiff = [] # centroid difference (= distance)
sigCorr = [] # signal correlation
distOrder = [] # 1,2, or 3 (just include the closest 3 that overlaps)
pixValThresh = 0.33
numpixThresh = 10
nSessions = len(stitchedOps['useSessionNames'])



#%% First, test in a session
# for si in range(len(stitchedOps['useSessionNames'])):
si = 0

# print(f'Processing session index {si}/{nSessions}')
sn = stitchedOps['useSessionNames'][si][4:]
currSessionFrames = np.arange(frameNums[si], frameNums[si+1])

# sessionOps = np.load(f'{h5PlaneDir}{sn}/plane0/ops.npy', allow_pickle=True).item()
sessionIscell = np.load(f'{h5PlaneDir}{sn}/plane0/iscell.npy', allow_pickle=True).tolist()
sessionCellInds = np.where(np.array([ic[0] for ic in sessionIscell]))[0]
sessionStat = np.load(f'{h5PlaneDir}{sn}/plane0/stat.npy', allow_pickle=True).tolist()
sessionSpk = np.load(f'{h5PlaneDir}{sn}/plane0/spks.npy', allow_pickle=True).tolist()

sessionStat = [sessionStat[i] for i in sessionCellInds]
sessionSpk = [np.array(sessionSpk[i]) for i in sessionCellInds]


rigid_y1 = stitchedOps['rigid_offsets_y1'][si]
rigid_x1 = stitchedOps['rigid_offsets_x1'][si]
nonrigid_y1 = stitchedOps['nonrigid_offsets_y1'][si]
nonrigid_x1 = stitchedOps['nonrigid_offsets_x1'][si]
block_size1 = stitchedOps['block_size1']
rigid_y2 = stitchedOps['rigid_offsets_y2'][si]
rigid_x2 = stitchedOps['rigid_offsets_x2'][si]
nonrigid_y2 = stitchedOps['nonrigid_offsets_y2'][si]
nonrigid_x2 = stitchedOps['nonrigid_offsets_x2'][si]
block_size2 = stitchedOps['block_size2']

# Try registering all cell ID
# See how much it differs from the original cell ID

tempCellMap = np.zeros((len(sessionCellInds),Ly,Lx), dtype=np.int16)

for i, _ in enumerate(sessionCellInds):
    tempCellMap[i,sessionStat[i]['ypix'], sessionStat[i]['xpix']] = 1
regCellMap = twostep_register(tempCellMap, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                     rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2)

# Just rounding makes the result area too small when transformation is large
regCellMap[np.where(regCellMap>pixValThresh)] = 1
regCellMap = np.round(regCellMap).astype(bool)

# Just use the ROIs within the FOV
withinFOVind = np.where(np.sum(regCellMap, axis=(1,2))>=numpixThresh)[0]
numTotalSession = len(withinFOVind)
edgeInd = []
for ci in withinFOVind:
    # Remove ROIs out of FOV
    coordinate = np.where(regCellMap[ci,:,:])
    ypix, xpix = coordinate[0], coordinate[1]
    # Just use the ROIs that does not touch the edge
    if 0 in ypix or 0 in xpix or Ly in ypix or Lx in xpix:
        edgeInd.append(ci)
withinFOVind = [i for i in withinFOVind if i not in edgeInd]
#%%
jiSession = [] # Jaccard Index
cdSession = [] # centroid difference
scSession = [] # signal correlation 
doSession = [] # distance order
ciStitched = [] # Cell index in the stitched
ciSession = [] # Cell index in the session
for ci in withinFOVind:
# ci = withinFOVind[0]
    
    # Calculate centroid (median pixel)
    ypix, xpix = np.where(regCellMap[ci,:,:])
    pixInds = np.ravel_multi_index(np.array([ypix,xpix]), (Ly,Lx))
    ymed, xmed = np.median(ypix), np.median(xpix)
    imin = np.argmin((xpix-xmed)**2 + (ypix-ymed)**2)
    # ymed, xmed = ypix[imin], xpix[imin]
    med = np.array([ypix[imin], xpix[imin]])
    spk = np.array(sessionSpk[ci])
    spkInds = np.where(spk)
    # Find 3 closest from allCellMap
    # dist = umPerPix * ((allYmed - ymed)**2 + (allXmed - xmed)**2)**0.5
    dist = umPerPix * (np.sum((allMed - med)**2, axis=1)**0.5)
    closeInds = np.argsort(dist)[:1]
    
    for i, allci in enumerate(closeInds):
    
    # i = 0
    # allci = closeInds[i]    
    
        tempAllSingleMap = allCellMap[allci,:,:]   
        
        intersect = regCellMap[ci,:,:] & tempAllSingleMap
        union = regCellMap[ci,:,:] | tempAllSingleMap
        
        jiSession.append(np.sum(intersect)/np.sum(union))
        cdSession.append(dist[allci])
        # calculate signal correlation. From nonzero spike frames in either cell
        # (To remove inactive frames)
        tempSpk = allSpk[allci][currSessionFrames]
        # tempSpkInds = np.union1d(np.where(tempSpk), spkInds)
        
        # scSession.append(np.corrcoef(tempSpk[tempSpkInds], spk[tempSpkInds])[0,1])
        scSession.append(np.corrcoef(tempSpk, spk)[0,1])
        doSession.append(i)
        ciStitched.append(allci)
        ciSession.append(ci)
        
       
# jaccardInd.append(jiSession)
# centDiff.append(cdSession)
# sigCorr.append(scSession)
# distOrder.append(doSession)

#%%
    
# fig, ax = plt.subplots(2,2)
# ax[0,0].scatter(jaccardInd[0], centDiff[0])
# ax[0,0].set_xlabel('Jaccard index')
# ax[0,0].set_ylabel('Distance')

# ax[0,1].scatter(sigCorr[0], centDiff[0])
# ax[0,1].set_xlabel('Signal correlation')
# ax[0,1].set_ylabel('Distance')

# ax[1,0].scatter(sigCorr[0], jaccardInd[0])
# ax[1,0].set_xlabel('Signal correlation')
# ax[1,0].set_ylabel('Jaccard index')


# ax[1,1].scatter(centDiff[0], distOrder[0])
# ax[1,1].set_xlabel('Distance')
# ax[1,1].set_ylabel('Distance order')
    
# fig.tight_layout()
    
    
    #%%
        
fig, ax = plt.subplots(2,2)
ax[0,0].scatter(jiSession, cdSession)
ax[0,0].set_xlabel('Jaccard index')
ax[0,0].set_ylabel('Distance')

ax[0,1].scatter(scSession, cdSession)
ax[0,1].set_xlabel('Signal correlation')
ax[0,1].set_ylabel('Distance')

ax[1,0].scatter(scSession, jiSession)
ax[1,0].set_xlabel('Signal correlation')
ax[1,0].set_ylabel('Jaccard index')


ax[1,1].scatter(cdSession, doSession)
ax[1,1].set_xlabel('Distance')
ax[1,1].set_ylabel('Distance order')
    
fig.tight_layout()


# #%%
# plt.figure()
# ax = plt.axes(projection = '3d')
# ax.plot3D(jiSession, cdSession, scSession, 'k.')
# ax.set_xlabel('Jaccard index')
# ax.set_ylabel('Centroid distance')
# ax.set_zlabel('Signal correlation')




'''
Some shows low signal correlation with high overlap. Why?
'''

#%%
jiInd = np.where(np.array(jiSession) > 0.8)[0]
testInd = jiInd[np.argmin(np.array(scSession)[jiInd])]
sigCorrTemp = scSession[testInd]
jaccIndTemp = jiSession[testInd]

tempStitchedInd = ciStitched[testInd]
tempSessionInd = ciSession[testInd]

testAllMap = np.zeros((Ly,Lx,3))
tempAllMap = np.sum(allCellMap,axis=0)
testAllMap[:,:,0] = tempAllMap
testAllMap[:,:,2] = tempAllMap
testAllMap[:,:,1] = regCellMap[tempSessionInd,:,:]

testSingleMap = np.zeros((Ly,Lx,3))
tempSingleMap = allCellMap[tempStitchedInd,:,:]
testSingleMap[:,:,0] = tempSingleMap
testSingleMap[:,:,2] = tempSingleMap
testSingleMap[:,:,1] = regCellMap[tempSessionInd,:,:]

tempSessionSpk = np.array(sessionSpk[tempSessionInd])
tempStitchedSpk = allSpk[tempStitchedInd][currSessionFrames]
tempSpkInds = np.union1d(np.where(tempStitchedSpk), np.where(tempSessionSpk))
corrAll = np.corrcoef(tempSessionSpk, tempStitchedSpk)[0,1]
corrActive = np.corrcoef(tempSessionSpk[tempSpkInds], tempStitchedSpk[tempSpkInds])[0,1]

fig, ax = plt.subplots(1,3)
ax[0].imshow(testAllMap)
ax[0].set_title(f'Stitched cell index = {tempStitchedInd}\nSession cell index = {tempSessionInd}')
ax[1].imshow(testSingleMap)
ax[1].set_title(f'Overlap = {jaccIndTemp:.2}\nSignal correlation = {sigCorrTemp:.2}')
ax[2].scatter(tempSessionSpk, tempStitchedSpk)
ax[2].scatter(tempSessionSpk[tempSpkInds], tempStitchedSpk[tempSpkInds])
ax[2].set_title(f'All correlation = {corrAll:.2}\nActive correlation = {corrActive:.2}')
fig.tight_layout()





#%% 


'''
Some shows low overlap with high signal correlation. Why?
'''

#%%
scInd = np.where(np.array(scSession) > 0.95)[0]
testInd = scInd[np.argmin(np.array(jiSession)[scInd])]

tempMap = np.zeros_like(regCellMap[0,:,:])
testImg = np.zeros((3,Ly,Lx))
testImg[0,:,:] = tempMap
testImg[2,:,:] = tempMap
testImg[1,:,:] = regCellMap[int(testInd/3),:,:]
testImg = np.moveaxis(testImg, 0,-1)
plt.figure()
plt.imshow(testImg)




#%%
plt.figure()
plt.imshow(regCellMap[int(156/3),:,:])


'''
One of the cell map includes long protruding dendrites, reducing the Jaccard index (overlap quantification)
'''


#%%
    
