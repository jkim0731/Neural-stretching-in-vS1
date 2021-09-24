# -*- coding: utf-8 -*-
"""
ROI detection parameter search for the stitched binary file.
Copied from 210910_test_stitched_roi_detection.py
Results in suite2p files as well as "stitchedOps" file.

2021/09/24 JK
"""
import numpy as np
from matplotlib import pyplot as plt
from suite2p.run_s2p import run_s2p
from auto_pre_cur import auto_pre_cur
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