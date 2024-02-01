# -*- coding: utf-8 -*-
"""
Run suite2p from select sessions.
Automatically detect the best threshold parameter for ROI selection.

First, register from select h5 files.
Manually look at h5 file list and check it. (Do it volume by volume)
Then, test ROI selection with thresholds between 0 and 1. Try 0.2 interval first, and then apply auto pre-curation. 
Then, around the peak value x, try x-0.1:x+0.1:0.`. Then, around the peak value y, try y-0.05:y+0.05:0.05.
With 10 iterations, get 0.05 resolution of threshold

Compare the result with OnACID

2021/06/08 JK

"""

import numpy as np
from suite2p.run_s2p import run_s2p, default_ops
import os, glob, shutil
from auto_pre_cur import auto_pre_cur
import matplotlib.pyplot as plt

#%% Basic settings
h5Dir = 'D:/TPM/JK/h5/'

mice = [25,  27,  30,  36,  37,  38,  39,  41,  52,  53,  54,  56]
zoom = [2,   2,   2,   1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
freq = [7.7, 7.7, 7.7, 7.7, 6.1, 6.1, 6.1, 6.1, 7.7, 7.7, 7.7, 7.7]

mi = 0
mouse = mice[mi]
planes = [1,2,3,4]
# planes = [5,6,7,8]

removeSessionNames = ['025_016_', '025_018_', '025_5555_001', '025_5555_103', '025_9999_1']

ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 500
ops['batch_size'] = 5000
ops['maxregshift'] = 0.2
ops['block_size'] = [64, 64]
ops['maxregshiftNR'] = 20

ops['two_step_registration'] = True
ops['keep_movie_raw'] = True
ops['smooth_sigma_time'] = 2
ops['move_bin'] = True

ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']

#%% First, check file names to be processed
for planei in planes:
    mouseDir = f'{h5Dir}{mouse:03}/'
    planeDir = f'{mouseDir}plane_{planei}/'
    if not os.path.isdir(f'{planeDir}select_session'):
        tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{planei}.h5')
        repFn = [fn for fn in tempFnList if all(rsn not in fn for rsn in removeSessionNames)]
        print(*repFn, sep = '\n')

#%% Then, register them and search for the best threshold 
# Target 0.05 resolution from 10 iterations
# For now, assume no two thresholds will give the same # of cells 2021/06/09

def roi_detection(ops, db):
    '''
    Runs roi detection and saves the results in a sub-directory
    (db includes the threshold)
    '''
    opsPath = db['save_path0']
    threshold = db['threshold_scaling']
    subDir = f'{opsPath}/select_session/plane0/th{threshold*100:03}/'
    if os.path.isdir(subDir):
       raise (f'Path already exists for threshold {threshold}')
    else:
       os.mkdir(subDir)
    run_s2p(ops,db)
    flist = [f.name for f in os.scandir(f'{planeDir}select_session/plane0') if f.is_file()]
    for fn in flist:
        if fn[-4:] != '.bin':
            shutil.copy(f'{planeDir}select_session/plane0/{fn}', f'{subDir}{fn}')
    numCell, numNotCell = auto_pre_cur(subDir)
    return numCell, numNotCell

if len(planes) > 4:
    raise('# of planes should less than or equal to 4.')

f, ax = plt.subplots(2,2)
for pi, planei in enumerate(planes):
    mouseDir = f'{h5Dir}{mouse:03}/'
    planeDir = f'{mouseDir}plane_{planei}/'
    if not os.path.isdir(f'{planeDir}select_session'):
        # Register
        tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{planei}.h5')
        repFn = [fn for fn in tempFnList if all(rsn not in fn for rsn in removeSessionNames)]
        db = {'h5py': repFn,
                'h5py_key': ['data'],
                'data_path': [],
                'save_path0': planeDir,
                'save_folder': 'select_session',
                'fast_disk': f'{planeDir}/select_session',
                'roidetect': False,
            }
        run_s2p(ops,db)
        
        # Detect ROI's - 1st round. 0.2 resolution
        numroi = {} # dict, paired between threshold and numCell, numNotCell
        thresholdList = np.arange(0,1,0.2)
        for threshold in thresholdList:
            db = {'data_path': [],
                'do_registration': 0, # Forcing to not run registration
                'save_path0': planeDir,
                'save_folder': 'select_session',
                'smooth_sigma_time': 2,
                'rerun_jk': 1,
                'allow_overlap': False,
                'max_overlap': 0.3,
                'threshold_scaling': threshold
            }
            numCell, numNotCell = roi_detection(ops, db)
            numroi[threshold] = [numCell, numNotCell] 
        
        # Pick the best, and then run the 2nd round. 0.1 resolution
        numroiArray = np.array(list(numroi.values()))[:,0]
        maxind = np.argmax(numroiArray)
        maxthreshold = list(numroi.keys())[maxind]
        
        thresholdList = [maxthreshold-0.1, maxthreshold+0.1]
        for threshold in thresholdList:
            db = {'data_path': [],
                'do_registration': 0, # Forcing to not run registration
                'save_path0': planeDir,
                'save_folder': 'select_session',
                'smooth_sigma_time': 2,
                'rerun_jk': 1,
                'allow_overlap': False,
                'max_overlap': 0.3,
                'threshold_scaling': threshold
            }
            numCell, numNotCell = roi_detection(ops, db)
            numroi[threshold] = [numCell, numNotCell]
        
        # Pick the best, and then run the 3rd round. 0.05 resolution
        numroiArray = np.array(list(numroi.values()))[:,0]
        maxind = np.argmax(numroiArray)
        maxthreshold = list(numroi.keys())[maxind]
        
        thresholdList = [maxthreshold-0.05, maxthreshold+0.05]
        for threshold in thresholdList:
            db = {'data_path': [],
                'do_registration': 0, # Forcing to not run registration
                'save_path0': planeDir,
                'save_folder': 'select_session',
                'smooth_sigma_time': 2,
                'rerun_jk': 1,
                'allow_overlap': False,
                'max_overlap': 0.3,
                'threshold_scaling': threshold
            }
            numCell, numNotCell = roi_detection(ops, db)
            numroi[threshold] = [numCell, numNotCell]
        
        # Pick the best, and copy the result to the plane0 directory
        # And draw a plot for numCells and numNotCells against thresholds
        numroiArray = np.array(list(numroi.values()))[:,0]
        maxind = np.argmax(numroiArray)
        maxthreshold = list(numroi.keys())[maxind]
        
        bestDir = f'{planeDir}select_session/plane0/th{maxthreshold*100:03}/'
        flist = [f.name for f in os.scandir(bestDir) if f.is_file()]
        for fn in flist:
            shutil.copy(f'{bestDir}{fn}', f'{planeDir}select_session/plane0/{fn}')
        
        xid = pi % 2
        yid = pi // 2
        
        numcells = numroiArray
        numnc = np.array(list(numroi.values()))[:,1]
        thresholds = np.array(list(numroi.keys()))
        isorted = np.argsort(thresholds)
        thresholds = thresholds[isorted]
        numcells = numcells[isorted]
        numnc = numnc[isorted]
        ax[yid,xid].plot(thresholds, numcells, 'bo-')
        ax[yid,xid].set_ylabel('# of cells')
        ax[yid,xid].set_xlabel('Threshold')
        ax[yid,xid].tick_params(axis = 'y', colors = 'b')
        axr = ax[yid,xid].twinx()
        axr.plot(thresholds, numnc, 'yo-')
        axr.set_ylabel('# of not-cells')
        axr.tick_params(axis = 'y', colors = 'y')
        ax[yid, xid].set_title(f'JK{mouse:03} plane {planei}')
f.tight_layout()
        
        
        