"""
Run suite2p from each session. (Copied from 210608_s2p_selected_sessions.py)
Automatically detect the best threshold parameter for ROI selection.

Use already registered data.bin (results from register_all_sessions.py)
Test ROI selection with thresholds between 0 and 1. 
Try 0.2 interval first (0:1:0.2), and then around the peak value x, try x-0.1:x+0.1:0.05.
Then, around the peak value y, try y-0.05:y+0.05:0.05.
With 9 iterations, get 0.05 resolution of threshold

2021/07/24 JK

Updates: 
    2021/12/07 JK
        Need to apply auto_pre_cur updated version (After 2021/10/01)
        Run each session is stat.npy does not exist, 
        or stat.npy made before 2021/10/01
"""

import numpy as np
from suite2p.run_s2p import run_s2p, default_ops
import os, glob, shutil
from auto_pre_cur import auto_pre_cur
import matplotlib.pyplot as plt
import time
import datetime

import gc
gc.enable()

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

#%% Basic settings
h5Dir = 'D:/TPM/JK/h5/'
fastDir = 'C:/JK/' # This better be in SSD

mice = [25,  27,  30,  36,  37,  38,  39,  41,  52,  53,  54,  56]
zoom = [2,   2,   2,   1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
freq = [7.7, 7.7, 7.7, 7.7, 6.1, 6.1, 6.1, 6.1, 7.7, 7.7, 7.7, 7.7]

ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['save_mat'] = False
ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works

# thFolderLen = 9 # to test done folders
# fnresults = ['F.npy', 'Fneu.npy', 'iscell.npy', 'spks.npy', 'stat.npy', 'data.bin', 'ops.npy'] # to test done folders

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

ops['nbinned']: 10000  # max number of binned frames for cell detection        
ops['max_overlap'] = 0.75  # cells with more overlap than this get removed during triage, before refinement
ops['allow_overlap'] = False
#%%
# for mi in [1,2]:
for mi in [8]:
    mouse = mice[mi]
    ops['fs'] = freq[mi]
    ops['zoom'] = zoom[mi]
    ops['umPerPix'] = 1.4/ops['zoom']

    # for pn in range(1,9):
    for pn in range(2,9):
        mouseDir = f'{h5Dir}{mouse:03}/'
        planeDir = f'{mouseDir}plane_{pn}/'
        tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{pn}.h5')

        # Make a list of session names and corresponding files
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
        
        for sntemp, sftemp in zip(sessionNames, sessionFiles):
            if len(sntemp.split('_')[2]) > 0:
                sn1 = sntemp.split('_')[1]
                sn2 = sntemp.split('_')[2]
                sn = f'{sn1}_{sn2}'
            else:
                sn = sntemp.split('_')[1]
                
            # Check if ROI param search was done in this folder
            # (# of thxxx folder == 9) & (isfile, F.npy, Fnew.npy, iscell.npy, spks.npy, stat.npy)
            # Updated: 2021/12/07
            # Run if no stat.npy or stat.npy made < 2021/10/01
            runFlag = 0
            
            tempDir = os.path.join(planeDir, f'{sn}/plane0/')
            # ls = os.listdir(tempDir)
            # tempThLen = sum([os.path.isdir(os.path.join(tempDir, lstr)) for lstr in ls])
            # tempResultLen = len([b for b in fnresults if b in ls])
            statFn = f'{tempDir}stat.npy'
            if not os.path.isfile(statFn):
                runFlag = 1
            else: 
                statTimestamp = os.path.getmtime(statFn)
                dateString = '10/01/2021'
                date = datetime.datetime.strptime(dateString, '%m/%d/%Y')            
                timestamp = datetime.datetime.timestamp(date)            
                if statTimestamp < timestamp: # this folder is done before Oct 2021
                    runFlag = 1
            # if not ((tempThLen == thFolderLen) & (tempResultLen == len(fnresults))) # this folder is not done yet 
            if runFlag:
                print(f'Running session {sn} plane {pn}')
                # Check for registration. If not, run registration
                if not (os.path.isfile(f'{tempDir}ops.npy') & os.path.isfile(f'{tempDir}data.bin')):
                    db = {'h5py': sftemp,
                    'h5py_key': ['data'],
                    'data_path': [],
                    'save_path0': planeDir,
                    'save_folder': f'{sn}',
                    'fast_disk': f'{fastDir}',
                    'roidetect': False,
                    }
                    run_s2p(ops,db)
                    rawbinFn = f'{planeDir}{sn}/plane0/data_raw.bin'
                    os.remove(rawbinFn)

                # Detect ROI's - 1st round. 0.2 resolution
                numroi = {} # dict, paired between threshold and numCell, numNotCell
                thresholdList = [int(th*10)/10 for th in np.linspace(0,0.8,5)]
                for threshold in thresholdList:
                    db = {'data_path': [],
                        'do_registration': 0, # Forcing to not run registration
                        'save_path0': planeDir,
                        'save_folder': f'{sn}',
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
                        'save_path0': planeDir,
                        'save_folder': f'{sn}',
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
                        'save_path0': planeDir,
                        'save_folder': f'{sn}',
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
                
                bestDir = f'{planeDir}{sn}/plane0/th{mts:03}/'
                flist = [f.name for f in os.scandir(bestDir) if f.is_file()]
                for fn in flist:
                    shutil.copy(f'{bestDir}{fn}', f'{planeDir}{sn}/plane0/{fn}')
