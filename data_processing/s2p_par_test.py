
''' 
    Multiprocessing doesn't work. 
    It gives some kind of indexing error that does not happen during single processing.
    Spits out ops.npy but stops after that. 
        ValueError: operands could not be broadcast together with shapes (4905,) (4152,) 
        
        Traceback (most recent call last):
          File "C:\Users\shires\AppData\Local\Continuum\anaconda3\envs\suite2p\lib\multiprocessing\pool.py", line 121, in worker
            result = (True, func(*args, **kwds))
          File "C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Analysis\codes\run_s2p_parallel.py", line 9, in run_s2p_parallel
            opsEnd = run_s2p(ops, db)
          File "C:\Users\shires\AppData\Local\Continuum\anaconda3\envs\suite2p\lib\site-packages\suite2p\run_s2p.py", line 405, in run_s2p
            op = run_plane(op, ops_path=ops_path)
          File "C:\Users\shires\AppData\Local\Continuum\anaconda3\envs\suite2p\lib\site-packages\suite2p\run_s2p.py", line 183, in run_plane
            ops = registration.register_binary(ops, refImg=refImg) # register binary
          File "C:\Users\shires\AppData\Local\Continuum\anaconda3\envs\suite2p\lib\site-packages\suite2p\registration\register.py", line 416, in register_binary
            Lx=ops['Lx'],
          File "C:\Users\shires\AppData\Local\Continuum\anaconda3\envs\suite2p\lib\site-packages\suite2p\registration\register.py", line 49, in compute_crop
            badframes = np.logical_or(px > th_badframes * 100, badframes)
    Why??
'''            
import glob
import tqdm
import run_s2p_parallel as s2pp
from suite2p.run_s2p import default_ops
import multiprocessing
if __name__ == '__main__':
    baseDir = 'D:/TPM/JK/h5/'
    # mice = [25,27,30,36,37,38,39,41]
    ##mice = [30,36,37,38,39,41]
    mice = [25]
    sessions = [5554,5555]

    ops = default_ops()
    ops['delete_bin'] = True
    ops['spikedetect'] = False
    db = []

    for mouse in mice:
        for session in sessions:
            if session == 5554:
                planes = [5,6,7,8]
            else:
                planes = [1,2,3,4]
            for pi in planes:
                planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
                for fileSource in glob.iglob(f'{planeDir}{mouse:03}_{session}_*_plane_{pi}.h5'):
                    filePath = fileSource.split('\\')
                    fileNameH5 = filePath[-1]
                    fileNameH5List = fileNameH5.split('.')
                    fileName = fileNameH5List[0]
                    dblist = {
                        'h5py': fileSource,
                        'h5py_key': ['data'],
                        'look_one_level_down': False,
                        'data_path': [],
                        'save_folder': planeDir,
                        'matSaveFn': f'{planeDir}{fileName}_mimg.mat'
                    }
                    db.append(dblist)

    with multiprocessing.Pool(processes = 10) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(s2pp.run_s2p_parallel, db), total=len(db)):
            pass

#%%

import par_test as pt
import multiprocessing
if __name__ == '__main__':
    itest = range(10)
    with multiprocessing.Pool(processes = 10) as pool:
        pool.map(pt.par_test, itest)
        
        
#%%
# Spontaneous 30-41
# Console 2
import glob
from suite2p.run_s2p import run_s2p, default_ops
from scipy.io import savemat

baseDir = 'D:/TPM/JK/h5/'
# mice = [25,27,30,36,37,38,39,41]
mice = [30,36,37,38,39,41]
sessions = [5554,5555]

ops = default_ops()
ops['delete_bin'] = True
ops['spikedetect'] = False
db = []

for mouse in mice:
    for session in sessions:
        if session == 5554:
            planes = [5,6,7,8]
        else:
            planes = [1,2,3,4]
        for pi in planes:
            planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
            for fileSource in glob.iglob(f'{planeDir}{mouse:03}_{session}_*_plane_{pi}.h5'):
                filePath = fileSource.split('\\')
                fileNameH5 = filePath[-1]
                fileNameH5List = fileNameH5.split('.')
                fileName = fileNameH5List[0]
                dblist = {
                    'h5py': fileSource,
                    'h5py_key': ['data'],
                    'look_one_level_down': False,
                    'data_path': [],
                    'save_folder': planeDir,
                    'matSaveFn': f'{planeDir}{fileName}_mimg.mat'
                }
                db.append(dblist)

for dbi in db:
    opsEnd = run_s2p(ops=ops, db=dbi)
    saveFn = dbi['matSaveFn']
    savemat(f'{saveFn}', {'mimg':opsEnd['meanImg'].transpose()})
        
#%%
# Spontaneous 38-41
# Console 5
import glob
from suite2p.run_s2p import run_s2p, default_ops
from scipy.io import savemat

baseDir = 'D:/TPM/JK/h5/'
# mice = [25,27,30,36,37,38,39,41]
mice = [38,39,41]
sessions = [5554,5555]

ops = default_ops()
ops['delete_bin'] = True
ops['spikedetect'] = False
db = []

for mouse in mice:
    for session in sessions:
        if session == 5554:
            planes = [5,6,7,8]
        else:
            planes = [1,2,3,4]
        for pi in planes:
            planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
            for fileSource in glob.iglob(f'{planeDir}{mouse:03}_{session}_*_plane_{pi}.h5'):
                filePath = fileSource.split('\\')
                fileNameH5 = filePath[-1]
                fileNameH5List = fileNameH5.split('.')
                fileName = fileNameH5List[0]
                dblist = {
                    'h5py': fileSource,
                    'h5py_key': ['data'],
                    'look_one_level_down': False,
                    'data_path': [],
                    'save_folder': planeDir,
                    'matSaveFn': f'{planeDir}{fileName}_mimg.mat'
                }
                db.append(dblist)

for dbi in db:
    opsEnd = run_s2p(ops=ops, db=dbi)
    saveFn = dbi['matSaveFn']
    savemat(f'{saveFn}', {'mimg':opsEnd['meanImg'].transpose()})
        
#%%
# Spontaneous 52-56
# Console 1
from scipy.io import savemat
import os, glob
from suite2p.run_s2p import run_s2p, default_ops

baseDir = 'D:/TPM/JK/h5/'
# mice = [52,53,54,56]
mice = [54,56]
sessions = [5554,5555]

ops = default_ops()
ops['delete_bin'] = True
ops['spikedetect'] = False
db = []

for mouse in mice:
    for session in sessions:
        if session == 5554:
            planes = [5,6,7,8]
        else:
            planes = [1,2,3,4]
        for pi in planes:
            planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
            for sessionDir in glob.iglob(f'{planeDir}{mouse:03}_{session}_*_plane_{pi}'):
                if os.path.isdir(sessionDir):
                    flist = glob.glob(f'{sessionDir}{os.path.sep}{mouse:03}_{session}_*.h5')
                    h5fn = flist[0]
                    saveFnTemplate = sessionDir.split('\\')[-1]
                    dblist = {
                        'h5py': h5fn,
                        'h5py_key': ['data'],
                        'look_one_level_down': True,
                        'data_path': [],
                        'save_folder': planeDir,
                        'matSaveFn': f'{planeDir}{saveFnTemplate}_mimg.mat'
                    }
                    db.append(dblist)

for dbi in db:
    opsEnd = run_s2p(ops=ops, db=dbi)
    saveFn = dbi['matSaveFn']
    savemat(f'{saveFn}', {'mimg':opsEnd['meanImg'].transpose()})
        
#%% 
# Passive deflection data, from multiple files (mice 25-56)
# Console 3
import glob, os
from suite2p.run_s2p import run_s2p, default_ops
from scipy.io import savemat

baseDir = 'D:/TPM/JK/h5/'
# mice = [25,27,30,36,37,38,39,41,52,53,54,56]
mice = [30,36,37,38,39]
sessions = [9998,9999]

ops = default_ops()
ops['delete_bin'] = True
ops['spikedetect'] = False
db = []

for mouse in mice:
    for session in sessions:
        if session == 9998:
            planes = [5,6,7,8]
        else:
            planes = [1,2,3,4]
        for pi in planes:
            planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
            for sessionDir in glob.iglob(f'{planeDir}{mouse:03}_{session}_*_plane_{pi}'):
                if os.path.isdir(sessionDir):
                    flist = glob.glob(f'{sessionDir}{os.path.sep}{mouse:03}_{session}_*.h5')
                    h5fn = flist[0]
                    saveFnTemplate = sessionDir.split('\\')[-1]
                    dblist = {
                        'h5py': h5fn,
                        'h5py_key': ['data'],
                        'look_one_level_down': True,
                        'data_path': [],
                        'save_folder': planeDir,
                        'matSaveFn': f'{planeDir}{saveFnTemplate}_mimg.mat'
                    }
                    db.append(dblist)
for dbi in db:
    opsEnd = run_s2p(ops=ops, db=dbi)
    saveFn = dbi['matSaveFn']
    savemat(f'{saveFn}', {'mimg':opsEnd['meanImg'].transpose()})
    
    
    
#%% 
# Passive deflection data, from multiple files (mice 25-56)  
# Console 4
import glob, os
from suite2p.run_s2p import run_s2p, default_ops
from scipy.io import savemat
baseDir = 'D:/TPM/JK/h5/'
# mice = [25,27,30,36,37,38,39,41,52,53,54,56]
mice = [41,52,53,54,56]
sessions = [9998,9999]

ops = default_ops()
ops['delete_bin'] = True
ops['spikedetect'] = False
db = []

for mouse in mice:
    for session in sessions:
        if session == 9998:
            planes = [5,6,7,8]
        else:
            planes = [1,2,3,4]
        for pi in planes:
            planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
            for sessionDir in glob.iglob(f'{planeDir}{mouse:03}_{session}_*_plane_{pi}'):
                if os.path.isdir(sessionDir):
                    flist = glob.glob(f'{sessionDir}{os.path.sep}{mouse:03}_{session}_*.h5')
                    h5fn = flist[0]
                    saveFnTemplate = sessionDir.split('\\')[-1]
                    dblist = {
                        'h5py': h5fn,
                        'h5py_key': ['data'],
                        'look_one_level_down': True,
                        'data_path': [],
                        'save_folder': planeDir,
                        'matSaveFn': f'{planeDir}{saveFnTemplate}_mimg.mat'
                    }
                    db.append(dblist)
for dbi in db:
    opsEnd = run_s2p(ops=ops, db=dbi)
    saveFn = dbi['matSaveFn']
    savemat(f'{saveFn}', {'mimg':opsEnd['meanImg'].transpose()}) 
    







    
#%%
# Some left tests
# Fixing errors for FOV depth matching
from suite2p.run_s2p import run_s2p, default_ops
from scipy.io import savemat

h5fn = ['D:/TPM/JK/h5/036/plane_1/036_010_001_plane_1.h5', 'D:/TPM/JK/h5/036/plane_5/036_010_001_plane_5.h5', 'D:/TPM/JK/h5/036/plane_5/036_013_002_plane_5.h5',
        'D:/TPM/JK/h5/036/plane_1/036_013_002_plane_1.h5']
ops = default_ops()
ops['tau'] = 1.5
ops['fs'] = 7.7
ops['do_bidiphase'] = True
db = {
        'h5py': D:/TPM/JK/h5/025/plane_1/025_,
        'h5py_key': ['data'],
        'look_one_level_down': False,
        'data_path': [],
        'save_folder': planeDir,
        'matSaveFn': f'{planeDir}{fileName}_mimg.mat'
    }