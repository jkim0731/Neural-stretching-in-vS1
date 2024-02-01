# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 22:32:51 2021

@author: shires
"""
# import par_test as pt
# import multiprocessing
# if __name__ == '__main__':
#     itest = range(10)
#     with multiprocessing.Pool(processes = 10) as pool:
#         pool.map(pt.par_test, itest)
        
        
        
        
        
        
import glob
import tqdm
import run_s2p_parallel as s2pp
from suite2p.run_s2p import default_ops
import multiprocessing
from functools import partial
from itertools import repeat
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
        pool.starmap(s2pp.run_s2p_parallel, zip(db, repeat(ops)))
        # for _ in tqdm.tqdm(pool.imap_unordered(s2pp.run_s2p_parallel, db), total=len(db)):
        #     pass