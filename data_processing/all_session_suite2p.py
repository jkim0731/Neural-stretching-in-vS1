"""
All session suite2p
Run suite2p from all sessions of the same plane
Including spontaneous and passive deflection sessions
"""

from suite2p.run_s2p import run_s2p, default_ops
import glob, os

baseDir = 'D:/TPM/JK/h5/'
mice = [25,  27,  30,  36,  37,  38,  39,  41,  52,  53,  54,  56]
zoom = [2,   2,   2,   1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
freq = [7.7, 7.7, 7.7, 7.7, 6.1, 6.1, 6.1, 6.1, 7.7, 7.7, 7.7, 1.7 ]
planes= range(1,9)
ops = default_ops()
ops['tau'] = 1.5

ops['look_one_level_down'] = True
ops['do_bidiphase'] = True
ops['nimg_init'] = 1000
ops['batch_size'] = 5000
ops['save_mat'] = True
ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works

# for mi in [2,3]:
#     mouse = mice[mi]
#     ops['fs'] = freq[mi]
#     ops['zoom'] = zoom[mi]
#     ops['umPerPix'] = 1.4/ops['zoom']
#     # for planei in range(1,7):
#     #     mouseDir = f'{baseDir}{mouse:03}/'
#     #     planeDir = f'{mouseDir}plane_{planei}/'
#     #     tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{planei}.h5')
#     #     repFn = tempFnList[0]
#     #     db = {'h5py': repFn,
#     #             'h5py_key': ['data'],
#     #             'look_one_level_down': True,
#     #             'data_path': [],
#     #             'save_path0': planeDir,
#     #             'save_folder': 'all_session',
#     #             'fast_disk': f'{planeDir}/all_session', # to prevent a collision with other suit2p runs
#     #             'move_bin': True # moving data.bin to ops['save_path'], i.e., io.path.join(ops['save_path0'], ops['save_folder'])
#     #                              # this nullifies 'delete_bin'
#     #         }
#     #     run_s2p(ops,db)
    
#     for planei in range(1,9):
#         mouseDir = f'{baseDir}{mouse:03}/'
#         planeDir = f'{mouseDir}plane_{planei}/'
#         if not os.path.isdir(f'{planeDir}all_session'):
#             tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{planei}.h5')
#             repFn = tempFnList[0]
#             db = {'h5py': repFn,
#                     'h5py_key': ['data'],
#                     'look_one_level_down': True,
#                     'data_path': [],
#                     'save_path0': planeDir,
#                     'save_folder': 'all_session',
#                     'fast_disk': f'{planeDir}/all_session',
#                     'move_bin': True,
#                     'do_registration': 2, # forcing re-registration
#                     # Addendum for low SNR
#                     'two_step_registration': True,
#                     'keep_movie_raw': True,
#                     'smooth_sigma_time': 2
#                 }
#             run_s2p(ops,db)

mi = 7
mouse = mice[mi]
ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']
for planei in range(5,7):
    mouseDir = f'{baseDir}{mouse:03}/'
    planeDir = f'{mouseDir}plane_{planei}/'
    if not os.path.isdir(f'{planeDir}all_session'):
        tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{planei}.h5')
        repFn = tempFnList[0]
        db = {'h5py': repFn,
                'h5py_key': ['data'],
                'look_one_level_down': True,
                'data_path': [],
                'save_path0': planeDir,
                'save_folder': 'all_session',
                'fast_disk': f'{planeDir}/all_session',
                'move_bin': True,
                'do_registration': 2, # forcing re-registration
                # Addendum for low SNR
                'two_step_registration': True,
                'keep_movie_raw': True,
                'smooth_sigma_time': 2
            }
        run_s2p(ops,db)

#%%
# import h5py, glob
# import numpy as np

# baseDir = 'D:/TPM/JK/h5/'
# mice = [25,27,30,36,37,38,39,41,52,53,54,56]
# planei = 1
# mi = 2
# mouse = mice[mi]
# mouseDir = f'{baseDir}{mouse:03}/'
# planeDir = f'{mouseDir}plane_{planei}/'
# tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{planei}.h5')
# sizes = np.zeros((len(tempFnList),3))
# for i, fn in enumerate(tempFnList):
#     hf = h5py.File(fn, 'r')
#     sizes[i,:] = hf.get('data').shape
    