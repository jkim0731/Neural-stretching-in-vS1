
"""
Single session suite2p running
"""

from suite2p.run_s2p import run_s2p, default_ops
import glob

baseDir = 'D:/TPM/JK/h5/'
mice = [25,27,30,36,37,38,39,41,52,53,54,56]
freqs = [7.7, 7.7, 7.7, 7.7, 6.1, 6.1, 6.1, 6.1, 7.7, 7.7, 7.7]
planes= range(1,9)
ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['do_bidiphase'] = True
ops['nimg_init'] = 500
ops['batch_size'] = 1000
ops['save_mat'] = True
ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works

mi = 0
mouse = mice[mi]
ops['fs'] = freqs[mi]

session = 22
for planei in range(5,9):
    mouseDir = f'{baseDir}{mouse:03}/'
    planeDir = f'{mouseDir}plane_{planei}/'
    flist = glob.glob(f'{planeDir}{mouse:03}_{session:03}_*_plane_{planei}.h5')
    db = {'h5py': flist,
            'h5py_key': ['data'],
            'look_one_level_down': False,
            'data_path': [],
            'save_path0': planeDir,
            'save_folder': f'{session:03}',
            'fast_disk': f'{planeDir}/{session:03}',
            'move_bin': True,
            # Addendum for low SNR
            'two_step_registration': True,
            'keep_movie_raw': True, # this HAS to be 'True' for 'two_step_registration' to work
            'smooth_sigma_time': 2 # 0, 1, or 2
        }
    run_s2p(ops,db)


session = 25
for planei in range(1,9):
    mouseDir = f'{baseDir}{mouse:03}/'
    planeDir = f'{mouseDir}plane_{planei}/'
    flist = glob.glob(f'{planeDir}{mouse:03}_{session:03}_*_plane_{planei}.h5')
    db = {'h5py': flist,
            'h5py_key': ['data'],
            'look_one_level_down': False,
            'data_path': [],
            'save_path0': planeDir,
            'save_folder': f'{session:03}',
            'fast_disk': f'{planeDir}/{session:03}',
            'move_bin': True,
            # Addendum for low SNR
            'two_step_registration': True,
            'keep_movie_raw': True, # this HAS to be 'True' for 'two_step_registration' to work
            'smooth_sigma_time': 2 # 0, 1, or 2
        }
    run_s2p(ops,db)

mi = 4
mouse = mice[mi]
ops['fs'] = freqs[mi]

sessions = [901,1,6,7,8,13,19,23]
for session in sessions:
    for planei in range(1,9):
        mouseDir = f'{baseDir}{mouse:03}/'
        planeDir = f'{mouseDir}plane_{planei}/'
        flist = glob.glob(f'{planeDir}{mouse:03}_{session:03}_*_plane_{planei}.h5')
        db = {'h5py': flist,
                'h5py_key': ['data'],
                'look_one_level_down': False,
                'data_path': [],
                'save_path0': planeDir,
                'save_folder': f'{session:03}',
                'fast_disk': f'{planeDir}/{session:03}',
                'move_bin': True,
                # Addendum for low SNR
                'two_step_registration': True,
                'keep_movie_raw': True, # this HAS to be 'True' for 'two_step_registration' to work
                'smooth_sigma_time': 2 # 0, 1, or 2
            }
        run_s2p(ops,db)


#%%
# print([db['h5py']])

# %%
# Multi-file fixing for FOV depth matching

# from suite2p.run_s2p import run_s2p, default_ops
# import glob

# baseDir = 'D:/TPM/JK/h5/'
# mice = [36,36,41,41,41,52,53,54]
# sessions = [10,13,8,14,20,25,12,18]
# freqs = [7.7, 7.7, 6.1, 6.1, 6.1, 7.7, 7.7, 7.7]
# zoom = [1.7,1.7,1.7,1.7,1.7,1.7,1.7,1.7]
# planes= range(1,9)
# ops = default_ops()
# ops['tau'] = 1.5
# ops['look_one_level_down'] = False
# ops['do_bidiphase'] = True
# ops['nimg_init'] = 500
# ops['batch_size'] = 1000
# ops['save_mat'] = True

# params = [(False, 0), (True,0), (True, 1), (True, 2)]
# # for mi, mouse in enumerate(mice):
# # for mi in range(1,9):
# for mi in range(4,9):
#     mouse = mice[mi]
#     session = sessions[mi]
#     ops['fs'] = freqs[mi]
#     ops['zoom'] = zoom[mi]
#     ops['umPerPix'] = 1.4/ops['zoom']
#     for planei in range(1,9):
#         mouseDir = f'{baseDir}{mouse:03}/'
#         planeDir = f'{mouseDir}plane_{planei}/'
#         flist = glob.glob(f'{planeDir}{mouse:03}_{session:03}_*_plane_{planei}.h5')
#         for pi, param in enumerate(params):
#             db = {'h5py': flist,
#                 'h5py_key': ['data'],
#                 'look_one_level_down': False,
#                 'data_path': [],
#                 'save_path0': planeDir,
#                 'save_folder': f'{session:03}_param{pi}',
#                 'fast_disk': f'{planeDir}/{session:03}_param{pi}',
#                 'move_bin': True,
#                 # Addendum for low SNR
#                 'two_step_registration': param[0],
#                 'keep_movie_raw': param[0],
#                 'smooth_sigma_time': param[1]
#                 }
#             run_s2p(ops,db)

#%%
# Test run

# from suite2p.run_s2p import run_s2p, default_ops
# import glob

# ops = default_ops()

# baseDir = 'D:/TPM/JK/h5/'
# mouse = 25
# session = 1
# ops['fs'] = 7.7
# ops['zoom'] = 1.7
# ops['umPerPix'] = 1.4/ops['zoom']

# planes= range(1,9)

# ops['tau'] = 1.5
# ops['look_one_level_down'] = False
# ops['do_bidiphase'] = True
# ops['nimg_init'] = 500
# ops['batch_size'] = 1000
# ops['save_mat'] = True


# ops['umPerPix'] = 1.4/ops['zoom']
# # for planei in range(1,9):
# planei = 1
# mouseDir = f'{baseDir}{mouse:03}/'
# planeDir = f'{mouseDir}plane_{planei}/'
# flist = glob.glob(f'{planeDir}{mouse:03}_{session:03}_*_plane_{planei}.h5')
# db = {'h5py': flist,
#     'h5py_key': ['data'],
#     'look_one_level_down': False,
#     'data_path': [],
#     'save_path0': planeDir,
#     'save_folder': f'{session:03}_test',
#     'fast_disk': f'{planeDir}/{session:03}_test',
#     'delete_bin': True,
#     }
# run_s2p(ops,db)