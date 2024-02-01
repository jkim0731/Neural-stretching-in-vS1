# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:52:35 2021

Re-running roi detection using pre-registered data

@author: jinho
"""
import numpy as np
from suite2p.run_s2p import run_s2p, default_ops
import os, shutil
from auto_pre_cur import auto_pre_cur

baseDir = 'D:/TPM/JK/h5/'

mice = [25,  27,  30,  36,  37,  38,  39,  41,  52,  53,  54,  56]
zoom = [2,   2,   2,   1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7]
freq = [7.7, 7.7, 7.7, 7.7, 6.1, 6.1, 6.1, 6.1, 7.7, 7.7, 7.7, 7.7]

# planeDir = 'D:/TPM/JK/h5/037/plane_8/'

# ops = default_ops()
# ops['tau'] = 1.5
# ops['look_one_level_down'] = False
# ops['save_mat'] = True
# ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works
# ops['do_registration'] = 0 # Forcing to not run registration

# ops['fs'] = 6.1
# ops['zoom'] = 1.7
# ops['umPerPix'] = 1.4/ops['zoom']

# threshold_scaling = [0.7, 0.1]
# max_iteration = [20, 100]
# spatial_scale = [0, 1]
# nbins = [5000, 1000000]

# parami = 0
# for ts in threshold_scaling:
#     for maxiter in max_iteration:
#         for ss in spatial_scale:
#             for nbin in nbins:
#                 parami += 1
#                 db = {'data_path': [],
#                         'do_registration': 0, # Forcing to not run registration
#                         'save_path0': planeDir,
#                         'save_folder': 'all_session',
#                         'smooth_sigma_time': 2,
#                         'rerun_jk': 1,
#                         'allow_overlap': True,
#                         'threshold_scaling': ts,
#                         'max_iterations': maxiter,
#                         'spatial_scale': ss,
#                         'nbinned': nbin
#                     }                

#                 savePath = f'{planeDir}all_session/plane0/param{parami}/'
#                 if not os.path.isdir(savePath):              
#                     os.mkdir(savePath)
                    
#                     run_s2p(ops,db)
#                     flist = [f.name for f in os.scandir(f'{planeDir}all_session/plane0') if f.is_file()]
#                     for fn in flist:
#                         if fn[-4:] != '.bin':
#                             shutil.copy(f'{planeDir}all_session/plane0/{fn}', f'{savePath}{fn}')

#%%
# # 2021/05/12 copying previous cell detection files into "param0" sub-directory
# import os, shutil
# mouse = 25
# for pi in range(2,9):
#     tempPath = f'D:\\TPM\\JK\\h5\\{mouse:03}\\plane_{pi}\\all_session\\plane0\\'
#     savePath = f'D:\\TPM\\JK\\h5\\{mouse:03}\\plane_{pi}\\all_session\\plane0\\param0\\'
#     os.mkdir(savePath)
#     flist = [f.name for f in os.scandir(tempPath) if f.is_file()]
#     for fn in flist:
#         if fn[-4:] != '.bin':
#             shutil.copy(f'{tempPath}{fn}', f'{savePath}{fn}')

# #%%
# mouse = 37
# for pi in range(1,8):
#     tempPath = f'D:\\TPM\\JK\\h5\\{mouse:03}\\plane_{pi}\\all_session\\plane0\\'
#     savePath = f'D:\\TPM\\JK\\h5\\{mouse:03}\\plane_{pi}\\all_session\\plane0\\param0\\'
#     os.mkdir(savePath)
#     flist = [f.name for f in os.scandir(tempPath) if f.is_file()]
#     for fn in flist:
#         if fn[-4:] != '.bin':
#             shutil.copy(f'{tempPath}{fn}', f'{savePath}{fn}')
                        
#%% 2021/05/12
# mi = 0
# for pi in range(1,9):
#     mouse = mice[mi]
#     planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
#     ops = default_ops()
#     ops['tau'] = 1.5
#     ops['look_one_level_down'] = False
#     ops['save_mat'] = True
#     ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works
#     ops['do_registration'] = 0 # Forcing to not run registration
    
#     ops['fs'] = freq[mi]
#     ops['zoom'] = zoom[mi]
#     ops['umPerPix'] = 1.4/ops['zoom']
    
#     db = {'data_path': [],
#             'do_registration': 0, # Forcing to not run registration
#             'save_path0': planeDir,
#             'save_folder': 'all_session',
#             'smooth_sigma_time': 2,
#             'rerun_jk': 1,
#             'allow_overlap': True,
#             'threshold_scaling': 0.1
#         }
#     savePath = f'{planeDir}all_session/plane0/param9/'
#     if not os.path.isdir(savePath):
#         os.mkdir(savePath)
        
#         run_s2p(ops,db)
#         flist = [f.name for f in os.scandir(f'{planeDir}all_session/plane0') if f.is_file()]
#         for fn in flist:
#             if fn[-4:] != '.bin':
#                 shutil.copy(f'{planeDir}all_session/plane0/{fn}', f'{savePath}{fn}')
                
# mi = 4
# for pi in range(1,8):
#     mouse = mice[mi]
#     planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
#     ops = default_ops()
#     ops['tau'] = 1.5
#     ops['look_one_level_down'] = False
#     ops['save_mat'] = True
#     ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works
#     ops['do_registration'] = 0 # Forcing to not run registration
    
#     ops['fs'] = freq[mi]
#     ops['zoom'] = zoom[mi]
#     ops['umPerPix'] = 1.4/ops['zoom']
    
#     db = {'data_path': [],
#             'do_registration': 0, # Forcing to not run registration
#             'save_path0': planeDir,
#             'save_folder': 'all_session',
#             'smooth_sigma_time': 2,
#             'rerun_jk': 1,
#             'allow_overlap': True,
#             'threshold_scaling': 0.1
#         }                

#     savePath = f'{planeDir}all_session/plane0/param9/'
#     if not os.path.isdir(savePath):
#         os.mkdir(savePath)
        
#         run_s2p(ops,db)
#         flist = [f.name for f in os.scandir(f'{planeDir}all_session/plane0') if f.is_file()]
#         for fn in flist:
#             if fn[-4:] != '.bin':
#                 shutil.copy(f'{planeDir}all_session/plane0/{fn}', f'{savePath}{fn}')

#%% 2021/05/13
# thresholdList = np.arange(0.1,0.6,0.1)
# paramNumList = np.arange(21,26)
# mi = 0
# pi = 6
# mouse = mice[mi]
# planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
# ops = default_ops()
# ops['tau'] = 1.5
# ops['look_one_level_down'] = False
# ops['save_mat'] = True
# ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works

# ops['fs'] = freq[mi]
# ops['zoom'] = zoom[mi]
# ops['umPerPix'] = 1.4/ops['zoom']
    
# for i in range(len(thresholdList)):
#     db = {'data_path': [],
#             'do_registration': 0, # Forcing to not run registration
#             'save_path0': planeDir,
#             'save_folder': 'all_session',
#             'smooth_sigma_time': 2,
#             'rerun_jk': 1,
#             'allow_overlap': False,
#             'max_overlap': 0.3,
#             'threshold_scaling': thresholdList[i]
#         }
#     savePath = f'{planeDir}all_session/plane0/param{paramNumList[i]}/'
#     if not os.path.isdir(savePath):
#         os.mkdir(savePath)
        
#         run_s2p(ops,db)
#         flist = [f.name for f in os.scandir(f'{planeDir}all_session/plane0') if f.is_file()]
#         for fn in flist:
#             if fn[-4:] != '.bin':
#                 shutil.copy(f'{planeDir}all_session/plane0/{fn}', f'{savePath}{fn}')
                
# mi = 4
# pi = 8
# mouse = mice[mi]
# planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
# ops = default_ops()
# ops['tau'] = 1.5
# ops['look_one_level_down'] = False
# ops['save_mat'] = True
# ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works
# ops['do_registration'] = 0

# ops['fs'] = freq[mi]
# ops['zoom'] = zoom[mi]
# ops['umPerPix'] = 1.4/ops['zoom']
    
# for i in range(len(thresholdList)):
#     db = {'data_path': [],
#             'do_registration': 0, # Forcing to not run registration
#             'save_path0': planeDir,
#             'save_folder': 'all_session',
#             'smooth_sigma_time': 2,
#             'rerun_jk': 1,
#             'allow_overlap': False,
#             'max_overlap': 0.3,
#             'threshold_scaling': thresholdList[i]
#         }
#     savePath = f'{planeDir}all_session/plane0/param{paramNumList[i]}/'
#     if not os.path.isdir(savePath):
#         os.mkdir(savePath)
        
#         run_s2p(ops,db)
#         flist = [f.name for f in os.scandir(f'{planeDir}all_session/plane0') if f.is_file()]
#         for fn in flist:
#             if fn[-4:] != '.bin':
#                 shutil.copy(f'{planeDir}all_session/plane0/{fn}', f'{savePath}{fn}')
#%% 2021/05/18
# thresholdList = np.arange(0.6,1.0,0.1)
# paramNumList = np.arange(26,30)
# mi = 0
# pi = 6
# mouse = mice[mi]
# planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
# ops = default_ops()
# ops['tau'] = 1.5
# ops['look_one_level_down'] = False
# ops['save_mat'] = False
# ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works

# ops['fs'] = freq[mi]
# ops['zoom'] = zoom[mi]
# ops['umPerPix'] = 1.4/ops['zoom']
    
# for i in range(len(thresholdList)):
#     db = {'data_path': [],
#             'do_registration': 0, # Forcing to not run registration
#             'save_path0': planeDir,
#             'save_folder': 'all_session',
#             'smooth_sigma_time': 2,
#             'rerun_jk': 1,
#             'allow_overlap': False,
#             'max_overlap': 0.3,
#             'threshold_scaling': thresholdList[i]
#         }
#     savePath = f'{planeDir}all_session/plane0/param{paramNumList[i]}/'
#     if not os.path.isdir(savePath):
#         os.mkdir(savePath)
        
#         run_s2p(ops,db)
#         flist = [f.name for f in os.scandir(f'{planeDir}all_session/plane0') if f.is_file()]
#         for fn in flist:
#             if fn[-4:] != '.bin':
#                 shutil.copy(f'{planeDir}all_session/plane0/{fn}', f'{savePath}{fn}')


# thresholdList = np.arange(0.1,0.6,0.1)
# paramNumList = np.arange(21,26)
# mi = 4
# for pi in range(1,8):
#     mouse = mice[mi]
#     planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
#     ops = default_ops()
#     ops['tau'] = 1.5
#     ops['look_one_level_down'] = False
#     ops['save_mat'] = True
#     ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works
#     ops['do_registration'] = 0
    
#     ops['fs'] = freq[mi]
#     ops['zoom'] = zoom[mi]
#     ops['umPerPix'] = 1.4/ops['zoom']
        
#     for i in range(len(thresholdList)):
#         db = {'data_path': [],
#                 'do_registration': 0, # Forcing to not run registration
#                 'save_path0': planeDir,
#                 'save_folder': 'all_session',
#                 'smooth_sigma_time': 2,
#                 'rerun_jk': 1,
#                 'allow_overlap': False,
#                 'max_overlap': 0.3,
#                 'threshold_scaling': thresholdList[i]
#             }
#         savePath = f'{planeDir}all_session/plane0/param{paramNumList[i]}/'
#         if not os.path.isdir(savePath):
#             os.mkdir(savePath)
            
#             run_s2p(ops,db)
#             flist = [f.name for f in os.scandir(f'{planeDir}all_session/plane0') if f.is_file()]
#             for fn in flist:
#                 if fn[-4:] != '.bin':
#                     shutil.copy(f'{planeDir}all_session/plane0/{fn}', f'{savePath}{fn}')
#%%
# from auto_pre_cur import auto_pre_cur
# paramNumList = np.arange(26,30)
# mi = 0
# pi = 6
# mouse = mice[mi]
# planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
# for i in paramNumList:
#     savePath = f'{planeDir}all_session/plane0/param{i}/'
#     auto_pre_cur(savePath)

# paramNumList = np.arange(21,26)
# mi = 4
# mouse = mice[mi]
# for pi in range(1,8):
#     planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
#     for i in paramNumList:
#         savePath = f'{planeDir}all_session/plane0/param{i}/'
#         auto_pre_cur(savePath)
        
#%% 
# thresholdList = np.arange(0.1,1.0,0.1)
# paramNumList = np.arange(21,30)

# mi = 2

# mouse = mice[mi]
# ops = default_ops()
# ops['tau'] = 1.5
# ops['look_one_level_down'] = False
# ops['save_mat'] = False
# ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works
# ops['fs'] = freq[mi]
# ops['zoom'] = zoom[mi]
# ops['umPerPix'] = 1.4/ops['zoom']
# for pi in [2,5,8]:
#     planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
#     for i in range(len(thresholdList)):
#         db = {'data_path': [],
#                 'do_registration': 0, # Forcing to not run registration
#                 'save_path0': planeDir,
#                 'save_folder': 'all_session',
#                 'smooth_sigma_time': 2,
#                 'rerun_jk': 1,
#                 'allow_overlap': False,
#                 'max_overlap': 0.3,
#                 'threshold_scaling': thresholdList[i]
#             }
#         savePath = f'{planeDir}all_session/plane0/param{paramNumList[i]}/'
#         if not os.path.isdir(savePath):
#             os.mkdir(savePath)
            
#             run_s2p(ops,db)
#             flist = [f.name for f in os.scandir(f'{planeDir}all_session/plane0') if f.is_file()]
#             for fn in flist:
#                 if fn[-4:] != '.bin':
#                     shutil.copy(f'{planeDir}all_session/plane0/{fn}', f'{savePath}{fn}')
#             auto_pre_cur(savePath)

#%% 2021/05/23
# # After applying edge and vessel ROIs removal
# # before manual curation
# paramNumList = np.arange(21,26)
# mi = 4
# mouse = mice[mi]
# for pi in range(1,8):
#     planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
#     for i in paramNumList:
#         savePath = f'{planeDir}all_session/plane0/param{i}/'
#         auto_pre_cur(savePath)

#%%
# paramNumList = np.arange(21,30)
# mi = 2
# mouse = mice[mi]
# for pi in [2,5,8]:
#     planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
#     for i in paramNumList:
#         savePath = f'{planeDir}all_session/plane0/param{i}/'
#         auto_pre_cur(savePath)

#%% 2021/05/24
# thresholdList = np.arange(0.1,1.0,0.1)
# paramNumList = np.arange(21,30)

# mi = 3

# mouse = mice[mi]
# ops = default_ops()
# ops['tau'] = 1.5
# ops['look_one_level_down'] = False
# ops['save_mat'] = False
# ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works
# ops['fs'] = freq[mi]
# ops['zoom'] = zoom[mi]
# ops['umPerPix'] = 1.4/ops['zoom']
# for pi in range(1,9):
#     planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
#     for i in range(len(thresholdList)):
#         db = {'data_path': [],
#                 'do_registration': 0, # Forcing to not run registration
#                 'save_path0': planeDir,
#                 'save_folder': 'all_session',
#                 'smooth_sigma_time': 2,
#                 'rerun_jk': 1,
#                 'allow_overlap': False,
#                 'max_overlap': 0.3,
#                 'threshold_scaling': thresholdList[i]
#             }
#         savePath = f'{planeDir}all_session/plane0/param{paramNumList[i]}/'
#         if not os.path.isdir(savePath):
#             os.mkdir(savePath)
            
#             run_s2p(ops,db)
#             flist = [f.name for f in os.scandir(f'{planeDir}all_session/plane0') if f.is_file()]
#             for fn in flist:
#                 if fn[-4:] != '.bin':
#                     shutil.copy(f'{planeDir}all_session/plane0/{fn}', f'{savePath}{fn}')
#             auto_pre_cur(savePath)

#%%
# mi = 0
# mouse = mice[mi]
# paramNumList = np.arange(21,30)
# pi = 6
# planeDir = f'{baseDir}{mouse:03}/plane_{pi}/'
# for i in paramNumList:
#     savePath = f'{planeDir}all_session/plane0/param{i}/'
#     auto_pre_cur(savePath)



#%% 210609 testing 0 threshold
mi = 0
ops = default_ops()
ops['tau'] = 1.5
ops['look_one_level_down'] = False
ops['save_mat'] = False
ops['save_NWB'] = False # for now. Need to set up parameters and confirm it works
ops['fs'] = freq[mi]
ops['zoom'] = zoom[mi]
ops['umPerPix'] = 1.4/ops['zoom']
planeDir = 'D:/TPM/JK/h5/025/plane_1/'
db = {'data_path': [],
    'do_registration': 0, # Forcing to not run registration
    'save_path0': planeDir,
    'save_folder': 'all_session',
    'smooth_sigma_time': 2,
    'rerun_jk': 1,
    'allow_overlap': False,
    'max_overlap': 0.3,
    'threshold_scaling': 0
}

run_s2p(ops,db)
# It works

