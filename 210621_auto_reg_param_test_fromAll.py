"""
Too many combinations of parameters to change.
Make it automatic, and select the best parameter.
QC: Both phase correlation and intensity correlation at the center of FOV
(Crop edges that can be circshifted. It depends on mouse.)
"""
import h5py
import numpy as np
from suite2p.run_s2p import run_s2p, default_ops
import os, glob, shutil
import matplotlib.pyplot as plt
from skimage import exposure
import napari
import ffmpeg
import cv2
from suite2p.io.binary import BinaryFile
import time

# CLAHE each mean images
def clahe_each(img):
    newimg = (img - np.amin(img)) / (np.amax(img) - np.amin(img)) * (2**16-1)
    newimg = exposure.equalize_adapthist(newimg.astype(np.uint16))
    return newimg

def phase_corr(a,b):
    if a.shape != b.shape:
        raise('Dimensions must match')
    R = np.fft.fft2(a) * np.fft.fft2(b).conj()
    R /= np.absolute(R)
    r = np.absolute(np.fft.ifft2(R))
    ymax, xmax = np.unravel_index(np.argmax(r), r.shape)
    cmax = np.amax(r)
    center = r[0, 0]    
    return ymax, xmax, cmax, center, r

h5Dir = 'D:/TPM/JK/h5/'
mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]
zoom =          [2,     2,    2,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7,    1.7]
freq =          [7.7,   7.7,  7.7,  7.7,    6.1,    6.1,    6.1,    6.1,    7.7,    7.7,    7.7,    7.7]

framesPerSession = 50
#%%
for mi in [8]:
    mouse = mice[mi]
    refS = refSessions[mi]
    for planei in range(5,6):
        mouseDir = f'{h5Dir}{mouse:03}/'
        planeDir = f'{mouseDir}plane_{planei}/'
        
        #%% First, make ref session registration
        # Check if there's one. If not, then run suite2p registration with the default ops (two-step registration)
        refOpsFn= f'{planeDir}{refS:03}/plane0/ops.npy'
        if not os.path.isfile(refOpsFn):
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
            
            refFnList = glob.glob(f'{planeDir}{mouse:03}_{refS:03}_*_plane_{planei}.h5')
            db = {'h5py': refFnList,
                'h5py_key': ['data'],
                'data_path': [],
                'save_path0': planeDir,
                'save_folder': f'{refS:03}',
                'fast_disk': f'{planeDir}/{refS:03}',
                'roidetect': False,
            }
            run_s2p(ops,db)
        ops = np.load(refOpsFn, allow_pickle=True).item()
        refImg = ops['meanImg']
        
        #%% pick linearly separated 'framesPerSession' frames from each session, make a new h5 file
        # multiple sbx files (h5 files) from a single session should be merged
        # Piezo sessions should be merged first
        # For mouse > 50, Spont sessions should also be merged
        tempFnList = glob.glob(f'{planeDir}{mouse:03}_*_plane_{planei}.h5')    
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
            
        #%% Select linearly separated frames from each session.
        # Combine if multiple files in a session
        f = h5py.File(tempFnList[0], 'r')
        data = f['data']
        _, height, width = data.shape
        wfn = f'{planeDir}selected.h5'
        if not os.path.isfile(wfn):
            newdata = np.zeros((len(sessionNames)*framesPerSession, height, width), dtype = np.uint16)
            for i, fnlist in enumerate(sessionFiles):
                for j, fn in enumerate(fnlist):
                    f = h5py.File(fn, 'r')
                    if j == 0:
                        data = f['data']
                    else:
                        data = np.concatenate((data, f['data']), axis=0)
                numFrames, height, width = data.shape
                frames = np.linspace(0,numFrames-1, num=framesPerSession, dtype=int)
                for j in range(len(frames)):
                    newdata[i*framesPerSession+j, :, :] = data[frames[j],:,:]    
            with h5py.File(wfn, 'w') as wf:
                wf.create_dataset('data', data=newdata, dtype='uint16')
    
        #%% Second test (2021/07/02~)
        # wfn = f'{planeDir}selected.h5'
        ops = default_ops()
        ops['tau'] = 1.5
        ops['look_one_level_down'] = False
        ops['do_bidiphase'] = True
        ops['batch_size'] = 5000
        ops['two_step_registration'] = True
        ops['keep_movie_raw'] = True
        ops['smooth_sigma_time'] = 2
        ops['move_bin'] = True
        ops['fs'] = freq[mi]
        ops['zoom'] = zoom[mi]
        ops['umPerPix'] = 1.4/ops['zoom']
        ops['force_refImg'] = True
        ops['refImg'] = refImg
        
        maxregshiftNRList = [5, 10]
        block_sizeList = [[64, 64], [48, 48], [32, 32]]
        snr_threshList = [1.2, 1.3, 1.4, 1.5]
        
        paramSetInd = 200   
        
        ops['nonrigid'] = True
        ops['maxregshift'] = 0.3
        for mrsn in maxregshiftNRList:
            for bs in block_sizeList:
                for st in snr_threshList:
                    testDn = f'test{paramSetInd:02}'
                    
                    ops['maxregshiftNR'] = mrsn
                    ops['block_size'] = bs
                    ops['snr_thresh'] = st
                    
                    db = {'h5py': wfn,
                        'h5py_key': ['data'],
                        'data_path': [],
                        'save_path0': planeDir,
                        'save_folder': testDn,
                        'fast_disk': f'{planeDir}/{testDn}',
                        'roidetect': False,
                        'testFileList': sessionFiles,
                        'testSessionNames': sessionNames,
                        'framesPerSession': framesPerSession,
                    }
                    run_s2p(ops,db)
                    paramSetInd += 1
                
    # %% First test (~2021/07/01)
    # wfn = f'{planeDir}selected.h5'
    # ops = default_ops()
    # ops['tau'] = 1.5
    # ops['look_one_level_down'] = False
    # ops['do_bidiphase'] = True
    # ops['batch_size'] = 5000
    # ops['two_step_registration'] = True
    # ops['keep_movie_raw'] = True
    # ops['smooth_sigma_time'] = 2
    # ops['move_bin'] = True
    # ops['fs'] = freq[mi]
    # ops['zoom'] = zoom[mi]
    # ops['umPerPix'] = 1.4/ops['zoom']
    # ops['force_refImg'] = True
    # ops['refImg'] = refImg
    
    # maxregshiftList = [0.1, 0.2, 0.3]
    # maxregshiftNRList = [5, 10, 20]
    # block_sizeList = [[128,128], [96, 96], [64, 64]]
    # snr_threshList = [1, 1.1, 1.2, 1.3]
    
    # paramSetInd = 0
    
    # ops['nonrigid'] = False
    # for mrs in maxregshiftList:
    #     testDn = f'test{paramSetInd:02}'        
    #     ops['maxregshift'] = mrs
        
    #     db = {'h5py': wfn,
    #         'h5py_key': ['data'],
    #         'data_path': [],
    #         'save_path0': planeDir,
    #         'save_folder': testDn,
    #         'fast_disk': f'{planeDir}/{testDn}',
    #         'roidetect': False,
    #         'testFileList': sessionFiles,
    #         'testSessionNames': sessionNames,
    #         'framesPerSession': framesPerSession,
    #     }
    #     run_s2p(ops,db)
    #     paramSetInd += 1
    
    # ops['nonrigid'] = True
    # for mrs in maxregshiftList:
    #     for mrsn in maxregshiftNRList:
    #         for bs in block_sizeList:
    #             for st in snr_threshList:
    #                 testDn = f'test{paramSetInd:02}'
                    
    #                 ops['maxregshift'] = mrs
    #                 ops['maxregshiftNR'] = mrsn
    #                 ops['block_size'] = bs
    #                 ops['snr_thresh'] = st
                    
    #                 db = {'h5py': wfn,
    #                     'h5py_key': ['data'],
    #                     'data_path': [],
    #                     'save_path0': planeDir,
    #                     'save_folder': testDn,
    #                     'fast_disk': f'{planeDir}/{testDn}',
    #                     'roidetect': False,
    #                     'testFileList': sessionFiles,
    #                     'testSessionNames': sessionNames,
    #                     'framesPerSession': framesPerSession,
    #                 }
    #                 run_s2p(ops,db)
    #                 paramSetInd += 1
        

# #%% Visual inspection to decide FOV proportion selection

# dataDir = f'{planeDir}/{testDn}/plane0/'
# binfn = f'{dataDir}data.bin'
# opsfn = f'{dataDir}ops.npy'
# ops = np.load(opsfn, allow_pickle = True).item()
# Ly = ops['Ly']
# Lx = ops['Lx']
# nframes = ops['nframes']
# framesPerSession = ops['framesPerSession']
# numSessions = int(nframes/framesPerSession)
# # perFileMeanImg = np.zeros(shape = (Ly,Lx,numSessions))
# viewer = napari.Viewer()
# with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
#     for i in range(numSessions):
#         inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
#         frames = f.ix(indices=inds).astype(np.float32)
#         # perFileMeanImg[:,:,i] = frames.mean(axis=0)
#         viewer.add_image(frames.mean(axis=0))
        
        
# #%% Set top bottom left right margin in "pixel number"
# topMargin = 100
# bottomMargin = 20
# leftMargin = 20
# rightMargin = 50

# phaseCorr = np.zeros(shape = (paramSetInd,numSessions))
# intensityCorr = np.zeros(shape = (paramSetInd,numSessions))

# #%% 
# tic = time.time()
# framesPerSession = 20
# mouse = 25
# plane = 1

# planeDir = f'{mouseDir}plane_{plane}/'

# meanPC = np.zeros(111)
# meanIC = np.zeros(111)
# for testn in range(111):
#     testDir = f'{planeDir}/test{testn:02}/plane0/'
#     ops = np.load(f'{testDir}ops.npy', allow_pickle=True).item()
#     fpsession = ops['framesPerSession']
#     numSessions = len(ops['testFileList'])
#     binfn = f'{testDir}data.bin'
#     perSessionMeanImg = np.zeros((ops['Ly'],ops['Lx'],numSessions))
#     # viewer = napari.Viewer()
#     with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
#         for i in range(numSessions):
#             inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
#             frames = f.ix(indices=inds).astype(np.float32)
#             perSessionMeanImg[:,:,i] = clahe_each(frames.mean(axis=0))            
#             sessionName = ops['testFileList'][i].split('\\')[1].split('_plane_')[0]
    
#     yt = np.zeros((numSessions, numSessions))
#     xt = np.zeros((numSessions, numSessions))
#     phaseCorr = np.zeros((numSessions, numSessions))
#     imgCorr = np.zeros((numSessions, numSessions))
#     for i in range(numSessions):
#         img1 = perSessionMeanImg[:,:,i]
#         for j in range(i+1,numSessions):
#             img2 = perSessionMeanImg[:,:,j]
#             _, _, _, phaseCorr[i,j], _ = phase_corr(img1, img2)
#             imgCorr[i,j] = np.corrcoef(img1.flatten(), img2.flatten())[0,1]

#     meanPC[testn] = np.mean(phaseCorr[phaseCorr!=0])
#     meanIC[testn] = np.mean(imgCorr[imgCorr!=0])
# toc= time.time()
# timeInMin = np.round((toc-tic)/60)
# print(f'{timeInMin} minutes.')

# #%%
# f, ax = plt.subplots()
# ax.plot(meanPC, color='b')
# axr = ax.twinx()
# axr.plot(meanIC, color='y')
# #%%
# ax.set_ylabel('Phase correlation')
# ax.tick_params(axis= 'y', colors='b')
# axr.set_ylabel('Image correlation')
# axr.tick_params(axis= 'y', colors='y')
# ax.set_xlabel('Param #')


# #%%
# testn = 11
# testDir = f'{planeDir}/test{testn:02}/plane0/'
# ops = np.load(f'{testDir}ops.npy', allow_pickle=True).item()
# fpsession = ops['framesPerSession']
# numSessions = len(ops['testFileList'])
# binfn = f'{testDir}data.bin'
# perSessionMeanImg = np.zeros((ops['Ly'],ops['Lx'],numSessions))
# viewer = napari.Viewer()
# with BinaryFile(Ly = ops['Ly'], Lx = ops['Lx'], read_filename = binfn) as f:
#     for i in range(numSessions):
#         inds = np.arange(i*framesPerSession,(i+1)*framesPerSession)
#         frames = f.ix(indices=inds).astype(np.float32)
#         perSessionMeanImg[:,:,i] = frames.mean(axis=0)
        
#         sessionName = ops['testFileList'][i].split('\\')[1].split('_plane_')[0]
#         viewer.add_image(perSessionMeanImg[:,:,i], name = sessionName)
