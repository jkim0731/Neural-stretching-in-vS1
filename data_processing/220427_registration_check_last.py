# -*- coding: utf-8 -*-
"""
The last visual check on the session-to-session registration.
Best methods were found through 220302_another_registration_test.py
Before moving on to roi collection (should be an update version of 220107_roi_collection.py),
    finalize the methods by looking through each one with the best method.
    
2022/04/27 JK
"""

import numpy as np
from matplotlib import pyplot as plt
from suite2p.registration import rigid, nonrigid, utils, register

from pystackreg import StackReg
import os, glob
import napari
from suite2p.io.binary import BinaryFile
from skimage import exposure
import gc
gc.enable()

# h5Dir = 'D:/TPM/JK/h5/'
h5Dir = 'D:/'

mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
refSessions =   [4,     3,    3,    1,      7,      2,      1,      3,      3,      3,      3,      3]


def twostep_register(img, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                     rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2):
    frames = img.copy().astype(np.float32)
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

def s2p_2step_nr(mimgList, refMimg, op):
    op['block_size'] = [op['block_size_list'][0], op['block_size_list'][0]]
    mimgList1step, roff1, nroff1 = s2p_nonrigid_registration(mimgList, refMimg, op)
    op['block_size'] = [op['block_size_list'][1], op['block_size_list'][1]]
    mimgList2step, roff2, nroff2 = s2p_nonrigid_registration(mimgList1step, refMimg, op)
    return mimgList2step, roff1, roff2, nroff1, nroff2

def s2p_nonrigid_registration(mimgList, refImg, op):
    ### ------------- compute registration masks ----------------- ###
    Ly, Lx = refImg.shape
    maskMul, maskOffset = rigid.compute_masks(
        refImg=refImg,
        maskSlope= 3 * op['smooth_sigma'],
    )
    cfRefImg = rigid.phasecorr_reference(
        refImg=refImg,
        smooth_sigma=op['smooth_sigma'],
        # pad_fft=ops['pad_fft'], # False by default
    )

    yblock, xblock, nblocks, block_size, NRsm = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=op['block_size'])
    
    maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(
        refImg0=refImg,
        maskSlope=3 * op['smooth_sigma'], # slope of taper mask at the edges
        smooth_sigma=op['smooth_sigma'],
        yblock=yblock,
        xblock=xblock,
        # pad_fft=ops['pad_fft'], # False by default
    )

    ### ------------- register binary to reference image ------------ ###
    
    # mean_img = np.zeros((Ly, Lx))
    rigid_offsets, nonrigid_offsets = [], []
    frames = np.array(mimgList).astype(np.float32)
    fsmooth = frames.copy().astype(np.float32)

    # rigid registration
    ymax, xmax, cmax = rigid.phasecorr(
        data=rigid.apply_masks(data=fsmooth, maskMul=maskMul, maskOffset=maskOffset),
        cfRefImg=cfRefImg,
        maxregshift=op['maxregshift'],
        smooth_sigma_time=op['smooth_sigma_time'],
    )
    rigid_offsets.append([ymax, xmax, cmax])

    for frame, dy, dx in zip(frames, ymax, xmax):
        frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)

    # non-rigid registration
    # need to also shift smoothed data (if smoothing used)
    fsmooth = frames.copy()
        
    ymax1, xmax1, cmax1 = nonrigid.phasecorr(
        data=fsmooth,
        maskMul=maskMulNR.squeeze(),
        maskOffset=maskOffsetNR.squeeze(),
        cfRefImg=cfRefImgNR.squeeze(),
        snr_thresh=op['snr_thresh'],
        NRsm=NRsm,
        xblock=xblock,
        yblock=yblock,
        maxregshiftNR=op['block_size'][0]//10,
    )

    frames2 = nonrigid.transform_data(
        data=frames,
        nblocks=nblocks,
        xblock=xblock,
        yblock=yblock,
        ymax1=ymax1,
        xmax1=xmax1,
    )

    nonrigid_offsets.append([ymax1, xmax1, cmax1])
    
    return frames2, rigid_offsets, nonrigid_offsets

#%% Manual dictionary for registration methods
reg_method = {'JK025':
                  {'plane 1': 'bilinear',
                   'plane 2': 'bilinear',
                   'plane 3': 'suite2p-rolling',
                   'plane 4': 'bilinear',
                   'plane 5': 'suite2p-reference',
                   'plane 6': 'suite2p-reference',
                   'plane 7': 'suite2p-reference',
                   'plane 8': 'suite2p-reference'
                   },
              'JK027':
                  {'plane 1': 'bilinear',
                   'plane 2': 'bilinear',
                   'plane 3': 'bilinear',
                   'plane 4': 'bilinear',
                   'plane 5': 'suite2p-rolling',
                   'plane 6': 'suite2p-rolling',
                   'plane 7': '◘',
                   'plane 8': 'suite2p-reference'
                   }
              }
