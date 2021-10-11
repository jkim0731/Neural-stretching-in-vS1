# -*- coding: utf-8 -*-
"""
Results partly curated in Session selection JK027.pptx

There are some issues with session selection (related to session_selection.py)
Some sessions with the same headplate setting (spontaneous before and after training)
seem quite different from each other.
Why?
"""

import numpy as np
import matplotlib.pyplot as plt
import napari
from suite2p.io.binary import BinaryFile

def phase_corr(fixed, moving, transLim = 0):
    # apply np.roll(moving, (ymax, xmax), axis=(0,1)) to match moving to fixed
    # or, np.roll(fixed, (-ymax, -xmax), axis=(0,1))
    if fixed.shape != moving.shape:
        raise('Dimensions must match')
    R = np.fft.fft2(fixed) * np.fft.fft2(moving).conj()
    R /= np.absolute(R)
    r = np.absolute(np.fft.ifft2(R))
    if transLim > 0:
        r = np.block([[r[-transLim:,-transLim:], r[-transLim:, :transLim+1]],
                     [r[:transLim+1,-transLim:], r[:transLim+1, :transLim+1]]]
            )
        ymax, xmax = np.unravel_index(np.argmax(r), (transLim*2 + 1, transLim*2 + 1))
        ymax, xmax = ymax - transLim, xmax - transLim
        cmax = np.amax(r)
        center = r[transLim, transLim]
    else:
        ymax, xmax = np.unravel_index(np.argmax(r), r.shape)
        cmax = np.amax(r)
        center = r[0, 0]
    return ymax, xmax, cmax, center, r

#%% Test within training session
# Is the difference because of hydration level?
# If so, first 1/4 and last 1/4 of the regular training imaging session might look different,
# whereas the first 1/4 look similar to the before-training spontaneous session, and
# the last 1/4 look similar to the after-training spontaneous session.

baseDir = 'H:/'
mouse = '052'
pn = 4
session = '006'
before = '5555_002'
after = '5555_012'

#%%
division = 4
sessionOps = np.load(f'{baseDir}{mouse}/plane_{pn}/{session}/plane0/ops.npy', allow_pickle=True).item()
Ly = sessionOps['Ly']
Lx = sessionOps['Lx']
nframes = sessionOps['nframes']
sessionImgsDiv = []
with BinaryFile(Ly, Lx, read_filename=f'{baseDir}{mouse}/plane_{pn}/{session}/plane0/data.bin') as f:
    data = f.data
    for i in range(division):
        tempStartFrame = (i*nframes) // 4
        tempEndFrame = ((i+1)*nframes) // 4
        sessionImgsDiv.append(data[tempStartFrame:tempEndFrame,:,:].mean(axis=0))
#%%
beforeOps = np.load(f'{baseDir}{mouse}/plane_{pn}/{before}/plane0/ops.npy', allow_pickle=True).item()
beforeImg = beforeOps['meanImg']
ymax, xmax,_,_,_ = phase_corr(sessionImgsDiv[0], beforeImg)
beforeReg = np.roll(beforeImg, (ymax, xmax), axis=(0,1))

afterOps = np.load(f'{baseDir}{mouse}/plane_{pn}/{after}/plane0/ops.npy', allow_pickle=True).item()
afterImg = afterOps['meanImg']
ymax, xmax,_,_,_ = phase_corr(sessionImgsDiv[-1], afterImg)
afterReg = np.roll(afterImg, (ymax, xmax), axis=(0,1))

mov = [beforeReg, *sessionImgsDiv, afterReg]

napari.view_image(np.array(mov))

#%% Compare changes across single session
pn = 4
session = '025'

division = 4
sessionOps = np.load(f'{baseDir}{mouse}/plane_{pn}/{session}/plane0/ops.npy', allow_pickle=True).item()
Ly = sessionOps['Ly']
Lx = sessionOps['Lx']
nframes = sessionOps['nframes']
sessionImgsDiv = []
with BinaryFile(Ly, Lx, read_filename=f'{baseDir}{mouse}/plane_{pn}/{session}/plane0/data.bin') as f:
    data = f.data
    for i in range(division):
        tempStartFrame = (i*nframes) // 4
        tempEndFrame = ((i+1)*nframes) // 4
        sessionImgsDiv.append(data[tempStartFrame:tempEndFrame,:,:].mean(axis=0))

napari.view_image(np.array(sessionImgsDiv), name = f'JK{mouse} session {session} plane {pn}')


