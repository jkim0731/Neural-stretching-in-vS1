# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 19:58:43 2021

@author: shires
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py

sessions = ['003', '5555_010', '5555_110']
# sessions = ['003', '5554_010', '5554_110']
for si, sn in enumerate(sessions):
    fig, ax = plt.subplots(2,2)
    for pi, pn in enumerate(range(1,5)):
    # for pi, pn in enumerate(range(5,9)):
        ops = np.load(f'H:/036/plane_{pn}/{sn}/plane0/ops.npy', allow_pickle=True).item()
        yi = pi//2
        xi = pi%2
        ax[yi,xi].imshow(ops['meanImg'], cmap='gray')
        ax[yi,xi].set_title(f'Plane {pn}')
    fig.suptitle(f'Session {sn}')
    fig.tight_layout()

'''
Planes 1 and 3 are still swapped! 
Or was it not swapped before?
And so are planes 5 and 7.
'''
#%% Look at h5 file
sessions = ['003_000', '5554_010', '5554_110']
for si, sn in enumerate(sessions):
    fig, ax = plt.subplots(2,2)
    for pi, pn in enumerate(range(5,9)):
        fn = f'H:/036/plane_{pn}/036_{sn}_plane_{pn}.h5'
        with h5py.File(fn, 'r') as f:
            mimg = np.mean(np.array(f['data']),axis=0)
        yi = pi//2
        xi = pi%2
        ax[yi,xi].imshow(mimg, cmap='gray')
        ax[yi,xi].set_title(f'Plane {pn}')
    fig.suptitle(f'Session {sn}')
    fig.tight_layout()
'''
Confirmed
'''
#%%
pn= 5
ops=np.load(f'H:/036/plane_{pn}/{sn}/plane0/ops.npy', allow_pickle=True).item()

'''
Based on the ops, it seems JK036 did NOT have any issue in the first place.
Wrongfully swapped.
And now it's just swapped back.
'''

#%%
pn= 3
ops=np.load(f'H:/039/plane_{pn}/{sn}/plane0/ops.npy', allow_pickle=True).item()