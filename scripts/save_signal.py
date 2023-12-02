import numpy as np
import pandas as pd
from suite2p.gui import drawroi

def save_signal(mouse, plane, session, base_dir):    
    """Extract signal and save the results
    To remove artifacts from laser blocking,
    remove the first and last frame of each trial.

    Prerequisites:
    - final_mask.npy
    - _frame_time.pkl

    Args:
        mouse (int): mouse number
        plane (int): plane number
        session (str): session name
        base_dir (Path): path to the base directory
    """
    plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    session_dir = plane_dir / f'{session}/plane0'
    roi_dir = session_dir / 'roi'
    final_mask_fn = roi_dir / 'final_mask.npy'
    final_mask = np.load(final_mask_fn)
    ops_fn = session_dir / 'ops.npy'
    ops = np.load(ops_fn, allow_pickle=True).item()
    ops['reg_file'] = session_dir / 'data.bin'

    # Get signals from the mask
    numROI = final_mask.shape[2]
    statBlank = []
    stat0 = []
    for ci in range(numROI):
        roi_mask = final_mask[:,:,ci]
        (ypix, xpix) = np.unravel_index(np.where(roi_mask.flatten())[0], roi_mask.shape)
        lam = np.ones(ypix.shape)/len(ypix)
        med = (np.median(ypix), np.median(xpix))
        stat0.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size, 'med': med})
    F, Fneu, _, _, spks, ops, stat = drawroi.masks_and_traces(ops, stat0, statBlank)
    iscell = np.ones((numROI,2), 'uint8')

    # Get frame time to remove first and last frame of each trial
    # to remove negative ticks from laser blocking
    frame_time_fn = plane_dir / f'{mouse:03}_{session}_plane_{plane}_frame_time.pkl'
    frame_time = pd.read_pickle(frame_time_fn)
    reduced_frame_time = frame_time.groupby('trialNum').apply(lambda x: x.iloc[1:-1]).reset_index(drop=True)
    reduced_frame_indice = reduced_frame_time.frame_index.values.astype(int)

    F_reduced = F[:,reduced_frame_indice]
    Fneu_reduced = Fneu[:,reduced_frame_indice]
    spks_reduced = spks[:,reduced_frame_indice]

    # Save the results
    F_fn = roi_dir / 'F_reduced.npy'
    Fneu_fn = roi_dir / 'Fneu_reduced.npy'
    spks_fn = roi_dir / 'spks_reduced.npy'
    reduced_frame_time_fn = roi_dir / 'reduced_frame_time.pkl'
    ops_fn = roi_dir / 'ops.npy'
    stat_fn = roi_dir / 'stat.npy'
    iscell_fn = roi_dir / 'iscell.npy'
    np.save(F_fn, F_reduced)
    np.save(Fneu_fn, Fneu_reduced)
    np.save(spks_fn, spks_reduced)
    reduced_frame_time.to_pickle(reduced_frame_time_fn)
    np.save(ops_fn, ops)
    np.save(stat_fn, stat)
    np.save(iscell_fn, iscell)