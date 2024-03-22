import numpy as np
from matplotlib import pyplot as plt


def get_matched_volume_roi_inds_ordered(base_dir, mouse, top_plane, sessions, remove_notcell=True):
    """ Get the indices of the ROIs in a volume 
    that are matched across sessions and ordered by the master ROI index.
    Naming convention: p0c0000

    Args:
    base_dir: Path
        The base directory where the data is stored
    mouse: int
        The mouse number
    plane: int
        The plane number
    sessions: list
        The list of session numbers
        
    Returns:
    roi_ind_matched_ordered: list
        The list of indices of the ROIs in a volume 
        that are matched across sessions and ordered by the master ROI index.
        Naming convention: p0c0000
    """

    roi_ind_matched_all = [[] for si in range(len(sessions))]
    for plane in range(top_plane, top_plane+4):
        roi_ind_matched_ordered, _ = get_matched_roi_inds_ordered(base_dir, mouse, plane, sessions, remove_notcell=remove_notcell)
        for si in range(len(sessions)):
            roi_ind_matched_all[si].extend([f'p{plane}c{ri:04}' for ri in roi_ind_matched_ordered[si]])
    return roi_ind_matched_all


def get_matched_roi_inds_ordered(base_dir, mouse, plane, sessions, remove_notcell=True):
    """ Get the indices of the ROIs that are matched across sessions and ordered by the master ROI index.
    
    Args:
    base_dir: Path
        The base directory where the data is stored
    mouse: int
        The mouse number
    plane: int
        The plane number
    sessions: list
        The list of session numbers
        
    Returns:
    roi_ind_matched_ordered: list
        The list of indices of the ROIs that are matched across sessions and ordered by the master ROI index.
        These indices are matched to spks, stats, and iscell.npy
    matched_master_roi_inds: list
        The list of indices of the master ROIs that are matched across sessions.
    """
    plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    master_roi_fn = plane_dir / f'JK{mouse:03}_plane{plane}_cellpose_master_roi.npy'
    master_roi_results = np.load(master_roi_fn, allow_pickle=True).item()
    session_inds_master_roi = [np.where(np.array(master_roi_results['session_nums']) == sn)[0][0] for sn in sessions]

    roi_matching_fn = plane_dir / f'JK{mouse:03}_plane{plane}_cellpose_roi_session_to_master.npy'
    roi_matching = np.load(roi_matching_fn, allow_pickle=True).item()

    # Gather master ROI indices from each session
    session_to_master = []
    for i, si in enumerate(session_inds_master_roi):
        if remove_notcell:
            session = sessions[i]
            roi_dir = plane_dir / f'{session:03}/plane0/roi'
            iscell = np.load(roi_dir / 'iscell.npy')
            iscell_inds = np.where(iscell[:,0])[0]
            iscell_inds_viable_cell_index = np.where(np.isin(master_roi_results['viable_cell_index_list'][si],
                                                            iscell_inds))[0]
        else:
            iscell_inds_viable_cell_index = range(len(master_roi_results['viable_cell_index_list'][si]))
        session_to_master.append(roi_matching['matching_master_roi_index_list'][si][iscell_inds_viable_cell_index])

    # Find matched master ROIs across the sessions
    matched_master_roi_inds = session_to_master[0]
    for si in range(1,len(sessions)):
        matched_master_roi_inds = np.intersect1d(matched_master_roi_inds, session_to_master[si])

    # Find session ROI index to each matched master ROIs, ordered in the same order
    roi_ind_matched_ordered = []
    if len(matched_master_roi_inds) > 0:        
        viable_cell_index = [master_roi_results['viable_cell_index_list'][si] for si in session_inds_master_roi]
        session_to_master_again = [roi_matching['matching_master_roi_index_list'][si] for si in session_inds_master_roi]
        for si in range(len(sessions)):
            viable_ind_matched_ordered = np.array([np.where(session_to_master_again[si]==ri)[0][0] for ri in matched_master_roi_inds])
            session_roi_ind_matched_ordered = np.array([viable_cell_index[si][i] for i in viable_ind_matched_ordered])
            roi_ind_matched_ordered.append(session_roi_ind_matched_ordered)
            assert len(session_roi_ind_matched_ordered) == len(matched_master_roi_inds)
    
    if remove_notcell:
        # Check again that all orders are iscell
        for i in range(len(sessions)):
            session = sessions[i]
            roi_dir = plane_dir / f'{session:03}/plane0/roi'
            iscell = np.load(roi_dir / 'iscell.npy')
            iscell_inds = np.where(iscell[:,0])[0]
            assert np.isin(roi_ind_matched_ordered[i], iscell_inds).all()
        
    return roi_ind_matched_ordered, matched_master_roi_inds


def compare_group_matched_inds(base_dir, mouse, plane, sessions):
    """ Compare the indices of the ROIs that are matched across sessions.
    
    """
    plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    master_roi_fn = plane_dir / f'JK{mouse:03}_plane{plane}_cellpose_master_roi.npy'
    master_roi_results = np.load(master_roi_fn, allow_pickle=True).item()
    session_inds_master_roi = [np.where(np.array(master_roi_results['session_nums']) == sn)[0][0] for sn in sessions]

    roi_matching_fn = plane_dir / f'JK{mouse:03}_plane{plane}_cellpose_roi_session_to_master.npy'
    roi_matching = np.load(roi_matching_fn, allow_pickle=True).item()

    session_to_master = []
    for si in session_inds_master_roi:
        session_to_master.append(roi_matching['matching_master_roi_index_list'][si])
    matched_master_roi_inds = session_to_master[0]
    for si in range(1,len(sessions)):
        matched_master_roi_inds = np.intersect1d(matched_master_roi_inds, session_to_master[si])

    session_maps = [master_roi_results['session_map_list'][si] for si in session_inds_master_roi]
    matched_session_maps = []
    for i in range(len(session_inds_master_roi)):
        session_roi_inds_matched = np.where(np.isin(session_to_master[i], matched_master_roi_inds))[0]
        matched_session_maps.append(session_maps[i][session_roi_inds_matched,:,:])

    fig, ax = plt.subplots(len(sessions),1, figsize=(15,len(sessions)*5))
    for si in range(len(sessions)):
        ax[si].imshow(matched_session_maps[si].sum(axis=0))
        ax[si].imshow(matched_session_maps[si].sum(axis=0))
    return fig, ax


def compared_ordered_roi_matching_napari(base_dir, mouse, plane, sessions):
    """ Show matched ROI maps via napari (before transformation)

    """
    import napari
    roi_ind_matched_ordered, matched_master_roi_inds = get_matched_roi_inds_ordered(base_dir, mouse, plane, sessions)
    plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    stats = []
    for session in sessions:
        roi_dir = plane_dir / f'{session:03}/plane0/roi'
        stat = np.load(roi_dir / 'stat_refined.npy', allow_pickle=True)
        stats.append(stat)
    ops = np.load(roi_dir / 'ops.npy', allow_pickle=True).item()

    matched_roi_maps = np.zeros((len(sessions), len(matched_master_roi_inds), ops['Ly'], ops['Lx']))

    for si in range(len(sessions)):
        for ri in range(len(matched_master_roi_inds)):
            x = stats[si][roi_ind_matched_ordered[si][ri]]['xpix']
            y = stats[si][roi_ind_matched_ordered[si][ri]]['ypix']
            matched_roi_maps[si,ri,y,x] = 1

    viewer = napari.Viewer()
    for si in range(len(sessions)):
        if si == 0:
            viewer.add_image(matched_roi_maps[si], name=f'session_{sessions[si]}')
        else:
            viewer.add_image(matched_roi_maps[si], name=f'session_{sessions[si]}', opacity=0.5)
