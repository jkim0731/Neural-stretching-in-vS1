import numpy as np


def border_filter_mask(session_tuple, base_dir):
    mouse = session_tuple[0]
    plane = session_tuple[1]
    session = session_tuple[2]
    pre_final_roi_fn = base_dir / f'{mouse:03}/plane_{plane}/{session}/plane0/roi/final_roi_results_{mouse:03}_plane_{plane}_{session}_wo_dendrite_filtering.npy'
    pre_final_roi = np.load(pre_final_roi_fn, allow_pickle=True).item()
    ops_fn = base_dir / f'{mouse:03}/plane_{plane}/{session}/plane0/ops.npy'
    ops = np.load(ops_fn, allow_pickle=True).item()

    range_y, range_x = border_mask(ops)
    
    filter_mask = np.ones((ops['Ly'], ops['Lx']), dtype=bool)
    filter_mask[range_y[0]:range_y[1], range_x[0]:range_x[1]] = False

    final_mask = pre_final_roi['final_mask'].copy()
    num_pre_rois = final_mask.shape[2]
    for ri in range(num_pre_rois):
        if np.any(final_mask[:,:,ri] * filter_mask):
            final_mask[:,:,ri] = False
    
    final_mask, _ = reorder_mask(final_mask)

    return final_mask


def border_mask(ops):
    max_y = np.ceil(max(ops['yoff'].max(), 1)).astype(int)
    min_y = np.floor(min(ops['yoff'].min(), 0)).astype(int)
    max_x = np.ceil(max(ops['xoff'].max(), 1)).astype(int)
    min_x = np.floor(min(ops['xoff'].min(), 0)).astype(int)
    maxshiftNR = ops['maxregshiftNR']
    range_y = [-min_y + maxshiftNR, -max_y - maxshiftNR]
    range_x = [-min_x + maxshiftNR, -max_x - maxshiftNR]
    return range_y, range_x
    

def reorder_mask(mask_3d, startswith=1):
    # Reorder maks ids to be 1 to n
    ids = np.setdiff1d(np.unique(mask_3d), 0)
    new_mask = np.zeros((*mask_3d.shape[:2], len(ids)), dtype=np.uint16)
    for i, id in enumerate(ids):
        y, x = np.where(mask_3d == id)[:2]
        new_mask[y, x, i] = i + startswith
    new_ids = np.setdiff1d(np.unique(new_mask), 0)
    return new_mask, new_ids    