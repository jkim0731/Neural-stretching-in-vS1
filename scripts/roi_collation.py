import numpy as np
from pathlib import Path
from time import time

def get_iou_ioa_mat(mask_3d1, mask_3d2):
    # mask_3d are ordered to have 1 to n ids
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]

    iou_mat = np.zeros((mask_3d1.shape[2], mask_3d2.shape[2]))
    ioa_mat1 = np.zeros((mask_3d1.shape[2], mask_3d2.shape[2]))
    ioa_mat2 = np.zeros((mask_3d1.shape[2], mask_3d2.shape[2]))

    # going from mask1 to mask2
    # Collect all the roi_inds2 that are visited by roi_inds1
    visited_roi_inds2 = []
    for j in range(mask_3d1.shape[2]):
        temp_roi_ind1_map = np.sum(mask_3d1 == j + 1, axis=2)
        y, x = np.where(temp_roi_ind1_map > 0)
        
        temp_roi_ind2_visit = np.setdiff1d(np.unique(mask_3d2[y, x, :]), [0])
        visited_roi_inds2.extend(temp_roi_ind2_visit)
        for temp_roi_2 in temp_roi_ind2_visit:
            k = temp_roi_2 - 1
            temp_roi_ind2_map = np.sum(mask_3d2 == temp_roi_2, axis=2)
            intersection_map = temp_roi_ind1_map * temp_roi_ind2_map > 0
            union_map = temp_roi_ind1_map + temp_roi_ind2_map > 0
            iou_mat[j, k] = np.sum(intersection_map) / np.sum(union_map)
            ioa_mat1[j, k] = np.sum(intersection_map) / np.sum(temp_roi_ind1_map)
            ioa_mat2[j, k] = np.sum(intersection_map) / np.sum(temp_roi_ind2_map)
    # going from mask2 to mask1
    # Collect all the roi_inds1 that are not visited by roi_inds2 in the previous step
    left_roi_inds2 = np.setdiff1d(np.arange(1,mask_3d2.shape[2]), np.unique(visited_roi_inds2))
    for temp_roi_ind2 in left_roi_inds2:
        k = temp_roi_ind2 - 1
        temp_roi_ind2_map = np.sum(mask_3d2 == temp_roi_ind2, axis=2)
        y, x = np.where(temp_roi_ind2_map > 0)
        temp_roi_ind1_visit = np.setdiff1d(np.unique(mask_3d1[y, x, :]), [0])
        for temp_roi_ind1 in temp_roi_ind1_visit:
            j = temp_roi_ind1 - 1
            temp_roi_ind1_map = np.sum(mask_3d1 == temp_roi_ind1, axis=2)
            intersection_map = temp_roi_ind2_map * temp_roi_ind1_map > 0
            union_map = temp_roi_ind2_map + temp_roi_ind1_map > 0
            iou_mat[j, k] = np.sum(intersection_map) / np.sum(union_map)
            ioa_mat1[j, k] = np.sum(intersection_map) / np.sum(temp_roi_ind1_map)
            ioa_mat2[j, k] = np.sum(intersection_map) / np.sum(temp_roi_ind2_map)
    return iou_mat, ioa_mat1, ioa_mat2

def deal_with_multi_overlap(ioa_mat1, ioa_mat2, mask_3d1, mask_3d2, ioa_threshold=0.5):
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]
    # Only care about multi-to-one, since that was the only cases that I've seen
    # when there are multiple rois from mask2 that overlap with one roi from mask1
    multi_ind1 = np.where(np.sum(ioa_mat2 >= ioa_threshold, axis=1)>1)[0]
    # when there are multiple rois from mask1 that overlap with one roi from mask2
    multi_ind2 = np.where(np.sum(ioa_mat1 >= ioa_threshold, axis=0)>1)[0]
    # These two are assumed to be disjoint, because there was no multi-to-multi case

    area_dist1 = np.array([np.sum(mask_3d1==id) for id in range(1, mask_3d1.max()+1)])
    area_dist2 = np.array([np.sum(mask_3d2==id) for id in range(1, mask_3d2.max()+1)])
    area_dist = np.concatenate([area_dist1, area_dist2])

    ind_remove_from_mask1 = []
    ind_remove_from_mask2 = []

    for ind in multi_ind1:
        temp_overlap_roi_inds2 = np.where(ioa_mat2[ind,:] >= ioa_threshold)[0]
        # pick the one(s) with closer to median area
        percentile_to_med1 = np.abs(0.5 - np.sum(area_dist < area_dist1[ind])/len(area_dist))
        percentile_to_med2 = [np.abs(0.5 - np.sum(area_dist < area_dist2[i])/len(area_dist)) for i in temp_overlap_roi_inds2]
        if (percentile_to_med2 > percentile_to_med1).any():
            for temp_ind in temp_overlap_roi_inds2:
                ind_remove_from_mask2.append(temp_ind)
        else:
            ind_remove_from_mask1.append(ind)

    # repeat for multi_ind2
    for ind in multi_ind2:
        temp_overlap_roi_inds1 = np.where(ioa_mat1[:,ind] >= ioa_threshold)[0]
        # pick the one(s) with closer to median area
        percentile_to_med2 = np.abs(0.5 - np.sum(area_dist < area_dist2[ind])/len(area_dist))
        percentile_to_med1 = [np.abs(0.5 - np.sum(area_dist < area_dist1[i])/len(area_dist)) for i in temp_overlap_roi_inds1]
        if (percentile_to_med1 > percentile_to_med2).any():
            for temp_ind in temp_overlap_roi_inds1:
                ind_remove_from_mask1.append(temp_ind)
        else:
            ind_remove_from_mask2.append(ind)
    for ind in ind_remove_from_mask1:
        ioa_mat1[ind,:] = 0
        ioa_mat2[ind,:] = 0
    for ind in ind_remove_from_mask2:
        ioa_mat1[:,ind] = 0
        ioa_mat2[:,ind] = 0
    return ind_remove_from_mask1, ind_remove_from_mask2, ioa_mat1, ioa_mat2


def deal_with_one_to_one(ioa_mat1, ioa_mat2, mask_3d1, mask_3d2,
                        ind_remove_from_mask1, ind_remove_from_mask2,
                        ioa_threshold=0.5):
    ###
    # After multi_overlap is dealt with
    ###
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]
    # Sweeping through one direction is enough, because only one-to-one match (if any) is left
    area_dist1 = np.array([np.sum(mask_3d1==id) for id in range(1, mask_3d1.max()+1)])
    area_dist2 = np.array([np.sum(mask_3d2==id) for id in range(1, mask_3d2.max()+1)])
    area_dist = np.concatenate([area_dist1, area_dist2])
    for ind1 in range(ioa_mat1.shape[0]):
        if (ioa_mat1[ind1,:] >= ioa_threshold).any() or (ioa_mat2[ind1,:] >= ioa_threshold).any():
            match_ind2 = np.concatenate([np.where(ioa_mat1[ind1,:] >= ioa_threshold)[0],
                                         np.where(ioa_mat2[ind1,:] >= ioa_threshold)[0]])
            match_ind2 = np.unique(match_ind2)

            percentile_to_med1 = np.abs(0.5 - np.sum(area_dist < area_dist1[ind1])/len(area_dist))
            percentile_to_med2 = np.abs(0.5 - np.sum(area_dist < np.sum(area_dist2[match_ind2]))/len(area_dist))
            # summing area_dist2 in case where there are multiple overlapping rois in mask2
            # It's rare, but when there is, it shows neighboring ROIs that seem like one ROI
            if percentile_to_med1 < percentile_to_med2:
                ind_remove_from_mask2.extend(match_ind2)
            else:
                ind_remove_from_mask1.append(ind1)
    return ind_remove_from_mask1, ind_remove_from_mask2


def clean_and_merge_masks(mask_3d1, mask_3d2, ind_remove_from_mask1, ind_remove_from_mask2):
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    assert mask_3d1.max() == mask_3d1.shape[2]
    assert mask_3d2.max() == mask_3d2.shape[2]
    # just to be sure (ids1 = range(1, mask_3d1.max()+1)
    ids1 = np.setdiff1d(np.unique(mask_3d1), 0)
    ids2 = np.setdiff1d(np.unique(mask_3d2), 0)
    for ind1 in ind_remove_from_mask1:
        mask_3d1[mask_3d1==ids1[ind1]] = 0
    for ind2 in ind_remove_from_mask2:
        mask_3d2[mask_3d2==ids2[ind2]] = 0
    mask_3d1, _ = reorder_mask(mask_3d1)
    if mask_3d1.shape[2] > 0:
        mask_3d2, _ = reorder_mask(mask_3d2, startswith=mask_3d1.max()+1)
    else:  # there could be no roi in mask_3d1
        mask_3d2, _ = reorder_mask(mask_3d2, startswith=1)
    mask_3d_merged = np.dstack([mask_3d1, mask_3d2])
    return mask_3d_merged


def reorder_mask(mask_3d, startswith=1):
    # Reorder maks ids to be 1 to n
    ids = np.setdiff1d(np.unique(mask_3d), 0)
    new_mask = np.zeros((*mask_3d.shape[:2], len(ids)), dtype=np.uint16)
    for i, id in enumerate(ids):
        y, x = np.where(mask_3d == id)[:2]
        new_mask[y, x, i] = i + startswith
    new_ids = np.setdiff1d(np.unique(new_mask), 0)
    return new_mask, new_ids


def make_3d_mask(mask_2d):
    """Make 3D mask from 2D mask.
    Also reorder mask ids to be 1 to n
    """
    ids = np.setdiff1d(np.unique(mask_2d), 0)
    mask_3d = np.zeros((mask_2d.shape[0], mask_2d.shape[1], len(ids)), dtype=np.uint16)
    for i, id_ in enumerate(ids):
        y,x = np.where(mask_2d == id_)
        mask_3d[y, x, i] = i+1
    return mask_3d


def collate_masks(mask_3d1, mask_3d2, iou_threshold=0.3):
    """Collate two masks into one.
    Also reorder mask ids to be 1 to n.
    Plus, calculate unique ROIs going forward and backward (from mask_3d1 to mask_3d2 and vice versa)
    using IoU threshold
    """
    assert len(mask_3d1.shape) == len(mask_3d2.shape) == 3
    if mask_3d1.shape[2] == 0:
        if mask_3d2.shape[2] == 0:
            return np.zeros((mask_3d1.shape[0], mask_3d1.shape[1], 0), dtype=np.uint16), (0,0)
        else:
            return mask_3d2, (0, mask_3d2.shape[2])
    elif mask_3d2.shape[2] == 0:
        return mask_3d1, (mask_3d1.shape[2], 0)
    else:
        assert mask_3d1.max() == mask_3d1.shape[2]
        assert mask_3d2.max() == mask_3d2.shape[2]
    iou_mat, ioa_mat1, ioa_mat2 = get_iou_ioa_mat(mask_3d1, mask_3d2)

    num_unique_in_mask1 = len(np.where(np.sum(iou_mat >= iou_threshold, axis=1) == 0)[0])
    num_unique_in_mask2 = len(np.where(np.sum(iou_mat >= iou_threshold, axis=0) == 0)[0])

    ind_remove_from_mask1, ind_remove_from_mask2, \
        ioa_mat1, ioa_mat2 = deal_with_multi_overlap(ioa_mat1, ioa_mat2,
                                                    mask_3d1, mask_3d2)
    ind_remove_from_mask1, ind_remove_from_mask2 \
        = deal_with_one_to_one(ioa_mat1, ioa_mat2, mask_3d1, mask_3d2,
                               ind_remove_from_mask1, ind_remove_from_mask2)
    new_mask = clean_and_merge_masks(mask_3d1, mask_3d2, 
                                     ind_remove_from_mask1,
                                     ind_remove_from_mask2)
    return new_mask, (num_unique_in_mask1, num_unique_in_mask2)


def process_mask_merging(base_dir, mouse, plane, session):
    # if base_dir is str, convert to Path
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    t0 = time()
    print(f'Processing {mouse:03d}_plane_{plane}_{session}...')

    plane_dir = base_dir / f'{mouse:03}' / f'plane_{plane}' / f'{session}' / 'plane0'
    roi_dir = plane_dir / 'roi'
    if not roi_dir.exists():
        roi_dir.mkdir()
    cp_masks = np.load(plane_dir / f'{mouse:03}_plane_{plane}_{session}_cp_masks.npy', allow_pickle=True).item()
    
    num_rois_plane = []
    num_unique_rois_plane = []
    merged_masks = []
    for key in cp_masks.keys():
        mask = cp_masks[key]
        num_rois = []
        num_unique_rois = []
        for i in range(len(mask)-1):
            if i == 0:
                new_mask = make_3d_mask(mask[i])
                num_rois.append(new_mask.shape[2])
            next_mask = make_3d_mask(mask[i+1])
            new_mask, unique_roi_num = collate_masks(new_mask, next_mask)
            num_rois.append(new_mask.shape[2])
            num_unique_rois.append(unique_roi_num)
        num_rois_plane.append(num_rois)
        num_unique_rois_plane.append(num_unique_rois)
        merged_masks.append(new_mask)
    np.save(roi_dir / f'num_rois_{mouse:03d}_plane_{plane}_{session}.npy', num_rois_plane)
    np.save(roi_dir / f'num_unique_rois_{mouse:03d}_plane_{plane}_{session}.npy', num_unique_rois_plane)
    results_merged_masks = {'meanImg': merged_masks[0],
                  'meanImgE': merged_masks[1],
                  'max_proj': merged_masks[2]}
    np.save(roi_dir / f'merged_masks_{mouse:03d}_plane_{plane}_{session}.npy', results_merged_masks)

    # Create final ROI (without dendrite filtering)
    mean_and_meanE_mask, unique_roi_num_01 = collate_masks(merged_masks[0], merged_masks[1])
    final_mask, unique_roi_num_02 = collate_masks(mean_and_meanE_mask, merged_masks[2])
    final_roi_results = {'mean_and_meanE_mask': mean_and_meanE_mask,
                         'final_mask': final_mask,
                         'unique_roi_num_01': unique_roi_num_01,
                         'unique_roi_num_02': unique_roi_num_02}
    np.save(roi_dir / f'final_roi_results_{mouse:03d}_plane_{plane}_{session}_wo_dendrite_filtering.npy', final_roi_results)
    t1 = time()
    print(f'{mouse:03d}_plane_{plane}_{session} done: {(t1-t0)/60:.2f} minutes.')


def get_filtered_mask(mask, mouse, dendrite_threshold):
    if mouse < 31:
        pix_size = 0.7  # in microns
    else:
        pix_size = 0.82  # in microns

    filtering_pixel = (dendrite_threshold / pix_size / 2) ** 2 * np.pi

    mask_filtered = mask.copy()
    roi_nums = np.setdiff1d(np.unique(mask),0)
    for i in roi_nums:
        if np.sum(mask == i) < filtering_pixel:
            mask_filtered[mask == i] = 0
    return mask_filtered