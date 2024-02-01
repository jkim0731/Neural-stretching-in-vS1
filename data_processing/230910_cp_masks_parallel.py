import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing
from itertools import starmap
from cellpose import models
from suite2p import extraction
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

BASE_DIR_1 = Path(r'F:')
BASE_DIR_2 = Path(r'D:')
# BASE_DIR_1 = Path(r'E:\TPM\JK\h5')
# BASE_DIR_2 = Path(r'E:\TPM\JK\h5')

def generate_multi_cp_masks(mouse, plane, session):
    base_dir = BASE_DIR_1 if mouse < 31 else BASE_DIR_2
    diameter_range = np.arange(10,17) if mouse < 31 else np.arange(8,16)
    ops = np.load(base_dir /f'{mouse:03}/plane_{plane}/{session}/plane0/ops.npy', allow_pickle=True).item()
    mean_mask_list=[]
    meanE_mask_list=[]
    max_mask_list=[]
    for diameter in diameter_range:
        mean_mask, meanE_mask, max_mask = get_cp_masks(ops, images=None, diameter=diameter)
        for image_type in ['mean', 'meanE', 'max']:
            dendrite_threshold = 6
            if image_type == 'meanE':
                temp_mask = meanE_mask
                filtered_mask = get_filtered_mask(temp_mask, mouse, dendrite_threshold)
                meanE_mask_list.append(filtered_mask)
            elif image_type == 'mean':
                temp_mask = mean_mask
                filtered_mask = get_filtered_mask(temp_mask, mouse, dendrite_threshold)
                mean_mask_list.append(filtered_mask)
            else:
                temp_mask = max_mask
                filtered_mask = get_filtered_mask(temp_mask, mouse, dendrite_threshold)
                max_mask_list.append(filtered_mask)
    # save the results
    results = {'mean_mask_list': mean_mask_list,
               'meanE_mask_list': meanE_mask_list,
               'max_mask_list': max_mask_list}
    save_dir = base_dir / f'{mouse:03}/plane_{plane}/{session}/plane0/'
    np.save(save_dir / f'{mouse:03}_plane_{plane}_{session}_cp_masks.npy', results)


def get_cp_masks(ops, images=None, model_type='cyto', diameter=13, channels=[0,0]):
    model = models.CellposeModel(gpu=True, model_type=model_type)
    
    if images is None:
        mean_img = ops['meanImg']
        if 'meanImgE' not in ops.keys():
            ops = extraction.enhanced_mean_image(ops)
        
        meanE_img = ops['meanImgE']
        max_img = ops['max_proj']
        
        mean_mask = model.eval(mean_img, channels=channels, diameter=diameter)[0]
        mean_mask[mean_mask>0] += 10
        meanE_mask = model.eval(meanE_img, channels=channels, diameter=diameter)[0]
        meanE_mask[meanE_mask>0] += np.max(mean_mask) + 10
        max_mask = model.eval(max_img, channels=channels, diameter=diameter)[0]
        max_mask[max_mask>0] += np.max(meanE_mask) + 10

        if max_img.shape != mean_img.shape:
            temp_max_max = np.zeros_like(mean_img)
            yrange = ops['yrange']
            xrange = ops['xrange']
            temp_max_max[yrange[0]:yrange[1], xrange[0]:xrange[1]] = max_mask
            max_mask = temp_max_max

        return mean_mask, meanE_mask, max_mask
    else:
        if images == 'mean':
            img = ops['meanImg']
        elif images == 'meanE':
            img = ops['meanImgE']
        elif images == 'max':
            img = ops['max_proj']
        mask = model.eval(img, channels=channels, diameter=diameter)[0]
        mask[mask>0] += 10
        return mask

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

if __name__ == "__main__":
    mice = [25,27,30,36,39,52]
    planes = range(1,9)

    # Generate tuples of sessions
    # session_tuples = []
    # for mouse in mice:
    #     base_dir = BASE_DIR_1 if mouse < 31 else BASE_DIR_2
    #     for plane in planes:
    #         plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    #         sessions = [x.name for x in plane_dir.iterdir() if x.is_dir() and x.name[0].isdigit()]
    #         # list of (mouse,plane,session) tuples
    #         test_sessions = [(mouse, plane, session) for session in sessions]
    #         session_tuples.extend(test_sessions)
    
    # session_tuples = []
    # for mouse in mice:
    #     base_dir = BASE_DIR_1 if mouse < 31 else BASE_DIR_2
    #     for plane in planes:
    #         plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    #         sessions = [x.name for x in plane_dir.iterdir() if x.is_dir() and x.name[0].isdigit()]
    #         # list of (mouse,plane,session) tuples
    #         for session in sessions:
    #             session_dir = plane_dir / session / 'plane0'
    #             mask_fn = f'{mouse:03}_plane_{plane}_{session}_cp_masks.npy'
    #             if not (session_dir / mask_fn).exists():
    #                 session_tuples.append((mouse,plane,session))

    mouse = 27
    session = '004'
    session_tuples = [(mouse, plane, session) for plane in range(1,9)]

    # get the number of available cores for multiprocessing
    num_cores = multiprocessing.cpu_count()-2

    with multiprocessing.Pool(num_cores) as pool:
        pool.starmap(generate_multi_cp_masks, session_tuples)
