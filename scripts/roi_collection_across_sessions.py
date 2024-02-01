import numpy as np
import pandas as pd
from pystackreg import StackReg
from skimage import exposure
from suite2p.registration import rigid, nonrigid
from functools import reduce
import os
from pathlib import Path
import itertools
from multiprocessing import Pool, cpu_count

# Helper functions

# Functions for registration
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


def fix_reg_collection(reg_result):
    '''
    Fixing registration result collection of the 'new' suite2p method.
    2022/07/13 JK
    '''
    num_sessions = len(reg_result['selected_session_num'])
    roff1_y = [0]
    roff1_x = [0]
    roff1_c = [0.1]
    roff1 = reg_result['suite2p_result']['roff1']
    for i in range(num_sessions-1):
        roff1_y.append([x[0] for x in roff1[i]][0])
        roff1_x.append([x[0] for x in roff1[i]][1])
        roff1_c.append([x[0] for x in roff1[i]][2])
    roff1 = [[np.array(roff1_y), np.array(roff1_x), np.array(roff1_c)]]
    reg_result['suite2p_result']['roff1'] = roff1

    roff2_y = [0]
    roff2_x = [0]
    roff2_c = [0.1]
    roff2 = reg_result['suite2p_result']['roff2']
    for i in range(num_sessions-1):
        roff2_y.append([x[0] for x in roff2[i]][0])
        roff2_x.append([x[0] for x in roff2[i]][1])
        roff2_c.append([x[0] for x in roff2[i]][2])
    roff2 = [[np.array(roff2_y), np.array(roff2_x), np.array(roff2_c)]]
    reg_result['suite2p_result']['roff2'] = roff2

    offset_len = len(reg_result['suite2p_result']['nroff1'][0][0][0])
    nroff1_y = [np.zeros(offset_len)]
    nroff1_x = [np.zeros(offset_len)]
    nroff1_c = [np.ones(offset_len)/10]
    nroff1 = reg_result['suite2p_result']['nroff1']
    for i in range(num_sessions-1):
        nroff1_y.append([x[0] for x in nroff1[i]][0])
        nroff1_x.append([x[0] for x in nroff1[i]][1])
        nroff1_c.append([x[0] for x in nroff1[i]][2])
    nroff1 = [[np.array(nroff1_y).astype(np.float32), np.array(nroff1_x).astype(np.float32), np.array(nroff1_c).astype(np.float32)]]
    reg_result['suite2p_result']['nroff1'] = nroff1
    
    offset_len = len(reg_result['suite2p_result']['nroff2'][0][0][0])
    nroff2_y = [np.zeros(offset_len)]
    nroff2_x = [np.zeros(offset_len)]
    nroff2_c = [np.ones(offset_len)/10]
    nroff2 = reg_result['suite2p_result']['nroff2']
    for i in range(num_sessions-1):
        nroff2_y.append([x[0] for x in nroff2[i]][0])
        nroff2_x.append([x[0] for x in nroff2[i]][1])
        nroff2_c.append([x[0] for x in nroff2[i]][2])
    nroff2 = [[np.array(nroff2_y).astype(np.float32), np.array(nroff2_x).astype(np.float32), np.array(nroff2_c).astype(np.float32)]]
    reg_result['suite2p_result']['nroff2'] = nroff2

    return reg_result


# Functions for ROI collection
def calculate_regCell_threshold(cellMap, numPix, thresholdResolution = 0.01):
    trPrecision = len(str(thresholdResolution).split('.')[1])
    thresholdRange = np.around(np.arange(0.3,1+thresholdResolution/10,thresholdResolution), trPrecision)
    threshold = thresholdRange[np.argmin([np.abs(numPix - np.sum(cellMap>=threshold)) for threshold in thresholdRange])]
    cutMap = (cellMap >= threshold).astype(bool)
    return cutMap, threshold


def perimeter_area_ratio(img: bool):
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    if len(img.shape) != 3:
        raise('Input image dimension should be either 2 or 3.')
    img = img.astype('bool')
    numCell = img.shape[0]
    par = np.zeros(numCell)
    for ci in range(numCell):
        tempImg = img[ci,:,:]
        inside = tempImg*np.roll(tempImg,1,axis=0)*np.roll(tempImg,-1,axis=0)*np.roll(tempImg,1,axis=1)*np.roll(tempImg,-1,axis=1)
        perimeter = np.logical_xor(tempImg, inside)
        par[ci] = np.sum(perimeter)/np.sum(tempImg) # tempImg instead of inside just to prevent dividing by 0 for some scattered rois
    return par


def check_multi_match_pair(multiMatchMasterInd, multiMatchNewInd, masterPar, newPar, 
                           overlapMatrix, overlaps, unions, delFromMasterInd, delFromNewInd):
    tempDfMasterInd = np.zeros(0,'int')
    tempDfNewInd = np.zeros(0,'int')
    remFromMMmaster = np.zeros(0,'int') # Collect multiMatchMasterInd that is already processed (in the for loop)
    remFromMMnew = np.zeros(0,'int') # Collect multiMatchNewInd that is already processed (in the for loop)
    # Remove remFromMM* at the end.
    
    # First, deal with delFromMasterInd
    if len(multiMatchMasterInd)>0:
        for mci in range(len(multiMatchMasterInd)):
            masterCi = multiMatchMasterInd[mci]
            if masterCi in remFromMMmaster: # For the updates from the loop.
                continue
            else:
                # cell index from the new map, that matches with this master ROI.
                newCis = np.where(overlapMatrix[masterCi,:])[0] # This should be longer than len == 1, by definition of multiMatchMasterInd.
                
                # Calculate any other master ROIs that overlap with this new ROI(s).
                masterBestMatchi = np.zeros(len(newCis), 'int')
                for i, nci in enumerate(newCis):
                    masterBestMatchi[i] = np.argmax(overlaps[:,nci]).astype(int)
                # Check if the best-matched master ROIs for these multiple new ROIs are this master ROI.
                # (I.e., when a large combined master ROI covers more than one new ROIs. 
                # See 220117 ROI collection across sessions with matching depth.pptx)
                # In this case, just remove this master ROI
                if any([len(np.where(masterBestMatchi==mi)[0])>1 for mi in masterBestMatchi]):
                    tempDfMasterInd = np.hstack((tempDfMasterInd, [masterCi]))
                    remFromMMmaster = np.hstack((remFromMMmaster, [masterCi]))
                
                # Else, check if there is a matching pair (or multiples)
                # I.e., multiple overlapping new ROIs with multiple master ROIs (usually couples in both ROI maps)
                # In this case, remove the pair (or multiples) with higher mean PAR 
                else:
                    newBestMatchi = np.zeros(len(masterBestMatchi), 'int')
                    for i, mci in enumerate(masterBestMatchi):
                        newBestMatchi[i] = np.argmax(overlaps[mci,:]).astype(int)

                    if all(newCis == newBestMatchi): # found a matching pair (or a multiple)
                        # Calculate mean perimeter/area ratio
                        masterMeanpar = np.mean(masterPar[masterBestMatchi])
                        newMeanpar = np.mean(newPar[newBestMatchi])
                        
                        # Remove the pair with higher mean par
                        if masterMeanpar <= newMeanpar:
                            tempDfNewInd = np.hstack((tempDfNewInd, newBestMatchi))
                        else:
                            tempDfMasterInd = np.hstack((tempDfMasterInd, masterBestMatchi))
    
                        # Collect indices already processed
                        remFromMMmaster = np.hstack((remFromMMmaster, masterBestMatchi))
                        remFromMMnew = np.hstack((remFromMMnew, newBestMatchi))
                              
    # Then, deal with delFromNewInd
    if len(multiMatchNewInd)>0:
        for nci in range(len(multiMatchNewInd)):
            newCi = multiMatchNewInd[nci]
            if newCi in remFromMMnew:
                continue
            else:
                masterCis = np.where(overlapMatrix[:,newCi])[0]
                    
                newBestMatchi = np.zeros(len(masterCis), 'int')
                for i, mci in enumerate(masterCis):
                    newBestMatchi[i] = np.argmax(overlaps[mci,:]).astype(int)
                    
                # Check if there are multiple same matched IDs 
                # In this case, just remove the new ROI
                if any([len(np.where(newBestMatchi==ni)[0])>1 for ni in newBestMatchi]):
                    tempDfNewInd = np.hstack((tempDfNewInd, [newCi]))
                    remFromMMnew = np.hstack((remFromMMnew, [newCi]))
                
                # Else, check if there is a matching pair (or multiples)
                else:
                    masterBestMatchi = np.zeros(len(newBestMatchi), 'int')
                    for i, nci in enumerate(newBestMatchi):
                        masterBestMatchi[i] = np.argmax(overlaps[:,nci]).astype(int)

                    if all(masterCis == masterBestMatchi): # found a matching pair
                        # Calculate mean perimeter/area ratio
                        masterMeanpar = np.mean(masterPar[masterBestMatchi])
                        newMeanpar = np.mean(newPar[newBestMatchi])
                        
                        # Remove the pair with higher mean par
                        if masterMeanpar <= newMeanpar:
                            tempDfNewInd = np.hstack((tempDfNewInd, newBestMatchi))
                        else:
                            tempDfMasterInd = np.hstack((tempDfMasterInd, masterBestMatchi))
    
                        # Collect indices already processed
                        remFromMMmaster = np.hstack((remFromMMmaster, masterBestMatchi))
                        remFromMMnew = np.hstack((remFromMMnew, newBestMatchi))
    
    # Remove collected indices
    if len(tempDfMasterInd)>0:
        delFromMasterInd.extend(tempDfMasterInd)
    if len(tempDfNewInd)>0:
        delFromNewInd.extend(tempDfNewInd)
    
    # Ideally, these multi indices should be empty.    
    multiMatchMasterInd = np.array([mi for mi in multiMatchMasterInd if mi not in remFromMMmaster])
    multiMatchNewInd = np.array([ni for ni in multiMatchNewInd if ni not in remFromMMnew])

    # If there are still multi matches left, match them using intesection/union
    intersect_over_union = overlaps/unions
    if len(multiMatchMasterInd) > 0:
        for mci in range(len(multiMatchMasterInd)):
            masterCi = multiMatchMasterInd[mci]
            if masterCi in remFromMMmaster: # For the updates from the loop.
                continue
            else:
                # cell index from the new map, that matches with this master ROI.
                newCis = np.where(overlapMatrix[masterCi,:])[0] # This should be longer than len == 1, by definition of multiMatchMasterInd.
                newBestMatchi = newCis[np.argmax(intersect_over_union[masterCi,newCis])]
                # Remove the cell with higher mean par
                if masterPar[masterCi] <= newPar[newBestMatchi]:
                    tempDfNewInd = np.hstack((tempDfNewInd, newBestMatchi))
                    remFromMMnew = np.hstack((remFromMMnew, newBestMatchi))
                else:
                    tempDfMasterInd = np.hstack((tempDfMasterInd, masterCi))
                remFromMMmaster = np.hstack((remFromMMmaster, masterCi))
    if len(multiMatchNewInd) > 0:
        for nci in range(len(multiMatchNewInd)):
            newCi = multiMatchNewInd[nci]
            if newCi in remFromMMnew: # For the updates from the loop.
                continue
            else:
                # cell index from the new map, that matches with this master ROI.
                masterCis = np.where(overlapMatrix[:,newCi])[0] # This should be longer than len == 1, by definition of multiMatchMasterInd.
                masterBestMatchi = masterCis[np.argmax(intersect_over_union[masterCis,newCi])]
                # Remove the cell with higher mean par
                if newPar[newCi] <= masterPar[masterBestMatchi]:
                    tempDfMasterInd = np.hstack((tempDfMasterInd, masterBestMatchi))
                    remFromMMmaster = np.hstack((remFromMMmaster, masterBestMatchi))
                else:
                    tempDfNewInd = np.hstack((tempDfNewInd, newCi))
                remFromMMnew = np.hstack((remFromMMnew, newCi))
    
    # Remove collected indices
    if len(tempDfMasterInd)>0:
        delFromMasterInd.extend(tempDfMasterInd)
    if len(tempDfNewInd)>0:
        delFromNewInd.extend(tempDfNewInd)
        
    # Ideally, these multi indices should be empty.    
    multiMatchMasterInd = np.array([mi for mi in multiMatchMasterInd if mi not in remFromMMmaster])
    multiMatchNewInd = np.array([ni for ni in multiMatchNewInd if ni not in remFromMMnew])


    return delFromMasterInd, delFromNewInd, multiMatchMasterInd, multiMatchNewInd


def run_roi_collection(mouse, pn, session_nums, h5_dir, roi_overlap_threshold=0.5, rerun=True):
    if isinstance(h5_dir, Path):
        h5_dir = str(h5_dir) + '/'
    planeDir = f'{h5_dir}{mouse:03}/plane_{pn}/'
    master_roi_fn = f'{planeDir}JK{mouse:03}_plane{pn}_cellpose_master_roi.npy'
    if os.path.isfile(master_roi_fn) and (rerun==False):
        print(f'JK{mouse:03} plane {pn} processed already.')
    else:
        ##
        ## ROI collection
        ##
        registration_fn = f'{planeDir}JK{mouse:03}_plane{pn}_session_to_session_registration.npy'
        reg_result = np.load(registration_fn, allow_pickle=True).item()
        num_sessions = len(reg_result['selected_session_num'])
        reg_meth = reg_result['registration_method']
        # Fix the error with suite2p method offsets (was collecting in a wrong way)
        # Match them with the 'old' method (2022/07/13 JK)
        if reg_meth == 'suite2p':
            reg_result = fix_reg_collection(reg_result)

        # Retrieve buffers, options, and dimensions
        if reg_meth == 'old':
            reg_result_ops = reg_result['old_result']
            ybuffer = max(abs(reg_result_ops['roff1'][0][0]))
            xbuffer = max(abs(reg_result_ops['roff1'][0][1]))
        elif reg_meth == 'suite2p':
            reg_result_ops = reg_result['suite2p_result']
            ybuffer = max(abs(reg_result_ops['roff1'][0][0]))
            xbuffer = max(abs(reg_result_ops['roff1'][0][1]))
        elif reg_meth == 'affine':
            reg_result_ops = reg_result['affine_result']
        elif reg_meth == 'bilinear':
            reg_result_ops = reg_result['bilinear_result']
        else:
            raise('Registration method mismatch.')
        reg_img = reg_result_ops['reg_image']
        # Ly, Lx = reg_img.shape[1:]

        # Session-to-session registration creates non-overlaping regions at the edges.
        # Set registration boundary to remove ROIs from each session that overlaps with the boundary.
        if reg_meth == 'old' or reg_meth == 'suite2p':
            top_edge = max(reg_result_ops['roff1'][0][0])+5 # Adding 5 for nonrigid and 2nd step registrations.
            bottom_edge = min(reg_result_ops['roff1'][0][0])-5
            left_edge = max(reg_result_ops['roff1'][0][1])+5
            right_edge = min(reg_result_ops['roff1'][0][1])-5
            registration_boundary = np.ones(reg_img.shape[1:], 'uint8')
            registration_boundary[top_edge:bottom_edge, left_edge:right_edge] = 0
        else:
            registration_boundary = np.sum(reg_img > 0,axis=0) < reg_img.shape[0]

        # Go through sessions and collect ROIs into the master ROI map
        # Pre-sessions (901 and 902) should be at the beginning

        masterMap = np.zeros(0, 'bool')

        srBi = StackReg(StackReg.BILINEAR)
        srAffine = StackReg(StackReg.AFFINE)

        leftBuffer = reg_result['edge_buffer']['leftBuffer']
        topBuffer = reg_result['edge_buffer']['topBuffer']
        rightBuffer = reg_result['edge_buffer']['rightBuffer']
        bottomBuffer = reg_result['edge_buffer']['bottomBuffer']

        master_map_list = []
        new_master_map_list = []
        session_map_list = []
        new_map_list = []
        viable_cell_index_list = []

        # for test
        # session_og_map_list = []
        # session_after_buffer_map_list = []
        # session_reg_map_list = []
        # session_cut_map_list = []

        print(f'ROI collection: JK{mouse:03} plane {pn}')
        for si in range(num_sessions):
            snum = reg_result['selected_session_num'][si]
            if snum in session_nums:
                sname = f'{mouse:03}_{snum:03}'
                print(f'Processing JK{mouse:03} plane {pn} {sname} {si}/{num_sessions-1}')
                
                if (reg_meth == 'old') or (reg_meth == 'suite2p'):
                    rigid_y1 = reg_result_ops['roff1'][0][0][si]
                    rigid_x1 = reg_result_ops['roff1'][0][1][si]
                    nonrigid_y1 = reg_result_ops['nroff1'][0][0][si]
                    nonrigid_x1 = reg_result_ops['nroff1'][0][1][si]
                    
                    rigid_y2 = reg_result_ops['roff2'][0][0][si]
                    rigid_x2 = reg_result_ops['roff2'][0][1][si]
                    nonrigid_y2 = reg_result_ops['nroff2'][0][0][si]
                    nonrigid_x2 = reg_result_ops['nroff2'][0][1][si]

                    block_size1 = reg_result_ops['block_size1']
                    block_size2 = reg_result_ops['block_size2']
                
                # Gather cell map and log session cell index for QC
                tempStat = np.load(f'{planeDir}{snum:03}/plane0/roi/stat.npy', allow_pickle=True)
                tempIscell = np.load(f'{planeDir}{snum:03}/plane0/roi/iscell.npy', allow_pickle=True)
                ops = np.load(f'{planeDir}{snum:03}/plane0/roi/ops.npy', allow_pickle=True).item()
                Ly, Lx = ops['Ly'], ops['Lx']

                if 'inmerge' in tempStat[0].keys():
                    merged_ind = np.where([ts['inmerge']>0 for ts in tempStat])[0]
                    if len(merged_ind) > 0:
                        tempIscell[merged_ind,0] = 0
                tempCelli = np.where(tempIscell[:,0])[0]
                numCell = len(tempCelli)
                tempMap = np.zeros((numCell,Ly,Lx), 'bool')
                for n, ci in enumerate(tempCelli):
                    xi = tempStat[ci]['xpix']
                    yi = tempStat[ci]['ypix']
                    tempMap[n,yi,xi] = 1

                # Remove ROIs overlapping with the registration buffer (for bidirectional noise and optotune ringing noise)
                registration_buffer = np.ones(tempMap.shape[1:], 'uint8')
                registration_buffer[topBuffer:-bottomBuffer, leftBuffer:-rightBuffer] = 0
                ind_remove_buffer = np.where(np.sum((tempMap * registration_buffer), axis=(1,2)))[0]
                tempMap = np.delete(tempMap, ind_remove_buffer, axis=0)
                tempCelli = np.delete(tempCelli, ind_remove_buffer)
                tempMap = tempMap[:,topBuffer:-bottomBuffer, leftBuffer:-rightBuffer]
                # session_og_map_list.append(tempMap)

                # remove ROIs overlappting with the registration boundary.
                # (a little chopped-off ROI could still remain and be transformed.)
                ind_remove_boundary = np.where(np.sum((tempMap * registration_boundary), axis=(1,2)))[0]
                tempMap = np.delete(tempMap, ind_remove_boundary, axis=0)
                tempCelli = np.delete(tempCelli, ind_remove_boundary)
                numCell = len(tempCelli)

                # session_after_buffer_map_list.append(tempMap)
                
                # Transform
                if (reg_meth == 'old') or (reg_meth == 'suite2p'):
                    tempRegMap = twostep_register(tempMap, rigid_y1, rigid_x1, nonrigid_y1, nonrigid_x1, block_size1, 
                                    rigid_y2, rigid_x2, nonrigid_y2, nonrigid_x2, block_size2)
                elif reg_meth == 'affine':
                    tempRegMap = np.zeros(tempMap.shape)
                    for trmi in range(tempRegMap.shape[0]):
                        tempRegMap[trmi,:,:] = srAffine.transform(tempMap[trmi,:,:], tmat=reg_result_ops['tforms'][si])
                elif reg_meth == 'bilinear':
                    tempRegMap = np.zeros(tempMap.shape)
                    for trmi in range(tempRegMap.shape[0]):
                        tempRegMap[trmi,:,:] = srBi.transform(tempMap[trmi,:,:], tmat=reg_result_ops['tforms'][si])
                else:
                    raise('Registration method mismatch')
                # session_reg_map_list.append(tempRegMap)

                # Transformation makes ROI map float values, not binary. 
                # Select threshold per cell after transformation, to have (roughly) matching # of pixels before the transformation
                # Save this threshold value per cell per session
                cutMap = np.zeros((numCell, *reg_img.shape[1:]), 'bool')
                delFromCut = []
                warpCellThresh = np.zeros(numCell)
                after_transform_boundary = np.ones(tempRegMap.shape[1:])
                after_transform_boundary[1:-1,1:-1] = 0
                for ci in range(numCell):
                    numPix = np.sum(tempMap[ci,:,:])
                    cutMap[ci,:,:], warpCellThresh[ci] = calculate_regCell_threshold(tempRegMap[ci,:,:], numPix, thresholdResolution = 0.01)
                    # Remove ROIs that have crossed the boundary
                    # or that have 0 pixels after cutting with threshold
                    if (np.sum(cutMap[ci,:,:])==0) or ((cutMap[ci,:,:] * after_transform_boundary).flatten().any()):
                        delFromCut.append(ci)
                # session_cut_map_list.append(cutMap)
                if len(delFromCut) > 0:
                    cutMap = np.delete(cutMap, np.array(delFromCut), axis=0)
                    viable_cell_index = np.delete(tempCelli, np.array(delFromCut))
                    numCell -= len(delFromCut)
                else:
                    viable_cell_index = tempCelli

                if cutMap.shape[0] != numCell:
                    raise('# of cell mismatch error.')
                # Chronological matching and addition of ROIs
                # When there are matching ROIs, choose the one that has lower perimeter/area ratio
                
                if si > 0: # When it is not the first loop (there exists masterMap from the previous round)
                    masterArea = np.sum(masterMap, axis=(1,2))
                    newArea = np.sum(cutMap, axis=(1,2))
                    masterPar = perimeter_area_ratio(masterMap)
                    newPar = perimeter_area_ratio(cutMap)
                    overlaps = np.zeros((masterMap.shape[0], numCell), 'uint16')
                    unions = np.zeros((masterMap.shape[0], numCell), 'uint16')
                    
                    # Find if there is any matched ROI, per new cells
                    # Calculate overlap and applying the threshold
                    for ci in range(numCell):
                        overlaps[:,ci] = np.sum(masterMap*cutMap[ci,:,:], axis=(1,2))            
                        unionmap = masterMap + np.tile(cutMap[ci,:,:],(masterMap.shape[0],1,1))
                        unions[:,ci] = np.array([len(np.where(um>0)[0]) for um in unionmap])
                    overlapRatioMaster = overlaps/np.tile(np.expand_dims(masterArea, axis=1), (1,numCell))
                    overlapRatioNew = overlaps/np.tile(np.expand_dims(newArea, axis=0), (masterMap.shape[0],1))
                    overlapMatrixOld = np.logical_or(overlapRatioMaster>=roi_overlap_threshold, overlapRatioNew>=roi_overlap_threshold)
                    # # Added matching calculation: Overlap pix # > roi_overlap_threshold of median ROI pix #
                    # # Median ROI calcualted from masterMap. If masterMap does not exist, then from the cutMap.
                    if len(masterArea) > 0:
                        roiPixThresh = roi_overlap_threshold * np.median(masterArea)
                    else:
                        roiPixThresh = roi_overlap_threshold * np.median(newArea)
                    overlapMatrix = np.logical_or(overlaps > roiPixThresh, overlapMatrixOld)
                    
                    # Deal with error cases where there can be multiple matching
                    multiMatchNewInd = np.where(np.sum(overlapMatrix, axis=0)>1)[0]
                    multiMatchMasterInd = np.where(np.sum(overlapMatrix, axis=1)>1)[0]
                    
                    # Deal with multi-matching pairs
                    # First with master ROI, then with new ROIs, because there can be redundancy in multi-matching pairs
                    delFromMasterInd = []
                    delFromNewInd = []
                    delFromMasterInd, delFromNewInd, multiMatchMasterInd, multiMatchNewInd = \
                        check_multi_match_pair(multiMatchMasterInd, multiMatchNewInd, masterPar, newPar, 
                                                overlapMatrix, overlaps, unions, delFromMasterInd, delFromNewInd)

                    if (len(multiMatchNewInd)>0) or (len(multiMatchMasterInd)>0):
                        print(f'{len(multiMatchNewInd)} multi-match for new rois')
                        print(f'{len(multiMatchMasterInd)} multi-match for master rois')
                        raise('Multiple matches found after fixing multi-match pairs.')
                    else:
                        ################ Select what to remove for matched cells based on PAR
                        # For now, treat if there is no multiple matches (because it was dealt in check_multi_match_pair)
                        # Do this until I comb through all examples that I have and decide how to treat
                        # multiple matches (and update check_multi_match_pair)

                        for ci in range(numCell): # for every new roi
                            if ci in delFromNewInd:
                                continue
                            else:
                                matchedMasterInd = np.where(overlapMatrix[:,ci]==True)[0]
                                matchedMasterInd = np.array([mi for mi in matchedMasterInd if mi not in delFromMasterInd])
                                if len(matchedMasterInd)>0: # found a match in the master roi
                                    # Compare perimeter-area ratio (par) between the matches
                                    # Keep smaller par, remove larger PAR
                                    if masterPar[matchedMasterInd] <= newPar[ci]:
                                        delFromNewInd.append(ci)
                                    else:
                                        delFromMasterInd.append(matchedMasterInd[0])
                        if len(delFromMasterInd)>0:
                            newMasterMap = np.delete(masterMap, np.array(delFromMasterInd), axis=0)
                            roiSessionInd = np.delete(roiSessionInd, np.array(delFromMasterInd))
                        else:
                            newMasterMap = masterMap.copy()
                        if len(delFromNewInd)>0:
                            newMap = np.delete(cutMap, np.array(delFromNewInd), axis=0)
                        else:
                            newMap = cutMap.copy()
                        roiNewSessionInd = np.ones(newMap.shape[0])*si
                        print(f'Delete from Master {delFromMasterInd}')
                        print(f'Delete from New {delFromNewInd}')
                    
                        masterMap = np.vstack((newMasterMap, newMap))
                        roiSessionInd = np.concatenate((roiSessionInd, roiNewSessionInd))
                else:
                    newMap = cutMap.copy()
                    masterMap = cutMap.copy()
                    roiSessionInd = np.ones(newMap.shape[0])*si
                    newMasterMap = cutMap.copy()

                # Collect the result        
                master_map_list.append(masterMap) # Master map after each round. The last one is the final master map to be used.
                session_map_list.append(cutMap) # Transformed ROI map of each session after removing those overlapping with the edge
                viable_cell_index_list.append(viable_cell_index) # index of cells from the session's iscell.npy file
                new_master_map_list.append(newMasterMap) # Map of ROIs from the last master map to be included in the master map in this round
                new_map_list.append(newMap) # Map of ROIs from this session to be included in the master map in this round
                print(f'{sname} done.')
            
        # Save the result
        result = {'master_map_list': master_map_list, # Master map after each round. The last one is the final master map to be used.
        'session_map_list': session_map_list,  # Transformed ROI map of each session after removing those overlapping with the edge
        'viable_cell_index_list': viable_cell_index_list, # index of cells from the session's iscell.npy file
        'new_master_map_list': new_master_map_list, # Map of ROIs from the last master map to be included in the master map in this round
        'new_map_list': new_map_list, # Map of ROIs from this session to be included in the master map in this round (After registration)
        'session_nums': session_nums, # Session numbers used for the registration
                }
        np.save(master_roi_fn, result)

        print(f'Collection done: JK{mouse:03} plane {pn}')


def run_roi_matching(mouse, pn, h5_dir, rerun=True):
    if isinstance(h5_dir, Path):
        h5_dir = str(h5_dir) + '/'
    planeDir = f'{h5_dir}{mouse:03}/plane_{pn}/'
    master_roi_fn = f'{planeDir}JK{mouse:03}_plane{pn}_cellpose_master_roi.npy'
    if not os.path.isfile(master_roi_fn):
        raise(f'JK{mouse:03} plane {pn} master ROI map not found.')
    else:
        save_fn = f'{planeDir}JK{mouse:03}_plane{pn}_cellpose_roi_session_to_master.npy'
    if os.path.isfile(save_fn) and (rerun==False):
        print(f'JK{mouse:03} plane {pn} processed already.')
    else:
        master_roi = np.load(master_roi_fn, allow_pickle=True).item()
        master_map_list = master_roi['master_map_list']
        session_map_list = master_roi['session_map_list']
        viable_cell_index_list = master_roi['viable_cell_index_list']
        session_nums = master_roi['session_nums']
        num_sessions = len(session_nums)
    
        print(f'Re-matching with master ROI map: JK{mouse:03} plane {pn}')
        master_map = master_map_list[-1]
        # master_area = np.sum(master_map, axis=(1,2))

        matching_master_roi_index_list = []
        for si in range(num_sessions):
            snum = session_nums[si]
            sname = f'{mouse:03}_{snum:03}'
            print(f'Processing re-matching JK{mouse:03} plane {pn} {sname} {si}/{num_sessions-1}')

            session_map = session_map_list[si]
            # session_area = np.sum(session_map, axis=(1,2))
            viable_cell_index = viable_cell_index_list[si] # This has the same order as the session_map    
            numCell = len(viable_cell_index)

            if session_map.shape[0] != numCell:
                raise('Session map and index mismatch.')

            iou = np.zeros((master_map.shape[0], numCell), 'float32')
            for ci in range(numCell):
                intersection = np.sum(master_map*session_map[ci,:,:], axis=(1,2))
                union = np.sum(master_map + session_map[ci,:,:], axis=(1,2))
                iou[:,ci] = intersection / union
            
            # Take the maximum overlap per ROI in each session map.
            session_matching_master_roi_index = np.argmax(iou, axis=0)
            if len(session_matching_master_roi_index) != numCell:
                raise('Number of cell mismatch error')
            if len(np.unique(session_matching_master_roi_index)) < len(session_matching_master_roi_index):
                # When there are multiple cells matched to the same master ROI, choose the one with the largest iou
                # and set the rest to -1
                multi_matched_ind = []
                for unique_ind in np.unique(session_matching_master_roi_index):
                    if len(np.where(session_matching_master_roi_index==unique_ind)[0])>1:
                        multi_matched_ind.append(unique_ind)
                for mi in multi_matched_ind:
                    session_matched_inds = np.where(session_matching_master_roi_index==mi)[0]
                    assert len(session_matched_inds) > 1
                    # Choose the one with the largest iou, and set the rest to -1
                    session_matched_ious = iou[mi, session_matched_inds]
                    max_ind = np.argmax(session_matched_ious)
                    # session_matching_master_roi_index[session_matched_inds[max_ind]] = mi                
                    session_matching_master_roi_index[session_matched_inds[np.arange(len(session_matched_inds))!=max_ind]] = -1
            matching_master_roi_index_list.append(session_matching_master_roi_index)
        ##
        ## save the result
        ##

        result = {
        'matching_master_roi_index_list': matching_master_roi_index_list, # ROI index in the final master map, that matches with the viable cell in each session.
                }
                
        np.save(save_fn, result)

        print(f'JK{mouse:03} plane {pn} saved.')


def run_roi_collection_and_match_back(mouse, pn, session_nums, h5_dir, roi_overlap_threshold=0.5, rerun=True):
    run_roi_collection(mouse, pn, session_nums, h5_dir, roi_overlap_threshold=roi_overlap_threshold, rerun=rerun)
    run_roi_matching(mouse, pn, h5_dir, rerun=rerun)


if __name__ == "__main__":
    h5_dir = Path(r'E:\TPM\JK\h5')
    expert_mice_df = pd.read_csv(h5_dir / 'expert_mice.csv', index_col=0)
    use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                 ~expert_mice_df['processing_error'].astype(bool) &
                                 ((expert_mice_df.session_type == 'training') |
                                  (expert_mice_df.session_type.str.contains('test')))]
    
    mice = expert_mice_df.mouse.unique()
    planes = np.arange(1,9)
    mm_error_planes = []
    roi_error_planes = []

    for mouse in mice:
        for pn in planes:
            plane_dir = h5_dir / f'{mouse:03}' / f'plane_{pn}'
            master_roi_fn = plane_dir / f'JK{mouse:03}_plane{pn}_cellpose_master_roi.npy'
            cp_roi_collection_results_fn = plane_dir / f'JK{mouse:03}_plane{pn}_cellpose_roi_collection.npy'
            if not master_roi_fn.exists():
                mm_error_planes.append((mouse, pn))
            if not cp_roi_collection_results_fn.exists():
                roi_error_planes.append((mouse, pn))
    mouse_list = [m for m, p in mm_error_planes]
    plane_list = [p for m, p in mm_error_planes]
    session_nums_list = []
    for i in range(len(mm_error_planes)):
        mouse = mouse_list[i]
        plane = plane_list[i]
        session_names = use_mice_df[(use_mice_df.mouse == mouse) & (use_mice_df.plane == plane)].session.values
        session_nums = [int(s) for s in session_names]
        session_nums_list.append(session_nums)
    num_cpu = 6
    with Pool(num_cpu) as p:
        p.starmap(run_roi_collection_and_match_back, [(mouse, plane, session_nums, h5_dir) for mouse, plane, session_nums in zip(mouse_list, plane_list, session_nums_list)])

    # mouse_plane_df = use_mice_df[['mouse', 'plane']].drop_duplicates().reset_index(drop=True)
    # session_nums_list = []
    # for i, row in mouse_plane_df.iterrows():
    #     mouse = row['mouse']
    #     plane = row['plane']
    #     session_names = use_mice_df[(use_mice_df.mouse == mouse) & (use_mice_df.plane == plane)].session.values
    #     session_nums = [int(s) for s in session_names]
    #     session_nums_list.append(session_nums)
    # mouse_list = mouse_plane_df['mouse'].values
    # plane_list = mouse_plane_df['plane'].values

    # num_cpu = 8
    # with Pool(num_cpu) as p:
    #     p.starmap(run_roi_collection_and_match_back, [(mouse, plane, session_nums, h5_dir) for mouse, plane, session_nums in zip(mouse_list, plane_list, session_nums_list)])

            
    # mice = [25, 27, 30, 36, 39, 52]
    # planes = range(1,9)
    # mouse_plane_tuples = list(itertools.product(mice, planes))

    # for mouse in mice:
    #     for pn in planes:
    #         plane_dir = h5_dir / f'{mouse:03}' / f'plane_{pn}'
    #         master_roi_fn = plane_dir / f'JK{mouse:03}_plane{pn}_cellpose_master_roi.npy'
    #         # cp_roi_collection_results_fn = plane_dir / f'JK{mouse:03}_plane{pn}_cellpose_roi_collection.npy'
    #         if not master_roi_fn.exists():
    #             mm_error_planes.append((mouse, pn))
    #         # if not cp_roi_collection_results_fn.exists():
    #         #     roi_error_planes.append((mouse, pn))


    # # cpu_count = cpu_count()
    # # num_cpu = cpu_count - 2
    # num_cpu = 10
    # with Pool(num_cpu) as p:
    #     p.starmap(run_roi_collection, [(mouse, plane, h5_dir) for mouse, plane in mm_error_planes])