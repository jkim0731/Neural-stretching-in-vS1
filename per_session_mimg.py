
from suite2p.io.binary import BinaryFile
import matplotlib.pyplot as plt
import numpy as np
import os

def per_session_mimg(planeDir, mouse, plane):
    '''
    Session-wide mean images from data.bin
    And their corresponding frame indices and session names from ops.npy
    
    Treatment for spontaneous sessions should be different among mice
    For mouse < 52: 
        each spontaneous session is recorded in saved h5
    For mouse == 52:
        First spontaneous session (052_555x_1yy) is in single h5
        All the rest spontaneous sessions are in 20 h5 files
    For mouse > 52:
        All spontaneous sessions are recorded in 20 h5 files
    
    Parameters
    ----------
    planeDir : str
        Directory name where data.bin and ops.npy are located
    mouse : int
        # of mouse. (36-56) Required for spontaneous session treatment
    plane : int
        # of plane. 1-8

    Returns
    -------
    None.
    Saves perSession{mimgs, frameIndsPerSession, frameIndsPerH5, h5List, sessionlist} in per_session_mimg.npy
    perSession['mimgs']: mean images from each session (after combining different h5 files from same session)
    perSession['frameIndsPerSession']: Index of frames in data.bin that belongs to each session
    perSession['frameIndsPerH5']: Index of frames in data.bin that belong to each h5 file
    perSession['h5List']: Each h5 file name (matched with 'frameInds')
    perSession['slist']: Session names (matched with 'frameIndsPerSession')
        
    '''
    if planeDir[-1] != os.path.sep:
        planeDir = f'{planeDir}/'
    binfn = f'{planeDir}data.bin'
    opsfn = f'{planeDir}ops.npy'
    ops = np.load(opsfn, allow_pickle = True).item()
    Ly = ops['Ly']
    Lx = ops['Lx']
    perFileMeanImg = []
    
    # Combining files from the same session
    # Multiple sbx from same training sessions were already combined when transferred to h5
    # Spontaneous and piezo sessions were not.
    # For mi < 8: spontaneous sessions were saved in each h5 file
    # For mi >= 8: spontaneous sessions were divided into 20 files starting from the second spont session
    # For all mice: pizeo sessions were divided.
    slist = []
    h5List = []
    h5ind = []
    if mouse < 52:
        for f5name in ops['h5list']:
            fstr1 = f5name.split(f'plane_{plane}\\')
            fstr2 = fstr1[1].split('_plane_')[0].split('_')
            h5List.append(f'{fstr2[0]}_{fstr2[1]}_{fstr2[2]}')        
            if int(fstr2[1]) < 9000:
                slist.append(f'{fstr2[0]}_{fstr2[1]}_{fstr2[2]}')
            else: # in case of piezo sessions
                sstr = f'{fstr2[0]}_{fstr2[1]}_{fstr2[2][0]}'
                if sstr not in slist: # make session name list a unique list
                    slist.append(sstr)
        for sn in slist:
            tempInd = [i for i, fn in enumerate(h5List) if sn in fn]
            h5ind.append(tempInd)
    elif mouse == 52:
        for f5name in ops['h5list']:
            fstr1 = f5name.split(f'plane_{plane}\\')
            fstr2 = fstr1[1].split('_plane_')[0].split('_')
            h5List.append(f'{fstr2[0]}_{fstr2[1]}_{fstr2[2]}')        
            if int(fstr2[1]) < 9000:
                if int(fstr2[2]) < 200: # single h5 file for the first spontaneous session in JK052
                    slist.append(f'{fstr2[0]}_{fstr2[1]}_{fstr2[2]}')
                else: # treat like piezo sessions (multiple h5 files for single session)
                    sstr = f'{fstr2[0]}_{fstr2[1]}_{fstr2[2][0]}'
                    if sstr not in slist: # make session name list a unique list
                        slist.append(sstr)
            else: # in case of piezo sessions
                sstr = f'{fstr2[0]}_{fstr2[1]}_{fstr2[2][0]}'
                if sstr not in slist: # make session name list a unique list
                    slist.append(sstr)
        for sn in slist:
            tempInd = [i for i, fn in enumerate(h5List) if sn in fn]
            h5ind.append(tempInd)
    else:
        for f5name in ops['h5list']:
            fstr1 = f5name.split(f'plane_{plane}\\')
            fstr2 = fstr1[1].split('_plane_')[0].split('_')
            h5List.append(f'{fstr2[0]}_{fstr2[1]}_{fstr2[2]}')        
            if int(fstr2[1]) < 5000:
                slist.append(f'{fstr2[0]}_{fstr2[1]}_{fstr2[2]}')
            else: # in case of spontaneous & piezo sessions
                sstr = f'{fstr2[0]}_{fstr2[1]}_{fstr2[2][0]}'
                if sstr not in slist: # make session name list a unique list
                    slist.append(sstr)
        for sn in slist:
            tempInd = [i for i, fn in enumerate(h5List) if sn in fn]
            h5ind.append(tempInd)    
        
    #%%
    # Some sessions are separated (piezo sessions)
    # So, I need to explicitly list the frames for each file
    # To gather them correctly later (into a single session)
    cumsumNframes = np.insert(np.cumsum(ops['nframes_per_folder']),0,0)
    frameIndsPerH5 = []
    for i in range(len(cumsumNframes)-1):
        frameIndsPerH5.append([*range(cumsumNframes[i],cumsumNframes[i+1])])
    
    #%%
    frameIndsPerSession = []
    for i in range(len(h5ind)):
        if len(h5ind[i]) < 1:
            raise(f'No h5 index at {i}')
        elif len(h5ind[i]) == 1:
            frameIndsPerSession.append(frameIndsPerH5[h5ind[i][0]])
        else:
            tempInds = []
            for j in range(len(h5ind[i])):
                tempInds = tempInds + [k for k in frameIndsPerH5[h5ind[i][j]]]
            frameIndsPerSession.append(tempInds)
    
    
    #%%
    with BinaryFile(Ly = Ly, Lx = Lx, read_filename = binfn) as f:
        for inds in frameIndsPerSession:
            frames = f.ix(indices=inds).astype(np.float32)
            perFileMeanImg.append(frames.mean(axis=0))

    #%%
    perSession = {'mimgs': perFileMeanImg, 'frameIndsPerSession': frameIndsPerSession, 'frameIndsPerH5': frameIndsPerH5, 
                  'h5List': h5List, 'sessionList': slist}
    
    np.save(f'{planeDir}per_session_mimg.npy', perSession)