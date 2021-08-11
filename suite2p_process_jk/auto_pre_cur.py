'''
Automatic curation of ROIs based on size and morphology
2021/02/10 JK

Added max radius threshold
2021/05/11 JK

Added max compact threshold
2021/05/18 JK

Added removing blood vessel ROIs and edge ROIs
2021/05/23 JK

Return # cell and # not-cell
2021/06/09 JK

'''
import numpy as np
import os

def auto_pre_cur(dataFolder, minRadiusThreUm = 4, maxRadiusThreUm = 20, arThre = 1.5, crThre = 1.4):
    # Load necessary files
    ops = np.load(f'{dataFolder}ops.npy', allow_pickle=True).item()
    f = np.load(f'{dataFolder}F.npy')
    fneu = np.load(f'{dataFolder}Fneu.npy')
    iscell = np.load(f'{dataFolder}iscell.npy')
    stat = np.load(f'{dataFolder}stat.npy', allow_pickle=True)
    
    # Make a list of all ROIs that have lower intensitiy compared to the surroundings
    # - Likely to be blood vessels
    # by selecting the ROIs that have higher average neuropil signal compared to "soma" fluorescence
    # throughout the whole imaging sessions
    
    frameSelect = []
    for i in range(0,len(ops['nframes_per_folder'])):
        tempRange = [*range(sum(ops['nframes_per_folder'][0:i]), sum(ops['nframes_per_folder'][0:i+1]))]
        frameSelect.append(tempRange)
    
    a = fneu - f
    b = np.zeros((np.shape(a)[0],len(frameSelect)))
    for i in range(len(frameSelect)):
        b[:,i] = a[:,frameSelect[i]].mean(axis=1) >= 0
    c = b.sum(axis=1)
    vesselList = list(*np.where(c==len(frameSelect)))

    # Make a list of all ROIs that "touches" the edge of the image
    xmin = ops['xrange'][0]
    xmax = ops['xrange'][1]-1
    ymin = ops['yrange'][0]
    ymax = ops['yrange'][1]-1
    
    edgeList = []
    for i in range(len(stat)):
        if any(stat[i]['xpix']==xmin) or any(stat[i]['xpix']==xmax) or any(stat[i]['ypix']==ymin) or any(stat[i]['ypix']==ymax):
            edgeList.append(i)
            
    # Settings for ROI removal based on size, aspect ratio, and compact
    if 'umPerPix' not in ops:
        mouse = int(ops['ops_path'].split('/plane_')[0][-3:])
        if mouse > 31:
            ops['zoom'] = 1.7
        else:
            ops['zoom'] = 2.0
        ops['umPerPix'] = 1.4/ops['zoom']
        np.save(f'{dataFolder}ops.npy', ops)
    minNpixThre = (minRadiusThreUm/ops['umPerPix']) ** 2 * np.pi
    maxNpixThre = (maxRadiusThreUm/ops['umPerPix']) ** 2 * np.pi
    if dataFolder[-1] != os.path.sep:
        dataFolder = f'{dataFolder}/'
        
    # Curate ROIs
    for i in range(0,len(stat)):        
        if (i in edgeList) or (i in vesselList):
            iscell[i][0] = 0
        else:
            if stat[i]['npix'] <= minNpixThre:
                iscell[i][0] = 0
            if stat[i]['npix'] > minNpixThre:
                if (stat[i]['aspect_ratio'] < arThre) & (stat[i]['compact'] < crThre):
                    iscell[i][0] = 1
                else:
                    iscell[i][0] = 0    
            if stat[i]['npix'] > maxNpixThre:
                iscell[i][0] = 0
    
    # Save to "iscell.npy"
    np.save(f'{dataFolder}iscell.npy', iscell)
    numCell = sum(iscell[:,0])
    numNotCell = len(iscell[:,0]) - numCell
    return numCell, numNotCell
# mouse = 25
# sessions = [22]
# baseDir = 'Y:/Whiskernas/JK/h5/'
# for session in sessions:
#     for pi in range(4,5):
#         dirName = f'{baseDir}{mouse:03}/plane_{pi}/{session:03}/plane0/'
#         auto_pre_cur(dirName, radiusThreUm=4, arThre=1.5)

# sessions = [25]
# baseDir = 'Y:/Whiskernas/JK/h5/'
# for session in sessions:
#     for pi in range(1,9):
#         dirName = f'{baseDir}{mouse:03}/plane_{pi}/{session:03}/plane0/'
#         auto_pre_cur(dirName, radiusThreUm=4, arThre=1.5)


# sessions = [2,3,4,14,19,22,25]
# baseDir = 'Y:/Whiskernas/JK/h5/'
# for session in sessions:
#     for pi in range(1,9):
#         dirName = f'{baseDir}{mouse:03}/plane_{pi}/{session:03}/plane0/'
#         auto_pre_cur(dirName, radiusThreUm=4, arThre=1.5)

# mouse = 37
# sessions = [901,1,6,7,8,13,19,23]
# for session in sessions:
#     for pi in range(1,9):
#         dirName = f'{baseDir}{mouse:03}/plane_{pi}/{session:03}/plane0/'
#         auto_pre_cur(dirName, radiusThreUm=4, arThre=1.5)
