"""
Summarizing the results of z-drift calculation (from 211026_z_drift_multi.py)
(1) Test z-drift consistency across optotune foci - comparing between plane 1 vs 4 and 5 vs 8 
(2) Test z-drift consistency across brain depth - comparing between plane 1 vs 5
(3) Summary statistics of z-drift in each session
(4) Summary statistics of depth of focus vs # of sessions 
2021/12/02 JK

"""
import numpy as np
import matplotlib.pyplot as plt
import glob

baseDir = 'C:/JK/h5temp/'
#%% (1) Test z-drift consistency across optotune foci - comparing between plane 1 vs 4 and 5 vs 8 
mouse = 2
planes = [1,4]
# planes = [5,8]

# excludeSnList = ['027_004']
excludeSnList = []

p0 = planes[0]
p1 = planes[1]
data0 = np.load(f'{baseDir}JK{mouse:03}_zdrift_plane{p0}.npy', allow_pickle=True).item()
data1 = np.load(f'{baseDir}JK{mouse:03}_zdrift_plane{p1}.npy', allow_pickle=True).item()

# Select sessions of the same name
# For depth matching, use all sessions
# For z-drift matching, use regular sessions (len(sname) < 4)
if mouse < 52:
    ssiList = data0['info']['selectedSi']
    snlist0 = [data0['info']['sessionNames'][si] for si in ssiList]
    ssiList = data1['info']['selectedSi']
    snlist1 = [data1['info']['sessionNames'][si] for si in ssiList]
else:
    snlist0 = data0['info']['sessionNames']
    snlist1 = data1['info']['sessionNames']
matchedSnList = [sn for sn in snlist1 if sn in snlist0]
matchi0 = [i for (i,sn) in enumerate(snlist0) if sn in matchedSnList]
matchi1 = [i for (i,sn) in enumerate(snlist1) if sn in matchedSnList]

#%% Best matching depths for session mimg
maxDepth0 = [np.argmax(data0['session']['bestCorrVals'][si,:]) for si in matchi0]
maxDepth1 = [np.argmax(data1['session']['bestCorrVals'][si,:]) for si in matchi1]
xrange = maxDepth0-np.mean(maxDepth0)
yrange = maxDepth1-np.mean(maxDepth1)
scatterMin = min(min(xrange), min(yrange))-1
scatterMax = max(max(xrange), max(yrange))+1
fig, ax = plt.subplots()
ax.scatter(maxDepth0-np.mean(maxDepth0), maxDepth1-np.mean(maxDepth1), 30, 'k')
ax.plot([scatterMin, scatterMax], [scatterMin, scatterMax], 'k--')
ax.set_xlim([scatterMin, scatterMax])
ax.set_ylim([scatterMin, scatterMax])
ax.set_xlabel(f'Plane {p0}', fontsize=12)
ax.set_ylabel(f'Plane {p1}', fontsize=12)
rho = np.corrcoef(xrange, yrange)[0,1]
ax.set_title(f'JK{mouse:03}\nr={rho:.2f}', fontsize=15)

#%% z-drift comparison for each regular session
regMatchedSnList = [sn for sn in matchedSnList if sn not in excludeSnList]
regMatchedSnList = [sn for sn in regMatchedSnList if len(sn)==7]
regSi0 = [i for (i,sn) in enumerate(snlist0) if sn in regMatchedSnList]
regSi1 = [i for (i,sn) in enumerate(snlist1) if sn in regMatchedSnList]

rhoList = []
firstx = []
firsty = []
fig, ax = plt.subplots()
for i in range(len(regSi0)):
    xdata = data0['zdriftList'][regSi0[i]]
    ydata = data1['zdriftList'][regSi1[i]]
    minlength = min(len(xdata), len(ydata))
    xdata = xdata[:minlength]
    ydata = ydata[:minlength]
    ax.plot(xdata, ydata, 'k-')
    ax.scatter(xdata[0], ydata[0], 30, 'r')
    if minlength > 4:
        rhoList.append(np.corrcoef(xdata, ydata)[0,1])
    firstx.append(xdata[0])
    firsty.append(ydata[0])
ax.set_xlabel(f'Plane {p0}', fontsize=12)
ax.set_ylabel(f'Plane {p1}', fontsize=12)
meanRho = np.nanmean(rhoList)
rho = np.corrcoef(firstx, firsty)[0,1]
ax.set_title(f'JK{mouse:03}\nmean(r)={meanRho:.2f}\n first r = {rho:.2f}', fontsize=15)


#%% (2) Test z-drift consistency across brain depth - comparing between plane 1 vs 5
mouse = 25
planes = [1,5]

# excludeSnList = ['027_004']
excludeSnList = []

p0 = planes[0]
p1 = planes[1]
data0 = np.load(f'{baseDir}JK{mouse:03}_zdrift_plane{p0}.npy', allow_pickle=True).item()
data1 = np.load(f'{baseDir}JK{mouse:03}_zdrift_plane{p1}.npy', allow_pickle=True).item()

# Select sessions of the same name
# For depth matching, use all sessions
# For z-drift matching, use regular sessions (len(sname) < 4)
if mouse < 52:
    ssiList = data0['info']['selectedSi']
    snlist0 = [data0['info']['sessionNames'][si] for si in ssiList]
    ssiList = data1['info']['selectedSi']
    snlist1 = [data1['info']['sessionNames'][si] for si in ssiList]
else:
    snlist0 = data0['info']['sessionNames']
    snlist1 = data1['info']['sessionNames']
matchedSnList = [sn for sn in snlist1 if sn in snlist0]
matchi0 = [i for (i,sn) in enumerate(snlist0) if sn in matchedSnList]
matchi1 = [i for (i,sn) in enumerate(snlist1) if sn in matchedSnList]

#% Best matching depths for session mimg
# maxDepth0 = [np.argmax(data0['session']['bestCorrVals'][si,:]) for si in matchi0]
# maxDepth1 = [np.argmax(data1['session']['bestCorrVals'][si,:]) for si in matchi1]
# xrange = maxDepth0-np.mean(maxDepth0)
# yrange = maxDepth1-np.mean(maxDepth1)
# scatterMin = min(min(xrange), min(yrange))-1
# scatterMax = max(max(xrange), max(yrange))+1
# fig, ax = plt.subplots()
# ax.scatter(maxDepth0-np.mean(maxDepth0), maxDepth1-np.mean(maxDepth1), 30, 'k')
# ax.plot([scatterMin, scatterMax], [scatterMin, scatterMax], 'k--')
# ax.set_xlim([scatterMin, scatterMax])
# ax.set_ylim([scatterMin, scatterMax])
# ax.set_xlabel(f'Plane {p0}', fontsize=12)
# ax.set_ylabel(f'Plane {p1}', fontsize=12)
# rho = np.corrcoef(xrange, yrange)[0,1]
# ax.set_title(f'JK{mouse:03}\nr={rho:.2f}', fontsize=15)

#% z-drift comparison for each regular session
regMatchedSnList = [sn for sn in matchedSnList if sn not in excludeSnList]
regMatchedSnList = [sn for sn in regMatchedSnList if len(sn)==7]
regSi0 = [i for (i,sn) in enumerate(snlist0) if sn in regMatchedSnList]
regSi1 = [i for (i,sn) in enumerate(snlist1) if sn in regMatchedSnList]

rhoList = []
fig, ax = plt.subplots()
for i in range(len(regSi0)):
    xdata = data0['zdriftList'][regSi0[i]]
    ydata = data1['zdriftList'][regSi1[i]]
    minlength = min(len(xdata), len(ydata))
    xdata = xdata[:minlength]
    ydata = ydata[:minlength]
    ax.plot(xdata, ydata, 'k-')
    ax.scatter(xdata[0], ydata[0], 30, 'r')
    if minlength >= 4:
        rhoList.append(np.corrcoef(xdata, ydata)[0,1])    
ax.set_xlabel(f'Plane {p0}', fontsize=12)
ax.set_ylabel(f'Plane {p1}', fontsize=12)
meanRho = np.nanmean(rhoList)
ax.set_title(f'JK{mouse:03}\nmean(r)={meanRho:.2f}', fontsize=15)
fig.tight_layout()


#%% (3) Summary statistics of z-drift in each session
# mean z-drift um/hr
mice = [25,27,30,36,39,52]
colors = ['k', 'r', 'm', 'b', 'c', 'g']

fig, ax = plt.subplots()
for pn in [1,4,5,8]:
    # first, find mice that have the corresponding plane
    fnlist = glob.glob(f'{baseDir}JK*_zdrift_plane{pn}.npy')
    numMice = len(fnlist)
    offsets = [-(numMice//2-i)*0.02 for i in range(numMice)]
    for fi, fn in enumerate(fnlist):
        mouse = fn.split('\\')[1].split('_')[0][3:]
        ci = [i for i in range(len(mice)) if mice[i]== int(mouse)][0]
        data = np.load(fn, allow_pickle = True).item()
        if mouse=='39' and pn == 8:
            zdriftList = data['zdriftList'][:-1]
        elif mouse=='27' and pn <5:
            zdriftList = data['zdriftList'][:3]+data['zdriftList'][4:]
        else:
            zdriftList = data['zdriftList']
        meanDrift = np.mean(np.array([(drift[-1]-drift[0])/len(drift)*6*2 for drift in zdriftList if len(drift) > 3]))
        stdDrift = np.std(np.array([(drift[-1]-drift[0])/len(drift)*6*2 for drift in zdriftList if len(drift) > 3]))
        print(f'{mouse} plane {pn}: {meanDrift:.2f} +- {stdDrift:.2f} um/hr')
        ax.errorbar(pn+offsets[fi], meanDrift, stdDrift, marker='o', markerfacecolor=colors[ci], color = colors[ci])

ax.set_xlabel('Imaging plane')
ax.set_ylabel(r'Mean z-drift ($\mu$m)')


#%% (4) Summary statistics of depth of focus vs # of sessions 
# only count regular sessions
slabList = [10, 15, 20, 25]
for pn in [1, 5]:
    fnlist = glob.glob(f'{baseDir}JK*_zdrift_plane{pn}.npy')
    for fn in fnlist:
        data = np.load(fn, allow_pickle = True).item()
        mouse = fn.split('\\')[1].split('_')[0][3:]
        nRegSession = np.amax(np.array([i for i, sn in enumerate(data['info']['sessionNames']) if len(sn)==7]))
        zdriftList = data['zdriftList'][:nRegSession]
        if mouse=='27' and pn ==1:
            zdriftList = zdriftList[:3]+zdriftList[4:]
        minDepth = np.amin(np.array([min(zd) for zd in zdriftList]))
        maxDepth = np.amax(np.array([max(zd) for zd in zdriftList]))
        for slab in slabList:
            slabNsession = []    
            for depth in range(minDepth, maxDepth-slab//2+1):
                depthNsession = []
                zrange = [depth, depth+slab/2]
                numSession = 0
                for zd in zdriftList:
                    timeInd = [z for z in zd if z >= depth and z <= depth+slab/2]
                    if len(timeInd) >= 3:
                        numSession += 1
                depthNsession.append(numSession)
                if len(depthNsession):
                    slabNsession.append(max(depthNsession))
            if len(slabNsession):
                maxNsession = max(slabNsession)
                print(f'JK0{mouse} plane {pn} with {slab} um: {maxNsession} sessions')
        
    
    
