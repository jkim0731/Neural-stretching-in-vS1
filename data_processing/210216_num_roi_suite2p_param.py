"""
Compare # of ROIs detected from different parameters
Excluding JK054 (high tdTomato expression from the nucleus)
2021/02/16 JK
"""
import numpy as np
import matplotlib.pyplot as plt

baseDir = 'D:/TPM/JK/h5/'
mice = [36,41,41,52,53]
sessions = [13,8,20,25,12]
numPlanes = 8
numParam = 4
numMice = len(mice)
numRoi = np.zeros((numMice,numParam,numPlanes))

for i in range(0,len(mice)):
    mouse = mice[i]
    session = sessions[i]
    for pi in range(1,9):
        for param in range(0,4):
            dirName = f'{baseDir}{mouse:03}/plane_{pi}/{session:03}_param{param}/plane0/'
            fn = f'{dirName}iscell.npy'
            iscell = np.load(fn)
            numRoi[i,param,pi-1] = np.sum(iscell[:,0])
#%%
propRoi = numRoi / np.tile(numRoi[:,0,:].reshape(5,1,8), [1,4,1])

#%%
fig, axs = plt.subplots(2,4, sharex = True, sharey = True, figsize=(9,6) )
for mi in range(0,5):
    for pi in range(0,8):
        axy, axx = np.divmod(pi, 4)
        if pi == 0:
            axs[axy,axx].plot(range(0,4), propRoi[mi,:,pi], label=f'session#{mi}')
            axs[axy,axx].legend(loc=2, ncol=1,prop={'size':10})
        else:
            axs[axy,axx].plot(range(0,4), propRoi[mi,:,pi])
        axs[axy,axx].set_title(f'Plane {pi+1}')
        
        

for ax in axs.flat:
    ax.set(xlabel='Parameter set', ylabel='Normalized # ROI', xticks=range(0,4))
    
for ax in axs.flat:
    ax.label_outer()

plt.tight_layout()

'''
Result: Always better at parameter 3, which was double registration (helped in upper volumes too)
and temporal smoothing gaussian sigma 2
'''
#%%
# One exception, from JK052 session 25 plane 8
# It turns out that JK052 has strong tdTomato from nuclei at lower volumes
mouse = 52
session = 25
baseDir = f'D:/TPM/JK/h5/{mouse:03}/'
mimgs = []
for pi in range(1,9):
    fn = f'{baseDir}plane_{pi}/{session:03}_param0/plane0/ops.npy'
    ops = np.load(fn, allow_pickle = True).item()
    temp = ops['meanImg']
    p1, p2 = np.percentile(temp, (1, 99.9))
    temp[temp<p1] = p1
    temp[temp>p2] = p2
    # temp = (temp-p1) / (p2-p1)
    mimgs.append(temp)

fig, axs = plt.subplots(2,4, figsize = (9,3), dpi = 600)
for pi in range(0,8):
    axy, axx = np.divmod(pi, 4)
    axs[axy,axx].imshow(mimgs[pi], cmap = 'gray')
    axs[axy,axx].set(xticks=[], yticks=[])
    axs[axy,axx].set_title(f'plane {pi+1}')
    
#%%
temp = ops['meanImg']
fig, axs = plt.subplots(1,2)
axs[0].imshow(temp, cmap = 'gray')
p1, p2 = np.percentile(temp, [5,95])
temp[temp<p1] = p1
temp[temp>p2] = p2
temp = (temp-p1) / (p2-p1)
axs[1].imshow(temp, cmap='gray')
plt.tight_layout()