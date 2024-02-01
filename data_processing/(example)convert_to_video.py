# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:37:54 2021

@author: jkim
"""

import h5py
import numpy as np
import napari
import cv2
import ffmpeg
#%%
imstack = []

sns = np.arange(201,209,dtype = int)
for sn in sns:
    fn = f'D:/TPM/JK/h5/027/plane_2/027_9999_{sn}_plane_2.h5'
    f = h5py.File(fn, 'r')
    data = f['data']
    numFrames, height, width = data.shape
    for i in range(numFrames):
        img = data[i,:,:]
        imstack.append(img)

#%% using napari (just to see)
vid = np.array(imstack)

viewer = napari.view_image(vid)

# Maybe later, there will be 'animation' plug in that actually works

#%% using opencv 
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# vidfn = 'test1.avi'

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
vidfn = 'test2.avi'
# DIVX results in smaller file size compared to MJPG

frameRate = 30
# vout = cv2.VideoWriter(vidfn, fourcc, frameRate, (width,height)) # (1)
vout = cv2.VideoWriter(vidfn, fourcc, frameRate, (width,height),0) # (2)
for i in range(numFrames):
    frameIm = np.uint8(data[i,:,:]/(2**8))
    # img = np.repeat(frameIm[:,:,np.newaxis],3, axis=2) # (1)
    img = frameIm # (2)
    vout.write(img)
vout.release()

'''
Key points:
    codec (fourcc), either *'MJPG' or *'DIVX', but not *'H264'
    data format: 3 dimension (height, width, RGB) in uint8
        either by (1) build 3D data, or by (2) adding '0' at the end of cv2.VideoWriter()
'''    

#%% using FFMPEG
'''
Not yet successful
'''
frameRate = 30
vcodec = 'mjpeg'
vidfn = 'test3.avi'
images = data
process = (
    ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height), r=frameRate)
        .output(vidfn, pix_fmt='yuvj420p', vcodec=vcodec)
        .overwrite_output()
        .run_async(pipe_stdin=True)
)
for frame in images:
    frame = np.transpose(np.uint8(frame/(2**8)))
    rgb = np.repeat(frame[:,:,np.newaxis], 3, axis=2)
    process.stdin.write(
        rgb
            .astype(np.uint8)
            .tobytes()
    )
process.stdin.close()
process.wait()