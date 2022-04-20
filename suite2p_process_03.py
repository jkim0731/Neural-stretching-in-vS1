# -*- coding: utf-8 -*-
"""
2022/01/26 JK
Overall suite2p process
"""

1. z-drift estimation
- JK0xx_zdrift_plane(1 or 5).npy @ h5Dir
2. Nonrigid registration
- s2p_nr_reg.npy @ planeDir
3. Manual ROI curation
- For merged ROIs, run 220120_check_gui2p_merge_spk.py
4. ROI collection
- JK0xxplane{pn}masterROI.npy @ planeDir
5. Apply master ROI map, extract signals (even lam)
- newF.npy, newFneu.npy, newSpks.npy @ {sessionDir}plane0/
6. GLM for touch responsiveness
7. GLM for whisker feature encoding

