# -*- coding: utf-8 -*-
"""
2021/07/09 JK
Overall suite2p process
"""

#%%
'''
1. Register each session.
'register_each.py'

2. Register mimg of each session to that of ref session.
'register_to_reference.py'
    Check registration and transformation quality in each session with visual inspection.

3. Select sessions that matched to the ref session.
'select_sessions.py'

4-1. Stitch all selected sessions with proper transformation.
'session_stitching.py'
    Check the result if FOV matched well. (Confirm the results from 3)

4-2. ROI detection from the stitched.
'roi_param_search_stitched.py'
    Results in stitchedOps
    
4-3. ROI detection in each select sessions.
'roi_param_search_selectSession.py'

5. Combine ROIs detected in each session and apply them to the stitched.
Along with ROIs detected from the stitched. Accumulate across sessions.
'match_rois_session_to_stitched.py'

Done.