import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

base_dir = Path(r'E:\TPM\JK\h5')

expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                 ~expert_mice_df['processing_error'].astype(bool) &
                                 ((expert_mice_df.session_type == 'training') |
                                  (expert_mice_df.session_type.str.contains('test')))]
mice = use_mice_df.mouse.unique()

planes = range(1,9)
bins = [*np.linspace(0, 10, 1000), 100]

cum_hist_allplanes = []
for plane in planes:
    cum_hist_mean = []
    for mouse in mice:
        if mouse != 52:
            cum_hist_plane = []
            session_names = use_mice_df.query('mouse==@mouse and plane==@plane').session.unique()
            for session_name in session_names:
                plane_dir = base_dir / f'{mouse:03}' / f'plane_{plane}'
                roi_dir = plane_dir / f'{session_name}/plane0/roi'
                stats = np.load(roi_dir / 'stat_refined.npy', allow_pickle=True)
                skews = [abs(s['skew']) for s in stats]
                hist_skew = np.histogram(skews, bins=bins)
                cum_hist = np.cumsum(hist_skew[0])/len(skews)
                cum_hist_plane.append(cum_hist)
            cum_hist_plane = np.asarray(cum_hist_plane)
            cum_hist_mean.append(np.mean(cum_hist_plane, axis=0))
    cum_hist_mean = np.asarray(cum_hist_mean)
    cum_hist_allplanes.append(cum_hist_mean)

percentile_threshold = 0.05
skew_thresh_all = []
for pi in range(8):
    skew_thresh_plane = []
    for mi in range(5):
        x = [0, *cum_hist_allplanes[pi][mi]]
        y = [*np.linspace(0, 10, 1000), 100]
        interp_func = interp1d(x, y)
        skew_thresh_plane.append(interp_func(percentile_threshold))
    skew_thresh_all.append(np.asarray(skew_thresh_plane))
skew_thresholds = [np.mean(sta) for sta in skew_thresh_all]

base_dir = Path(r'E:\TPM\JK\h5')
mouse = 52
planes = np.arange(1,9)
for pi in range(8):
    plane = planes[pi]
    plane_dir = base_dir / f'{mouse:03}' / f'plane_{plane}'
    skew_threshold = skew_thresholds[pi]
    session_names = use_mice_df.query('mouse==@mouse and plane==@plane').session.unique()
    for session_name in session_names:        
        session_dir = plane_dir / f'{session_name}' / 'plane0'
        roi_dir = session_dir / 'roi'
        iscell = np.load(roi_dir / 'iscell.npy', allow_pickle=True)
        stats = np.load(roi_dir / 'stat_refined.npy', allow_pickle=True)
        for i in range(len(stats)):
            if abs(stats[i]['skew']) < skew_threshold:
                iscell[i,:] = 0
        np.save(roi_dir / 'iscell.npy', iscell)

    