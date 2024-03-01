import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from time import time
from multiprocessing import Pool
import sys
sys.path.append(r'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Analysis\codes\data_analysis')
from merged_df_annotation import get_merged_df, \
    assign_pole_moving_frames, assign_touch_response_frames, \
        assign_baseline_frames, get_normalized_spikes
from touch_tuning import get_touch_response_from_baseline, \
    get_touch_cell_inds, get_tuned_cell_inds

base_dir = Path(r'E:\TPM\JK\h5')

expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                 ~expert_mice_df['processing_error'].astype(bool) &
                                 ((expert_mice_df.session_type == 'training') |
                                  (expert_mice_df.session_type.str.contains('test')))]

def save_touch_tuning(mouse, plane, session):
    print(f'Processing JK{mouse:03} S{session:02} plane {plane}')
    plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    behavior_frametime = pd.read_pickle(plane_dir / f'JK{mouse:03}_S{session:02}_plane{plane}_frame_whisker_behavior.pkl')
    roi_dir = plane_dir / f'{session:03}/plane0/roi'

    touch_tuning_dir = roi_dir / 'touch_tuning'
    touch_tuning_dir.mkdir(exist_ok=True)

    touch_windows = ['before_answer', 'after_answer', 'all']

    ophys_frametime = pd.read_pickle(roi_dir / 'refined_frame_time.pkl')
    merged_df = get_merged_df(ophys_frametime, behavior_frametime)
    merged_df = assign_pole_moving_frames(merged_df)
    merged_df = assign_baseline_frames(merged_df)
    merged_df = assign_touch_response_frames(merged_df, post_touch_frames=1)
    norm_spks = get_normalized_spikes(roi_dir, ophys_frametime, merged_df, spk_norm='std')
    assert (norm_spks.frame_index.values == merged_df.frame_index.values).all()

    for touch_window in touch_windows:
        save_fn = touch_tuning_dir / f'touch_tuning_{touch_window}.npy'
        if save_fn.exists():
            continue
        touch_response_from_baseline = \
            get_touch_response_from_baseline(merged_df,
                            norm_spks, touch_window=touch_window)
        touch_cell_inds, ttest_results = \
            get_touch_cell_inds(touch_response_from_baseline,
                                             merged_df, pval_threshold=0.01)
        tuned_cell_inds, stat_results = \
            get_tuned_cell_inds(touch_response_from_baseline,
                            merged_df, touch_cell_inds, pval_threshold=0.01)
        results = {'touch_response': touch_response_from_baseline,
                   'touch_cell_inds': touch_cell_inds,
                   'tuned_cell_inds': tuned_cell_inds,
                   'ttest_results': ttest_results,                   
                   'stat_results': stat_results}
        np.save(save_fn, results)

if __name__ == '__main__':
    t0 = time()
    with Pool(19) as p:
        p.starmap(save_touch_tuning, [(row.mouse, row.plane, int(row.session)) for _, row in use_mice_df.iterrows()])
    elapsed = time() - t0
    print(f"Elapsed time: {elapsed//3600} hr {(elapsed%3600)//60} min {elapsed%60} sec")

    # 25 min for expert use_mice