import numpy as np
import xarray as xr
import pandas as pd
import utils.merged_df_annotation as mda


def get_touch_response_volume_xr(base_dir, mouse, top_plane, session, touch_window='before_answer',
                                 spk_norm='std', post_touch_frames=1):
    assert touch_window in ['before_answer', 'after_answer', 'all'],\
                'touch_window should be either "before_answer", "after_answer", or "all"'
    planes = range(top_plane, top_plane + 4)
    for pi, plane in enumerate(planes):
        per_touch_response_xr_plane, per_touch_response_df_plane = \
            get_touch_response(base_dir, mouse, session, plane, touch_window=touch_window,
                               spk_norm=spk_norm, post_touch_frames=post_touch_frames)
        if pi == 0:
            per_touch_response_xr = per_touch_response_xr_plane.copy()
            per_touch_response_df = per_touch_response_df_plane.copy()
            touch_trial_nums = per_touch_response_xr_plane.trialNum.values
        else:
            # assert per_touch_response_df.equals(per_touch_response_df_plane)
            # assert per_touch_response_xr.shape[0] == per_touch_response_xr_plane.shape[0]
            # assert per_touch_response_xr.trialNum.equals(per_touch_response_xr_plane.trialNum)
            touch_trial_nums = np.intersect1d(touch_trial_nums, per_touch_response_xr_plane.trialNum.values)            
            per_touch_response_xr = xr.concat([per_touch_response_xr.sel(trialNum=touch_trial_nums),
                                               per_touch_response_xr_plane.sel(trialNum=touch_trial_nums)],
                                               dim='cell_id')
        per_touch_response_df = per_touch_response_df[per_touch_response_df.trialNum.isin(touch_trial_nums)]
        assert np.equal(per_touch_response_xr.trialNum.values, per_touch_response_df.trialNum.values).all()

    return per_touch_response_xr, per_touch_response_df


def get_touch_response(base_dir, mouse, session, plane,
                       touch_window='before_answer', spk_norm='std', post_touch_frames=1):
    assert touch_window in ['before_answer', 'after_answer', 'all'],\
                'touch_window should be either "before_answer", "after_answer", or "all"'
    plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    behavior_frametime = pd.read_pickle(plane_dir / f'JK{mouse:03}_S{session:02}_plane{plane}_frame_whisker_behavior.pkl')
    roi_dir = plane_dir / f'{session:03}/plane0/roi'
    ophys_frametime = pd.read_pickle(roi_dir / 'refined_frame_time.pkl')

    # get merged df
    merged_df = mda.get_merged_df(ophys_frametime, behavior_frametime)
    # assign pole_moving_up and pole_moving_down to the frames
    merged_df = mda.assign_pole_moving_frames(merged_df)

    # get spks
    norm_spks = mda.get_normalized_spikes(roi_dir, ophys_frametime, merged_df, spk_norm=spk_norm)
    
    merged_df['spks_frame_ind'] = np.arange(norm_spks.shape[1])

    # get touch response frames
    merged_df = mda.assign_touch_response_frames(merged_df, post_touch_frames=post_touch_frames)

    # Get touch response of spks per trial
    if touch_window == 'before_answer':
        touch_trial_nums = merged_df[merged_df.before_answer_touch_frame].trialNum.unique()
    elif touch_window == 'after_answer':
        touch_trial_nums = merged_df[merged_df.after_answer_touch_frame].trialNum.unique()
    else:
        touch_trial_nums = merged_df[merged_df.touch_response_frame].trialNum.unique()
    merged_df = merged_df[merged_df.trialNum.isin(touch_trial_nums)]
    merged_df['num_touch'] = np.nan

    per_touch_responses = []
    for tn in touch_trial_nums:
        trial_df = merged_df[merged_df.trialNum==tn]
        
        if touch_window == 'before_answer':
            touch_response_frame_inds = trial_df[trial_df.before_answer_touch_frame].spks_frame_ind
            num_touch = np.nansum(trial_df.before_answer_touch_count.values)
        elif touch_window == 'after_answer':
            touch_response_frame_inds = trial_df[trial_df.after_answer_touch_frame].spks_frame_ind
            num_touch = np.nansum(trial_df.after_answer_touch_count.values)
        else:
            touch_response_frame_inds = trial_df[trial_df.touch_response_frame].spks_frame_ind
            num_touch = np.nansum(trial_df.touch_count.values)
        merged_df.loc[merged_df.trialNum==tn,'num_touch'] = num_touch
        touch_response_spks = np.sum(norm_spks[:,touch_response_frame_inds], axis=1)
        per_touch_responses.append(touch_response_spks / num_touch)
    cell_ids = [f'p{plane}c{ci:04}' for ci in norm_spks.cell_index.values]
    per_touch_response_xr = xr.DataArray(per_touch_responses,
                                         dims=['trialNum', 'cell_id'],
                                         coords={'trialNum': touch_trial_nums, 'cell_id': cell_ids})
    per_touch_response_df = merged_df[['trialNum', 'pole_angle', 'correct', 'wrong', 'miss', 'num_touch']].drop_duplicates()
    return per_touch_response_xr, per_touch_response_df


def get_touch_response_xr_varexp_threshold(base_dir, mouse, top_plane, session, touch_window='before_answer',
                                           spk_norm='std', varexp_threshold=0.05,
                                           post_touch_frames=1):
    assert touch_window in ['before_answer', 'after_answer', 'all'],\
                'touch_window should be either "before_answer", "after_answer", or "all"'
    per_touch_response_xr, per_touch_response_df = \
        get_touch_response_volume_xr(base_dir, mouse, top_plane, session, touch_window=touch_window,
                                    spk_norm=spk_norm, post_touch_frames=post_touch_frames)
    fit_cell_inds = []
    for plane in range(top_plane, top_plane+4):
        glm_result = get_glm_results(base_dir, mouse, plane, session)
        cell_inds = get_cell_inds_varexp_threshold(per_touch_response_xr, glm_result, plane, varexp_threshold=varexp_threshold)
        fit_cell_inds.extend(cell_inds)
    per_touch_response_xr_fit = per_touch_response_xr.isel(cell_id=fit_cell_inds)
    assert len(fit_cell_inds) == per_touch_response_xr_fit.shape[1]
    return per_touch_response_xr_fit, per_touch_response_df, per_touch_response_xr


def get_glm_results(base_dir, mouse, plane, session):
    plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    roi_dir = plane_dir / f'{session:03}/plane0/roi'
    glm_dir = roi_dir / 'glm/touch_combined'
    glm_result = xr.open_dataset(glm_dir / 'glm_result.nc')
    return glm_result


def get_cell_inds_varexp_threshold(per_touch_response_xr, glm_result, plane, varexp_threshold=0.05):
    # get cell ids with varexp_model_final > varexp_threshold
    cell_ids = glm_result.cell_id.values
    varexp_model_final = glm_result.varexp_model_final.values
    cell_ids = cell_ids[varexp_model_final > varexp_threshold]
    return np.where([cid in [f'p{plane}c{ci:04}' for ci in cell_ids] for cid in per_touch_response_xr.cell_id.values])[0]

