import numpy as np
import pandas as pd
import xarray as xr


def get_merged_df(ophys_frametime, behavior_frametime):
    # remove trials with errors
    refined_ophys_frametime = ophys_frametime.query('remove_trial==False')
    assert refined_ophys_frametime.remove_frame.values.sum() == 0

    # merge refined_ophys_frametime and behavior_frametime
    reduced_behavior_columns = np.setdiff1d(behavior_frametime.columns,
                                            np.setdiff1d(refined_ophys_frametime.columns,
                                                         ['trialNum', 'frame_index']))
    reduced_behavior_df = behavior_frametime[reduced_behavior_columns]
    merged_df = pd.merge(refined_ophys_frametime, reduced_behavior_df,
                         on=['trialNum', 'frame_index'], how='inner')
    return merged_df


def assign_pole_moving_frames(merged_df):
    # Assigne pole_moving_up and pole_moving_down to the frames
    # First check if all trials have correct pole up pole moving frames
    # Sometimes there is no pole_moving_frame
    # Just use -1 of the first pole up and +1 of pole up frame as pole_in_frame and pole_out_frame
    
    merged_df = merged_df.query('trial_type != "oo"').reset_index(drop=True)
    assert not merged_df.groupby('trialNum').apply(lambda x: len(np.where(x['pole_up_frame'] == 1)[0]) == 0).any()
    values_to_assign = merged_df.groupby('trialNum').apply(
        lambda x: x['frame_index'] == x['frame_index'].values[np.where(x['pole_up_frame'] == 1)[0][0] - 1]).reset_index(
        drop=True).values
    assert len(values_to_assign) == len(merged_df)
    merged_df['pole_in_frame'] = values_to_assign
    values_to_assign = merged_df.groupby('trialNum').apply(apply_pole_out).reset_index(
            drop=True).values
    assert len(values_to_assign) == len(merged_df)
    merged_df['pole_out_frame'] = values_to_assign

    return merged_df


def apply_pole_out(x):
     '''Apply pole_out_frame to the merged_df
     It is the last frame of pole_up_frame.
     It should be when pole out sound cue is on.
     If it stayed up till the end, then it is ambiguous, so don't apply it.
     '''
     if np.where(x['pole_up_frame']==True)[0][-1] < len(x)-1:
          return (x['frame_index'] == x['frame_index'].values[np.where(x['pole_up_frame']==1)[0][-1]]).reset_index(drop=True)
     else:
          return pd.Series([False]*len(x))


def assign_touch_response_frames(merged_df, post_touch_frames=1):
    # get touch response frames
    # add before_answer_touch_frame and after_answer_touch_frame
    touch_response_frames = merged_df.groupby('trialNum').apply(lambda x: _get_touch_response_frames(x,post_touch_frames=post_touch_frames))
    touch_response_frames = np.concatenate(touch_response_frames.values)
    values_to_assign = merged_df['frame_index'].isin(touch_response_frames).values
    assert len(values_to_assign) == len(merged_df)
    merged_df['touch_response_frame'] = values_to_assign
    merged_df['before_answer_touch_frame'] = False
    merged_df['before_answer_touch_count'] = np.nan
    merged_df = merged_df.groupby('trialNum').apply(_get_before_answer_touch_frames)    
    merged_df['after_answer_touch_frame'] = False
    merged_df['after_answer_touch_count'] = np.nan
    merged_df = merged_df.groupby('trialNum').apply(_get_after_answer_touch_frames)
    return merged_df


def _get_touch_response_frames(x, post_touch_frames=1):
    # touch response frames = touch frames + 0,1,2 frames
    # per trial, to prevent touch rollover when pole was up till the end of the trial
    touch_frame_inds = np.where(x['touch_count']>0)[0]    
    touch_response_frame_inds = np.unique((touch_frame_inds[:,None] + np.arange(post_touch_frames+1)).flatten())
    touch_response_frame_inds = touch_response_frame_inds[touch_response_frame_inds < len(x)]
    touch_response_frames = x['frame_index'].values[touch_response_frame_inds]
    return touch_response_frames


def _get_before_answer_touch_frames(x):    
    if np.where(x['answer_lick_frame'])[0].size == 0:
        return x
    else:
        answer_lick_frame_ind = np.where(x['answer_lick_frame'])[0][0]
        x.iloc[:answer_lick_frame_ind]['before_answer_touch_frame'] = x.iloc[:answer_lick_frame_ind]['touch_response_frame']
        x.iloc[:answer_lick_frame_ind]['before_answer_touch_count'] = x.iloc[:answer_lick_frame_ind]['touch_count']
    return x


def _get_after_answer_touch_frames(x):
    if np.where(x['answer_lick_frame'])[0].size == 0:
        return x
    else:
        answer_lick_frame_ind = np.where(x['answer_lick_frame'])[0][0]        
        x.iloc[answer_lick_frame_ind+1:]['after_answer_touch_frame'] = x.iloc[answer_lick_frame_ind+1:]['touch_response_frame']
        x.iloc[answer_lick_frame_ind+1:]['after_answer_touch_count'] = x.iloc[answer_lick_frame_ind+1:]['touch_count']
    return x


# get normalized spikes
def get_normalized_spikes(roi_dir, ophys_frametime, merged_df, spk_norm='std'):
    spks = np.load(roi_dir / 'spks_reduced.npy')
    iscell = np.load(roi_dir / 'iscell.npy')
    cell_inds = np.where(iscell[:,0]==1)[0]
    spks = spks[cell_inds,:]
    assert spks.shape[1] == len(ophys_frametime)
    # deal with mismatched length
    if len(ophys_frametime) != len(merged_df):    
        removed_inds = np.where(ophys_frametime.frame_index.isin(merged_df.frame_index) == False)[0]
        # removed_tns = ophys_frametime.iloc[removed_inds].trialNum.unique()
        # print(f'JK{mouse:03} S{session:02} plane {plane} ophys_frametime and merged_df length mismatch:')
        # print(f'{len(removed_inds)} frames, {len(removed_tns)} trials')    
        spks = np.delete(spks, removed_inds, axis=1)
    assert spks.shape[1] == len(merged_df)

    # normalize spikes
    if spk_norm == 'std':
        norm_spks = (spks - spks.mean(axis=1)[:,np.newaxis]) / spks.std(axis=1)[:,np.newaxis]
    elif spk_norm == 'max':
        norm_spks = spks / spks.max(axis=1)[:,np.newaxis]
    elif spk_norm == 'none':
        norm_spks = spks
    else:
        raise ValueError('spk_norm should be either "std", "max", or "none"')
    norm_spks = xr.DataArray(norm_spks,
                             dims=('cell_index', 'frame_index'),
                             coords={'cell_index': cell_inds, 'frame_index': merged_df.frame_index.values})
    return norm_spks


def assign_baseline_frames(merged_df):
    def _get_baseline_frames(x):
        pole_in_frame_ind = np.where(x['pole_in_frame'])[0][0]
        x.iloc[:pole_in_frame_ind]['baseline_frame'] = True
        return x

    merged_df['baseline_frame'] = False
    merged_df = merged_df.groupby('trialNum').apply(_get_baseline_frames)
    return merged_df