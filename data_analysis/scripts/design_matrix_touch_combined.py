import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_design_dataframe(mouse, plane, session, base_dir,
                          touch_offsets=np.arange(0,3),
                          whisking_offsets=np.arange(-2,5),
                          lick_offsets=np.arange(-2,3),
                          sound_offsets=np.arange(0,5),
                          reward_offsets=np.arange(0,5)):
    plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    behavior_fn = plane_dir / f'JK{mouse:03}_S{session:02}_plane{plane}_frame_whisker_behavior.pkl'
    if not behavior_fn.exists():
        raise FileNotFoundError(f'{behavior_fn} does not exist')
    behavior_frametime = pd.read_pickle(plane_dir / f'JK{mouse:03}_S{session:02}_plane{plane}_frame_whisker_behavior.pkl')
    roi_dir = plane_dir / f'{session:03}/plane0/roi'
    ophys_fn = roi_dir / 'refined_frame_time.pkl'
    if not ophys_fn.exists():
        raise FileNotFoundError(f'{ophys_fn} does not exist')
    ophys_frametime = pd.read_pickle(roi_dir / 'refined_frame_time.pkl')

    # remove those with remove_trial==True (from expert_mice.csv)
    refined_ophys_frametime = ophys_frametime.query('remove_trial==False')
    assert refined_ophys_frametime.remove_frame.values.sum() == 0
    # extend each trial frames by 1 in each direction (those trimmed to make reduced_frame_time.pkl from frame_time.pkl)
    # so that I can have 2 more frame of information (from behavior_frametime)
    extended_ophys_df = refined_ophys_frametime.groupby('trialNum').apply(extend_dataframe).reset_index(drop=True)

    # merge with behavior_frametime
    reduced_behavior_columns = np.setdiff1d(behavior_frametime.columns,
                                            np.setdiff1d(extended_ophys_df.columns,
                                                         ['trialNum', 'frame_index']))
    reduced_behavior_df = behavior_frametime[reduced_behavior_columns]
    merged_df = pd.merge(extended_ophys_df, reduced_behavior_df,
                         on=['trialNum', 'frame_index'], how='inner')

    # remove catch trials
    catch_trial_nums = merged_df.query('trial_type == "oo"')['trialNum'].unique()
    merged_df = merged_df.query('trialNum not in @catch_trial_nums').reset_index().copy()
    assert 'oo' not in merged_df['trial_type'].unique()
    
    # Assign pole in sound and pole out sound cue frames
    assert not merged_df.groupby('trialNum').apply(lambda x: len(np.where(x['pole_up_frame']==1)[0])==0).any()
    merged_df['pole_in_frame'] = merged_df.groupby('trialNum').apply(lambda x: x['frame_index'] == x['frame_index'].values[np.where(x['pole_up_frame']==True)[0][0]-1]).reset_index(drop=True).values    
    merged_df['pole_out_frame'] = merged_df.groupby('trialNum').apply(apply_pole_out).reset_index(drop=True).values

    # Initialize names
    angles = merged_df['pole_angle'].unique()
    angles = angles[~np.isnan(angles)].astype(int)
    lick_names = ['num_lick_left', 'num_lick_right']
    whisking_names = ['num_whisks', 'midpoint', 'amplitude']
    reward_names = ['first_reward_lick_left', 'first_reward_lick_right']
    sound_names = ['pole_in_frame', 'pole_out_frame']
    touch_names = []

    # add specific angle touch
    for angle in angles:
        touch_names.append(f'touch_count_{angle}')
        merged_df[f'touch_count_{angle}'] = np.nan
        merged_df.loc[merged_df['pole_angle']==angle, f'touch_count_{angle}'] = merged_df.loc[merged_df['pole_angle']==angle, 'touch_count'].values
        merged_df[f'touch_count_{angle}'] = merged_df[f'touch_count_{angle}'].apply(lambda x: 0 if np.isnan(x) else x)

    # Build design dataframe
    design_df = merged_df[['trialNum','frame_index']].copy()
    for angle in angles:
        for offset in touch_offsets:
            design_df[f'touch_count_{angle}_{offset}'] = merged_df.groupby('trialNum').apply(lambda x: x[f'touch_count_{angle}'].shift(offset)).reset_index(drop=True).values
    for whisking_name in whisking_names:
        for offset in whisking_offsets:
            design_df[f'{whisking_name}_{offset}'] = merged_df.groupby('trialNum').apply(lambda x: x[whisking_name].shift(offset)).reset_index(drop=True).values
    for lick_name in lick_names:
        for offset in lick_offsets:
            design_df[f'{lick_name}_{offset}'] = merged_df.groupby('trialNum').apply(lambda x: x[lick_name].shift(offset)).reset_index(drop=True).values
    for sound_name in sound_names:
        for offset in sound_offsets:
            design_df[f'{sound_name}_{offset}'] = merged_df.groupby('trialNum').apply(lambda x: x[sound_name].shift(offset)).reset_index(drop=True).values
    for reward_name in reward_names:
        for offset in reward_offsets:
            design_df[f'{reward_name}_{offset}'] = merged_df.groupby('trialNum').apply(lambda x: x[reward_name].shift(offset)).reset_index(drop=True).values

    return design_df, merged_df


def extend_dataframe(group, n_before=1, n_after=1):
    before_rows = group.iloc[0:n_before].copy().reset_index(drop=True)
    before_rows[:] = np.nan
    before_rows.trialNum = group.trialNum.iloc[0]
    before_rows.frame_index = -1
    before_rows.loc[before_rows.index.max(), 'frame_index'] = group.frame_index.min()-1
    after_rows = group.iloc[-n_after:].copy().reset_index(drop=True)
    after_rows[:] = np.nan
    after_rows.trialNum = group.trialNum.iloc[-1]
    after_rows.frame_index = -1
    after_rows.loc[0,'frame_index'] = group.frame_index.max()+1
    extended_group = pd.concat([before_rows, group, after_rows], ignore_index=True)
    return extended_group


def apply_pole_out(x):
     if np.where(x['pole_up_frame']==True)[0][-1] < len(x)-1:
          return x['frame_index'] == x['frame_index'].values[np.where(x['pole_up_frame']==True)[0][-1]]
     else:
          return pd.Series([False]*len(x))
     

### QC
     
def qc_design_touch(design_df, merged_df, trial_num, angles, touch_offsets):
    temp_merged_df = merged_df.loc[merged_df['trialNum']==trial_num]
    temp_design_df = design_df.loc[design_df['trialNum']==trial_num]
    fig, ax = plt.subplots(figsize=(15,5))

    x_inds = np.where(temp_design_df.frame_index>=0)[0]
    yoffset = -0.1
    for angle in angles:
        for offset in touch_offsets:
            ax.plot(temp_design_df['frame_index'].values[x_inds],
                    temp_design_df[f'touch_count_{angle}_{offset}'].values[x_inds] + yoffset,
                    label=f'touch_count_{angle}_{offset}')
            yoffset += 0.03

    trial_angle = temp_merged_df.pole_angle.unique()
    trial_angle = trial_angle[~np.isnan(trial_angle)].astype(int)
    angle_ind = np.where(angles == trial_angle)[0][0]
    y_offset = -0.1 + angle_ind*0.03*len(touch_offsets)
    ax.plot(temp_merged_df['frame_index'].values[x_inds],
            temp_merged_df['touch_count'].values[x_inds] + y_offset,  '.',
            label='touch_count', color='k')
    ax.legend()

    return fig, ax

def qc_design_lick(design_df, merged_df, trial_num, lick_names, lick_offsets):
    temp_merged_df = merged_df.loc[merged_df['trialNum']==trial_num]
    temp_design_df = design_df.loc[design_df['trialNum']==trial_num]
    fig, ax = plt.subplots(figsize=(15,5))
    x_inds = np.where(temp_design_df.frame_index>=0)[0]
    yoffset = -0.1
    for lick_name in lick_names:
        for offset in lick_offsets:
            ax.plot(temp_design_df['frame_index'].values[x_inds],
                    temp_design_df[f'{lick_name}_{offset}'].values[x_inds] + yoffset,
                    label=f'{lick_name}_{offset}')
            yoffset += 0.03

    ax.plot(temp_merged_df['frame_index'].values[x_inds],
            temp_merged_df['num_lick_left'].values[x_inds] -0.04,  'k.', label='num_lick_left')
    ax.plot(temp_merged_df['frame_index'].values[x_inds],
            temp_merged_df['num_lick_right'].values[x_inds] + 0.11,  'b.', label='num_lick_right')

    ax.legend()
    return fig, ax

def qc_design_sound(design_df, merged_df, trial_num, sound_names, sound_offsets):
    temp_merged_df = merged_df.loc[merged_df['trialNum']==trial_num]
    temp_design_df = design_df.loc[design_df['trialNum']==trial_num]
    fig, ax = plt.subplots(figsize=(15,5))
    x_inds = np.where(temp_design_df.frame_index>=0)[0]
    yoffset = -0.1
    for sound_name in sound_names:
        for offset in sound_offsets:
            ax.plot(temp_design_df['frame_index'].values[x_inds],
                    temp_design_df[f'{sound_name}_{offset}'].values[x_inds] + yoffset,
                    label=f'{sound_name}_{offset}')
            yoffset += 0.03

    ax.plot(temp_merged_df['frame_index'].values[x_inds],
            temp_merged_df['pole_in_frame'].values[x_inds] - 0.1 - sound_offsets[0]*0.03,
            'k.', label='pole_in_frame')
    ax.plot(temp_merged_df['frame_index'].values[x_inds],
            temp_merged_df['pole_out_frame'].values[x_inds] - 0.1 - (sound_offsets[0]-len(sound_offsets))*0.03,
            'b.', label='pole_out_frame')

    ax.legend()
    return fig, ax
