import numpy as np
import pandas as pd
import xarray as xr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# look at the volume data
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


def get_touch_response(base_dir, mouse, session, plane,
                       touch_window='before_answer', spk_norm='std', post_touch_frames=1):
    assert touch_window in ['before_answer', 'after_answer', 'all'],\
                'touch_window should be either "before_answer", "after_answer", or "all"'
    plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
    behavior_frametime = pd.read_pickle(plane_dir / f'JK{mouse:03}_S{session:02}_plane{plane}_frame_whisker_behavior.pkl')
    roi_dir = plane_dir / f'{session:03}/plane0/roi'
    ophys_frametime = pd.read_pickle(roi_dir / 'refined_frame_time.pkl')

    # get merged df
    merged_df = get_merged_df(ophys_frametime, behavior_frametime)
    # assign pole_moving_up and pole_moving_down to the frames
    merged_df = assign_pole_moving_frames(merged_df)

    # get spks
    norm_spks = get_normalized_spikes(roi_dir, ophys_frametime, merged_df, spk_norm=spk_norm)
    
    merged_df['spks_frame_ind'] = np.arange(norm_spks.shape[1])

    # get touch response frames
    merged_df = assign_touch_response_frames(merged_df, post_touch_frames=post_touch_frames)

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
                             coords={'cell_index': cell_inds, 'frame_index': merged_df.frame_index.values},
                             attrs={'event_normalization': spk_norm})
    return norm_spks

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


def draw_pca_touch_response(base_dir, mouse, top_plane, session, ax, touch_window='before_answer',
                            spk_norm='std', post_touch_frames=1, pcs=[0,1]):
    assert touch_window in ['before_answer', 'after_answer', 'all'],\
                'touch_window should be either "before_answer", "after_answer", or "all"'
    volume = 1 if top_plane == 1 else 2
    per_touch_response_xr, per_touch_response_df = \
        get_touch_response_volume_xr(base_dir, mouse, top_plane, session, touch_window=touch_window,
                                     spk_norm=spk_norm, post_touch_frames=post_touch_frames)
    responses_all = per_touch_response_xr.values
    pca = PCA()
    pca.fit(responses_all)
    angles = np.unique(per_touch_response_df.pole_angle)
    colors = plt.cm.jet(np.linspace(0,1,len(angles)))
    for ai, angle in enumerate(angles):
        angle_tns = per_touch_response_df[per_touch_response_df.pole_angle==angle].trialNum.values
        responses_angle = per_touch_response_xr.sel(trialNum=angle_tns).values    
        pc = pca.transform(responses_angle)
        ax.scatter(pc[:,pcs[0]], pc[:,pcs[1]], color=colors[ai], label=angle)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    ax.set_title(f'JK{mouse:03} Volume {volume} Session {session:02}')
    ax.set_xlabel(f'PC{pcs[0]+1}')
    ax.set_ylabel(f'PC{pcs[1]+1}')
    return ax

        

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


def get_touch_response_xr_varexp_threshold(base_dir, mouse, top_plane, session, touch_window='before_answer',
                                           spk_norm='std', varexp_threshold=0.05,
                                           post_touch_frames=2):
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


def random_split(inds, num_split=4):
    inds = np.random.choice(inds, len(inds), replace=False)
    split = []
    group_ids = np.arange(len(inds)) % num_split
    for gi in range(num_split):
        split.append(inds[group_ids==gi])
    return split


def stratify_random_split(inds, stratify_class, num_splits=4):
    assert len(inds) == len(stratify_class)
    classes = np.unique(stratify_class)
    num_classes = len(classes)
    ci = 0
    splits = [[] for i in range(num_splits)]
    for ci in range(num_classes):
        class_inds = np.where(stratify_class==classes[ci])[0]
        split_temp = random_split(inds[class_inds], num_split=num_splits)
        for gi in range(num_splits):
            splits[gi] = np.concatenate([splits[gi], split_temp[gi]])
    return splits


def lda_cross_validate(X, y, splits_inds):
    num_splits = len(splits_inds)
    accuracies = []
    for si in range(num_splits):
        test_inds = splits_inds[si]
        train_inds = np.setdiff1d(np.arange(len(y)), test_inds)
        X_train = X[train_inds,:]
        y_train = y[train_inds]
        X_test = X[test_inds.astype(int),:]
        y_test = y[test_inds.astype(int)]
        lda = LDA()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies


def get_lda_accuracies(X, y, num_split=4, num_repeat=100):    
    all_mean_accuracy = []
    for ri in range(num_repeat):
        splits_inds = stratify_random_split(np.arange(len(y)), y, num_splits=num_split)
        accuracy = np.mean(lda_cross_validate(X, y, splits_inds))        
        all_mean_accuracy.append(accuracy)
    return np.mean(all_mean_accuracy)


def get_shuffle_lda_accuracies(X, y, num_split=4, num_shuffle=100):
    splits_inds = stratify_random_split(np.arange(len(y)), y, num_splits=num_split)    
    shuffle_accuracies = []
    for si in range(num_shuffle):
        shuffle_y = np.random.permutation(y)
        shuffle_accuracies.append(lda_cross_validate(X, shuffle_y, splits_inds))
    shuffle_accuracies = np.array(shuffle_accuracies)
    mean_shuffle_accuracy = np.mean([np.mean(sa) for sa in shuffle_accuracies])
    
    return mean_shuffle_accuracy