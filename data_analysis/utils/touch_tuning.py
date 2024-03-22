import numpy as np
import xarray as xr
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from merged_df_annotation import assign_touch_response_frames, assign_baseline_frames


def get_touch_response_from_baseline(merged_df, norm_spks,
                                     touch_window='before_answer'):
    if 'touch_response_frame' not in merged_df.columns:
        merged_df = assign_touch_response_frames(merged_df)
    if 'baseline_frame' not in merged_df.columns:
        merged_df = assign_baseline_frames(merged_df)
    touch_trial_nums = merged_df[merged_df.touch_response_frame].trialNum.unique()
    touch_merged_df = merged_df[merged_df.trialNum.isin(touch_trial_nums)]
    touch_response_from_baseline = []
    for tn in touch_trial_nums:
        trial_df = touch_merged_df.query('trialNum == @tn')
        if touch_window == 'before_answer':
            touch_response_frame_inds = trial_df.query('before_answer_touch_frame').frame_index.values
        elif touch_window == 'after_answer':
            touch_response_frame_inds = trial_df.query('after_answer_touch_frame').frame_index.values
        elif touch_window == 'all':
            touch_response_frame_inds = trial_df.query('touch_response_frame').frame_index.values
        else:
            raise ValueError('touch_window should be either "before_answer", "after_answer", or "all"')
        baseline_frame_inds = trial_df.query('baseline_frame').frame_index.values
        touch_response_spks = norm_spks.sel(frame_index=touch_response_frame_inds).sum(dim='frame_index')
        baseline_spks = norm_spks.sel(frame_index=baseline_frame_inds).sum(dim='frame_index')

        temp_r_f_b = touch_response_spks - baseline_spks
        touch_response_from_baseline.append(temp_r_f_b.values)
    touch_response_from_baseline = np.stack(touch_response_from_baseline)
    touch_response_from_baseline = xr.DataArray(touch_response_from_baseline,
                                                dims=('trialNum', 'cell_index'),
                                                coords={'trialNum': touch_trial_nums.astype(int),
                                                        'cell_index': norm_spks.cell_index.values},
                                                attrs={'touch_window': touch_window})

    return touch_response_from_baseline


def get_touch_cell_inds(touch_response_xr, merged_df, pval_threshold=0.01):
    angles = np.sort(merged_df.pole_angle.unique())
    touch_trial_nums = merged_df.query('touch_response_frame').trialNum.unique()
    ttest_results = {}
    for cell_index in touch_response_xr.cell_index.values:
        cell_data = touch_response_xr.sel(cell_index=cell_index)
        ttest_results[cell_index] = {}
        for angle in angles:
            trials = merged_df.query('pole_angle == @angle').trialNum.values
            trials = np.intersect1d(trials, touch_trial_nums)
            cell_trials = cell_data.sel(trialNum=trials).values
            ttest_results[cell_index][angle] = ttest_ind(cell_trials, np.zeros_like(cell_trials))
    
    touch_cell_inds = []
    for cell_index, angles_data in ttest_results.items():
        for angle, ttest_result in angles_data.items():
            if ttest_result.pvalue < pval_threshold:
                touch_cell_inds.append(cell_index)
                break

    touch_cell_inds = np.array(touch_cell_inds)
    return touch_cell_inds, ttest_results


def get_tuned_cell_inds(touch_response_xr, merged_df, touch_cell_inds, pval_threshold=0.01):
    touch_trial_nums = merged_df.query('touch_response_frame').trialNum.unique()
    angles = np.sort(merged_df.pole_angle.unique())
    if len(angles) > 2:
        # Perform ANOVA
        avova_results = {}
        angle_trials = []
        for angle in angles:
            trials = merged_df.query('pole_angle == @angle').trialNum.values
            trials = np.intersect1d(trials, touch_trial_nums)
            angle_trials.append(trials)
        for cell_index in touch_cell_inds:
            cell_data = touch_response_xr.sel(cell_index=cell_index)
            avova_results[cell_index] = f_oneway(*[cell_data.sel(trialNum=trials).values for trials in angle_trials])

        # Perform Tukey's HSD test
        tukey_results = {}
        trial_angles = [merged_df.query('trialNum == @tn').pole_angle.values[0] for tn in touch_trial_nums]
        assert len(trial_angles) == touch_response_xr.sizes['trialNum']
        tuned_cell_inds = []
        for cell_index in touch_cell_inds:
            if avova_results[cell_index].pvalue < pval_threshold:
                tukey_results[cell_index] = pairwise_tukeyhsd(touch_response_xr.sel(cell_index=cell_index),
                                                            trial_angles, alpha=0.05)
                if tukey_results[cell_index].reject.any():
                    tuned_cell_inds.append(cell_index)
        tuned_cell_inds = np.array(tuned_cell_inds)
        stat_results = {'anova': avova_results, 'tukey': tukey_results}

    elif len(angles) == 2:
        # Perform t-test
        ttest_results = {}
        angle_trials = []
        for angle in angles:
            trials = merged_df.query('pole_angle == @angle').trialNum.values
            trials = np.intersect1d(trials, touch_trial_nums)
            angle_trials.append(trials)
        tuned_cell_inds = []
        for cell_index in touch_cell_inds:
            cell_data = touch_response_xr.sel(cell_index=cell_index)
            ttest_results[cell_index] = ttest_ind(*[cell_data.sel(trialNum=trials).values for trials in angle_trials])
            if ttest_results[cell_index].pvalue < pval_threshold:
                tuned_cell_inds.append(cell_index)
        tuned_cell_inds = np.array(tuned_cell_inds)
        stat_results = {'ttest': ttest_results}
    else:
        raise ValueError('angles should have at least 2 unique values')

    return tuned_cell_inds, stat_results
