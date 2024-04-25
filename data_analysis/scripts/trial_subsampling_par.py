import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append(r'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Analysis\codes\data_analysis')

import utils.logistic_regression_angle as logireg_angle
import pickle
from multiprocessing import Pool, cpu_count
from time import time


def get_whisker_feature_df(mouse, volume, session, results_dir, touch_window='before_answer'):
    popres_dir = results_dir / 'pop_responses/touch_before_answer'
    popres_fn = popres_dir / f'JK{mouse:03}_volume{volume}_S{session:02}_ve_0.05_ptf_1.npy'
    popres = np.load(popres_fn, allow_pickle=True).item()
    b_df = popres['per_touch_response_df']

    whisker_feature_dir = results_dir / 'touch_whisker_features'
    wf_fn = whisker_feature_dir / f'JK{mouse:03}S{session:02}_touch_whisker_features.pkl'
    wf_df = pd.read_pickle(wf_fn)

    if touch_window == 'before_answer':
        wf_df = wf_df.groupby('trialNum').apply(
            lambda x: x.query('touch_offset_time < answer_lick_time')).reset_index(
                drop=True)
    elif touch_window == 'after_answer':
        wf_df = wf_df.groupby('trialNum').apply(
            lambda x: x.query('pole_onset_time >= answer_lick_time')).reset_index(
                drop=True)
    elif touch_window == 'all':
        pass
    else:
        raise ValueError('Invalid touch_window')
    
    wf_mean = wf_df.groupby('trialNum').mean()
    wf_mean['touch_count'] = wf_df.groupby('trialNum').size()

    wf_mean = wf_mean.merge(b_df[['trialNum','correct', 'miss', 'miss', 'pole_angle']],
                            on='trialNum').reset_index(drop=True)
    wf_mean = wf_mean.dropna()
    wf_mean['mouse'] = mouse
    wf_mean['session'] = session

    return wf_mean


def get_best_matching_pairs_inds(between_session_dist, coords, min_num_pairs=15,
                                 threshold=0.55, scaling_factor=50):
    num_trials = between_session_dist.shape
    row_ind, col_ind = linear_sum_assignment(between_session_dist)
    pair_distances = between_session_dist[row_ind, col_ind]
    sort_ind = np.argsort(pair_distances)
    
    lower_than_threshold_num_pairs = [0]
    higher_than_threshold_num_pairs = [min(num_trials)+1]

    num_pairs_tested = []
    acc_tested = []
    num_pairs = min(30, min(num_trials))
    while (num_pairs not in higher_than_threshold_num_pairs) and \
        (num_pairs not in lower_than_threshold_num_pairs) and \
        (num_pairs >= min_num_pairs) and \
        (num_pairs <= min(num_trials)):
        temp_row_ind = row_ind[sort_ind[:num_pairs]]
        temp_col_ind = col_ind[sort_ind[:num_pairs]]
        temp_x = np.vstack([coords[temp_row_ind, :], coords[temp_col_ind + num_trials[0], :]])
        temp_y = np.concatenate([np.zeros(num_pairs), np.ones(num_pairs)])
        min_num_trials_in_split = np.min(min_num_pairs, 15)
        temp_acc, _ = logireg_angle.get_logireg_results(temp_x, temp_y,
                                                        min_num_trials_in_split=min_num_trials_in_split)
        # print(num_pairs, temp_acc)
        num_pairs_tested.append(num_pairs)
        acc_tested.append(temp_acc)
        if temp_acc < threshold:
            lower_than_threshold_num_pairs.append(num_pairs)
            num_pairs = num_pairs + int((threshold - temp_acc) * scaling_factor)
            if num_pairs >= np.min(higher_than_threshold_num_pairs):
                num_pairs = np.min(higher_than_threshold_num_pairs) - 1
            elif num_pairs <= np.max(lower_than_threshold_num_pairs):
                num_pairs = np.max(lower_than_threshold_num_pairs) + 1
        else:
            higher_than_threshold_num_pairs.append(num_pairs)
            num_pairs = num_pairs - int((temp_acc - threshold) * scaling_factor)
            if num_pairs <= np.max(lower_than_threshold_num_pairs):
                num_pairs = np.max(lower_than_threshold_num_pairs) + 1
            elif num_pairs >= np.min(higher_than_threshold_num_pairs):
                num_pairs = np.min(higher_than_threshold_num_pairs) - 1
    if num_pairs < min_num_pairs:
        return None, None, None, None
    assert np.min(higher_than_threshold_num_pairs) == np.max(lower_than_threshold_num_pairs) + 1
    max_num_trials = np.max(lower_than_threshold_num_pairs)
    max_test_ind = np.where(np.array(num_pairs_tested) == max_num_trials)[0][0]
    accuracy = acc_tested[max_test_ind]
    choose_row_ind = row_ind[sort_ind[:max_num_trials]]
    choose_col_ind = col_ind[sort_ind[:max_num_trials]]

    return choose_row_ind, choose_col_ind, max_num_trials, accuracy


def get_matching_trial_nums_between_sessions(comp_sessions, wf_mean_all):
    during_feature_names = ['touch_count', 'delta_theta', 'delta_phi',
        'delta_kappaH', 'delta_kappaV', 'touch_duration', 'slide_distance']
    angles = [45, 135]

    comp_wf = wf_mean_all.query('session in @comp_sessions')
    trialNum_session_0 = []
    trialNum_session_1 = []
    for angle in angles:
        temp_df = comp_wf.query('pole_angle == @angle')
        std_df = temp_df.copy()
        for feature in during_feature_names:
            std_df[feature] = (temp_df[feature] - temp_df[feature].mean()) / temp_df[feature].std()
        coords = std_df[during_feature_names].values
        pairwise_dist = squareform(pdist(coords))
        num_trials = std_df.groupby('session').size().values
        between_session_dist = pairwise_dist[:num_trials[0], num_trials[0]:]
        # mean_within_dist = (pairwise_dist[:num_trials[0], :num_trials[0]][np.triu_indices(num_trials[0], k=1)].mean() +
        #                     pairwise_dist[num_trials[0]:, num_trials[0]:][np.triu_indices(num_trials[1], k=1)].mean()) / 2
        choose_row_ind, choose_col_ind, max_num_trials, accuracy = get_best_matching_pairs_inds(between_session_dist, coords)
        if choose_row_ind is None:
            return None

        trialNum_session_0_all = std_df.query('session == @comp_sessions[0]').trialNum.values
        assert (np.sort(trialNum_session_0_all) == trialNum_session_0_all).all()
        trialNum_session_0.append([trialNum_session_0_all[i] for i in choose_row_ind])
        trialNum_session_1_all = std_df.query('session == @comp_sessions[1]').trialNum.values
        assert (np.sort(trialNum_session_1_all) == trialNum_session_1_all).all()
        trialNum_session_1.append([trialNum_session_1_all[i] for i in choose_col_ind])

    assert [len(t) for t in trialNum_session_0] == [len(t) for t in trialNum_session_1]
    assert (len(np.intersect1d(trialNum_session_0[0], trialNum_session_0[1]))) == 0

    balanced_num_trials = min([len(t) for t in trialNum_session_0])
    trialNum_session_0 = [t[:balanced_num_trials] for t in trialNum_session_0]
    trialNum_session_1 = [t[:balanced_num_trials] for t in trialNum_session_1]
    trialNum_pair_dict = {f'{comp_sessions[0]}': np.concatenate(trialNum_session_0),
                          f'{comp_sessions[1]}': np.concatenate(trialNum_session_1)}
    return trialNum_pair_dict


def save_matching_trialNum_dict(mouse, volume, training_volume_df, results_dir):
    print(f'Processing JK{mouse:03} volume {volume}')
    t0 = time()
    save_dir = results_dir / 'trial_subsampling'
    save_dir.mkdir(exist_ok=True, parents=True)
    save_fn = save_dir / f'JK{mouse:03}_volume{volume}_matching_trialNum_dict.pkl'

    sessions = np.sort([int(s) for s in training_volume_df.query('mouse == @mouse and volume == @volume').session.values])
    wf_mean_all = None
    for session in sessions:
        wf_mean_temp = get_whisker_feature_df(mouse, volume, session, results_dir)
        if wf_mean_all is None:
            wf_mean_all = wf_mean_temp
        else:
            wf_mean_all = pd.concat([wf_mean_all, wf_mean_temp])

    matching_trialNum_dict = {}
    print(len(sessions))
    for i in range(len(sessions)-1):
        for j in range(i+1, len(sessions)):
            print(f'Processing {i} and {j}')
            comp_sessions = [sessions[i], sessions[j]]
            trialNum_pair_dict = get_matching_trial_nums_between_sessions(comp_sessions, wf_mean_all)
            matching_trialNum_dict[f'{comp_sessions[0]}_{comp_sessions[1]}'] = trialNum_pair_dict

    with open(save_fn, 'wb') as f:
        pickle.dump(matching_trialNum_dict, f)

    t1 = time()
    print(f'Saved JK{mouse:03} volume {volume} in {(t1-t0)/60:.2f} minutes.')


if __name__ == '__main__':
    t0 = time()
    base_dir = Path(r'C:\JK')

    results_dir = base_dir / 'results'
    expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
    use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                    ~expert_mice_df['processing_error'].astype(bool) &
                                    ((expert_mice_df.session_type == 'training') |
                                    (expert_mice_df.session_type.str.contains('test')))]
    use_volume_df = use_mice_df.query('plane in [1, 5]')
    use_volume_df.loc[:, 'volume'] = use_volume_df['plane'].apply(lambda x: 1 if x==1 else 2)
    training_volume_df = use_volume_df.query('session_type == "training"')
    remove_ind = training_volume_df.query('mouse==27 and session=="15"')
    training_volume_df = training_volume_df.drop(remove_ind.index)
    remove_ind = training_volume_df.query('mouse==36 and session=="9"')
    training_volume_df = training_volume_df.drop(remove_ind.index)
    
    # mice = training_volume_df.mouse.unique()
    # volumes = training_volume_df.volume.unique()

    # with Pool(12) as p:
    #     p.starmap(save_matching_trialNum_dict, [(mouse, volume, training_volume_df, results_dir) for mouse in mice for volume in volumes])

    mouse = 39
    volume = 2
    save_matching_trialNum_dict(mouse, volume, training_volume_df, results_dir)
    t1 = time()
    print(f'{(t1-t0)/60:.2f} minutes have passed')
