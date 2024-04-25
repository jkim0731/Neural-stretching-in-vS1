import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.style.use('default')
import sys
sys.path.append(r'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Analysis\codes\data_analysis')
import utils.logistic_regression_angle as logireg_angle
from multiprocessing import Pool, cpu_count
from itertools import product
from time import time


def get_whisker_feature_df(mouse, volume, session, results_dir, touch_window='before_answer'):
    # just to get behavior data
    popres_dir = results_dir / 'pop_responses/touch_before_answer' 
    popres_fn = popres_dir / f'JK{mouse:03}_volume{volume}_S{session:02}_ve_0.05_ptf_1.npy'
    popres = np.load(popres_fn, allow_pickle=True).item()
    b_df = popres['per_touch_response_df']

    # Get whisker features
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
    wf_mean.loc[:, 'touch_count'] = wf_df.groupby('trialNum').size()

    wf_mean = wf_mean.merge(b_df[['trialNum','correct', 'miss', 'miss', 'pole_angle']],
                            on='trialNum').reset_index(drop=True)
    wf_mean = wf_mean.dropna()
    wf_mean.loc[:, 'mouse'] = mouse
    wf_mean.loc[:, 'session'] = session

    return wf_mean


def get_logireg_session_classfy(mouse, volume, sessions, results_dir,
                                penalty='l2', calc_shuffle=False):
    wf_mean_all = None
    for session in sessions:
        wf_mean_temp = get_whisker_feature_df(mouse, volume, session, results_dir)
        if wf_mean_all is None:
            wf_mean_all = wf_mean_temp
        else:
            wf_mean_all = pd.concat([wf_mean_all, wf_mean_temp])
    
    whisker_feature_names = ['touch_count', 'delta_theta', 'delta_phi',
                             'delta_kappaH', 'delta_kappaV', 'touch_duration', 'slide_distance']

    logireg_acc_45 = np.zeros((len(sessions), len(sessions)))
    logireg_coeff_45 = np.zeros((len(sessions), len(sessions), len(whisker_feature_names)+1))
    logireg_acc_135 = np.zeros((len(sessions), len(sessions)))
    logireg_coeff_135 = np.zeros((len(sessions), len(sessions), len(whisker_feature_names)+1))
    if calc_shuffle:
        logireg_acc_shuffle_45 = np.zeros((len(sessions), len(sessions)))
        logireg_coeff_shuffle_45 = np.zeros((len(sessions), len(sessions), len(whisker_feature_names)+1))
        logireg_acc_shuffle_135 = np.zeros((len(sessions), len(sessions)))
        logireg_coeff_shuffle_135 = np.zeros((len(sessions), len(sessions), len(whisker_feature_names)+1))
    
    angles = [45, 135]

    for i in range(len(sessions)-1):
        for j in range(i+1, len(sessions)):
            session_i = sessions[i]
            session_j = sessions[j]

            comp_df = wf_mean_all.query('session in [@session_i, @session_j]')
            for angle in angles:
                temp_df = comp_df.query('pole_angle == @angle')

                y = temp_df.session.values == session_i
                X = temp_df[whisker_feature_names].values

                # Standardize X
                X = ((X - X.mean(axis=0)) / X.std(axis=0))

                logireg_acc, logireg_coef = logireg_angle.get_logireg_results(X, y, penalty=penalty)
                if calc_shuffle:
                    logireg_acc_shuffle, logireg_coef_shuffle = logireg_angle.get_shuffle_logireg_results(X, y, penalty=penalty)

                if angle == 45:
                    logireg_acc_45[i, j] = logireg_acc
                    logireg_coeff_45[i,j,:] = logireg_coef
                    if calc_shuffle:
                        logireg_acc_shuffle_45[i, j] = logireg_acc_shuffle
                        logireg_coeff_shuffle_45[i,j,:] = logireg_coef_shuffle
                else:
                    logireg_acc_135[i, j] = logireg_acc
                    logireg_coeff_135[i,j,:] = logireg_coef
                    if calc_shuffle:
                        logireg_acc_shuffle_135[i, j] = logireg_acc_shuffle
                        logireg_coeff_shuffle_135[i,j,:] = logireg_coef_shuffle
    if calc_shuffle:
        return logireg_acc_45, logireg_acc_135, logireg_acc_shuffle_45, logireg_acc_shuffle_135, \
            logireg_coeff_45, logireg_coeff_135, logireg_coeff_shuffle_45, logireg_coeff_shuffle_135
    else:
        return logireg_acc_45, logireg_acc_135, logireg_coeff_45, logireg_coeff_135


def draw_logireg_acc_fig(mouse, volume, sessions, logireg_acc_45, logireg_acc_135,
                         suptitle_text=None):
    logireg_mat_mean = np.mean([logireg_acc_45, logireg_acc_135], axis=0)
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(logireg_acc_45, vmin=0.5, vmax=1)
    ax[0].set_title('Angle = 45')
    ax[1].imshow(logireg_acc_135, vmin=0.5, vmax=1)
    ax[1].set_title('Angle = 135')
    im = ax[2].imshow(logireg_mat_mean, vmin=0.5, vmax=1)
    ax[2].set_title('Mean of 45 and 135 degrees')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)

    plt.colorbar(im, cax=cax, label='LDA performance')
    for i in range(3):
        ax[i].set_xticks(range(len(sessions)))
        ax[i].set_xticklabels(sessions)
        ax[i].set_xlabel('Session')
        ax[i].set_yticks(range(len(sessions)))
        ax[i].set_yticklabels(sessions)
    ax[0].set_ylabel('Session')

    if suptitle_text is None:
        suptitle_text = f'JK{mouse:03} volume{volume}\nSession classification using "during touch" whisker features\nLogistic regression'
    fig.suptitle(suptitle_text, fontsize=13)
    fig.subplots_adjust(top=0.80)
    return fig, ax


def save_results_and_fig(mouse, volume, training_volume_df, results_dir, penalty):
    sessions = np.sort([int(s) for s in training_volume_df.query('mouse == @mouse and volume == @volume').session.values])
    acc_45, acc_135, coeff_45, coeff_135 = get_logireg_session_classfy(mouse, volume, sessions, results_dir,
                                                                       penalty=penalty, calc_shuffle=False)
    fig, ax = draw_logireg_acc_fig(mouse, volume, sessions, acc_45, acc_135)

    logireg_results = {'accuracy_45': acc_45,
                       'accuracy_135': acc_135,
                       'coeff_45': coeff_45,
                       'coeff_135': coeff_135}
    save_dir = results_dir / 'during_touch_whisker_feature_discrim/session_discrim'
    save_dir.mkdir(parents=True, exist_ok=True)
    logireg_results_fn = save_dir / f'JK{mouse:03}_volume{volume}_session_discrim_logireg_results_{penalty}_during_touch.npy'
    np.save(logireg_results_fn, logireg_results)

    fig_fn = save_dir / f'JK{mouse:03}_volume{volume}_session_discrim_logireg_accuracy_{penalty}_during_touch.png'
    fig.savefig(fig_fn, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    
    t0 = time()

    # base_dir = Path(r'E:\TPM\JK\h5')
    base_dir = Path(r'C:\JK')

    results_dir = base_dir / 'results'
    # wf_dir = results_dir / 'touch_whisker_features'
    # b_dir = Path(r'E:\TPM\JK\SoloData')

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

    mice = [25,27,30,36,39,52]
    test_sessions = [[4,19], [3,8], [3,21], [1,17], [1,23], [3,21]]
    naive_sessions = [10, 4, 11, 6, 6, 11]
    
    penalties = ['l1', 'l2']
    volumes = [1,2]
    # itertool product of mice, volumes, and penalties
    input_list = list(product(mice, volumes, penalties))

    num_processes = cpu_count() - 2
    print(f'Using {num_processes} processes')
    with Pool(processes=num_processes) as pool:
        pool.starmap(save_results_and_fig, [(mouse, volume, training_volume_df, results_dir, penalty)
                                            for mouse, volume, penalty in input_list])
    t1 = time()
    print(f'{(t1-t0)/60:.1f} min elapsed')  
    
# 13.2 min