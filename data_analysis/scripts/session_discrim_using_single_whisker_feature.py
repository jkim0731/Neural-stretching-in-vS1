import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.style.use('default')
import matplotlib.gridspec as gridspec
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


def save_results_and_fig(mouse, volume, training_volume_df, results_dir, save_dir,
                         vmin=0.5, vmax=1.0):
    whisker_feature_names = ['theta_onset', 'phi_onset', 'kappaH_onset', 'kappaV_onset',
    'arcLength_onset', 'touch_count', 'delta_theta', 'delta_phi',
    'delta_kappaH', 'delta_kappaV', 'touch_duration', 'slide_distance']
    title_texts = ['45', '135', 'mean']
    
    fig_save_fn = save_dir / f'JK{mouse:03}_v{volume}_logistic_regression.png'
    result_save_fn = save_dir / f'JK{mouse:03}_v{volume}_logistic_regression.npy'

    sessions = np.sort([int(s) for s in training_volume_df.query('mouse == @mouse and volume == @volume').session.values])
    wf_mean_all = None
    for session in sessions:
        wf_mean_temp = get_whisker_feature_df(mouse, volume, session, results_dir)
        if wf_mean_all is None:
            wf_mean_all = wf_mean_temp
        else:
            wf_mean_all = pd.concat([wf_mean_all, wf_mean_temp])
    
    acc_mat = np.zeros((len(whisker_feature_names), len(sessions), len(sessions), 3))
    angles = [45, 135]
    for i in range(len(sessions)-1):
        for j in range(i+1, len(sessions)):
            session_i = sessions[i]
            session_j = sessions[j]
            comp_df = wf_mean_all.query('session in [@session_i, @session_j]')
            for angle in angles:
                temp_df = comp_df.query('pole_angle == @angle')

                y = temp_df.session.values == session_i
                for wfi in range(len(whisker_feature_names)):
                    whisker_feature = whisker_feature_names[wfi]
                    X = temp_df[whisker_feature].values
                    X = (X - X.mean()) / X.std()

                    acc, _ = logireg_angle.get_logireg_results(X, y)

                    if angle == 45:
                        acc_mat[wfi, i, j, 0] = acc
                    else:
                        acc_mat[wfi, i, j, 1] = acc
    acc_mat[:,:,:,2] = np.mean(acc_mat[:,:,:,:2], axis=3)
    
    fig = plt.figure(figsize=(34, 30))
    outer = gridspec.GridSpec(1,2, wspace=0.2)
    for spi in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(6,3,
        subplot_spec=outer[spi])
        for j in range(6):
            for k in range(3):
                ax = plt.Subplot(fig, inner[j,k])
                ax.imshow(acc_mat[spi*6+j,:,:,k], vmin=vmin, vmax=vmax)
                ax.set_title(f'{whisker_feature_names[spi*6+j]} {title_texts[k]}', fontsize=25)
                fig.add_subplot(ax)

    fig.suptitle(f'JK{mouse:03} volume {volume}\nLogistic Regression', fontsize=30)
    fig.subplots_adjust(top=0.93)

    fig.savefig(fig_save_fn, bbox_inches='tight', pad_inches=0.1)
    np.save(result_save_fn, acc_mat)
    plt.close(fig)


if __name__ == '__main__':
    t0 = time()

    # base_dir = Path(r'E:\TPM\JK\h5')
    base_dir = Path(r'C:\JK')

    results_dir = base_dir / 'results'
    # wf_dir = results_dir / 'touch_whisker_features'
    # b_dir = Path(r'E:\TPM\JK\SoloData')
    save_dir = results_dir / 'whisker_feature_discrim/single_feature_session_discrim'
    save_dir.mkdir(exist_ok=True, parents=True)

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

    # mouse = 25
    # volume = 1
    # save_results_and_fig(mouse, volume, training_volume_df, results_dir, save_dir)
    
    volumes = [1,2]
    # itertool product of mice, volumes
    input_list = list(product(mice, volumes))

    num_processes = cpu_count() - 2
    print(f'Using {num_processes} processes')
    
    with Pool(processes=num_processes) as pool:
        pool.starmap(save_results_and_fig, [(mouse, volume, training_volume_df, results_dir, save_dir)
                                            for mouse, volume in input_list])
    t1 = time()
    print(f'{(t1-t0)/60:.1f} min elapsed')  
    
# 77 min
    