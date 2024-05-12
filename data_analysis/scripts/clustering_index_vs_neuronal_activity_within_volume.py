from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cdist
import socket
hostname = socket.gethostname()

if hostname == 'HNB228-LABPC6':
    base_dir = Path(r'D:\JK\h5')
elif hostname == 'hnb228-jinho':
    base_dir = Path(r'C:\JK')
else:
    raise('Check the hostname.')


results_dir = base_dir / 'results'
pop_res_dir = results_dir / 'pop_responses' / 'touch_before_answer'
touch_dir = results_dir / 'touch_tuning'
glm_dir = results_dir / 'neuron_glm/ridge/touch_combined'

touch_cell_fn = touch_dir / 'glm_vs_ttest_ind_cell_df.pkl'
ind_cell_df = pd.read_pickle(touch_cell_fn)
ind_cell_df['num_touch_glm'] = ind_cell_df.apply(lambda x: len(x['ind_touch_glm']), axis=1)
ind_cell_df['num_tuned_cells'] = ind_cell_df.apply(lambda x: len(np.intersect1d(x['ind_touch_glm'], x['ind_tuned_ttest_before_answer'])), axis=1)



def clustering_index_vs_neuronal_activity(mouse, volume, session,
                                   num_dim=7,
                                   angles=None,
                                   pca_specific=True, # when using subset of angles (e.g., 
                                    # 45 and 135 from test sessions)
                                   num_trials_choose=30, #
                                #    num_cells_choose=75, # 
                                   rand_cell_prop_range = (1/3, 5/6), 
                                   num_repeat=500,
                                   num_shuffle_trials=10000):
    ''' 
    Always balance trials between angles.
    Intended to use with fit cells only.
    rand_cell_prop_range: range of random cell proportion to choose from

    '''
    planes = range(1,5) if volume==1 else range(5,9)

    touch_response_fn = f'JK{mouse:03}_volume{volume}_S{session:02}_ve_0.05_ptf_1.npy'
    touch_response_results = np.load(pop_res_dir / touch_response_fn, allow_pickle=True).item()
    response_xr = touch_response_results['per_touch_response_xr_fit']
    response_df = touch_response_results['per_touch_response_df']

    
    for pi, plane in enumerate(planes):
        glm_fn = glm_dir / f'JK{mouse:03}S{session:02}_plane{plane}_glm_result.nc'
        with xr.open_dataset(glm_fn) as glm_result:
            glm_result.load()
        cell_id_str = [f'p{plane}c{id:04}' for id in glm_result.cell_id.values]
        glm_result = glm_result.assign_coords(cell_id=cell_id_str)
        touch_encoding = glm_result.varexp_drop.sel(dropped_feature='touch')
        if pi==0:
            touch_encoding_all = touch_encoding
        else:
            touch_encoding_all = xr.concat([touch_encoding_all, touch_encoding], dim='cell_id')

    if angles is None:
        angles = np.unique(response_df.pole_angle)
    else:
        exist_inds = np.where(np.isin(angles, np.unique(response_df.pole_angle)))[0]
        angles = angles[exist_inds]
    if pca_specific:
        response_df = response_df.query('pole_angle in @angles')
        angle_tns = response_df.trialNum.values
        response_xr = response_xr.sel(trialNum=angle_tns)
    # Standardize touch response
    tr_norm = (response_xr - response_xr.mean(axis=0)) / response_xr.std(axis=0)

    # Set empty dataframe
    clustering_index_repeats = pd.DataFrame(columns=['clustering_index', 'num_cells', 'mean_touch_encoding',
    'num_tuned_cells', 'mean_tuning_amplitude', 'exp_var'])

    num_cells = tr_norm.cell_id.size

    for ri in range(num_repeat):
        # Rnadom selection of num cells
        rand_prop = np.random.random() * (rand_cell_prop_range[1] - rand_cell_prop_range[0]) + \
                    rand_cell_prop_range[0]
        num_cells_choose = np.round(num_cells * rand_prop).astype(int)

        # randomly choosing cells
        inds = np.random.choice(num_cells, num_cells_choose, replace=False)
        cell_ids = tr_norm.cell_id[inds]
        tr_norm_sub = tr_norm.sel(cell_id=cell_ids)

        # mean touch encoding
        mean_touch_encoding = touch_encoding_all.sel(cell_id=cell_ids).mean().values 

        # num tuned cells
        cid = ind_cell_df.query('mouse==@mouse and volume==@volume and session==@session').apply(lambda x:
                                np.intersect1d(x.ind_touch_glm, x.ind_tuned_ttest_before_answer), axis=1).values[0]
        cell_id_tuned = [f'p{str(id)[0]}c0{str(id)[1:]}' for id in cid]
        cell_id_tuned = np.intersect1d(cell_id_tuned, tr_norm_sub.cell_id)
        num_tuned_cells = len(cell_id_tuned)

        # mean tuning amplitude from tuned cells
        if num_tuned_cells > 0:
            mean_45 = tr_norm_sub.sel(trialNum=response_df.query('pole_angle==45').trialNum.values,
                                    cell_id=cell_id_tuned).mean(dim='trialNum')
            mean_135 = tr_norm_sub.sel(trialNum=response_df.query('pole_angle==135').trialNum.values,
                                        cell_id=cell_id_tuned).mean(dim='trialNum')
            mean_tuning_amplitude = np.abs(mean_45 - mean_135).mean().values
        else:
            mean_tuning_amplitude = np.array([0]) # to make it compatible with .astype(float)

        pca = PCA()
        pca.fit(tr_norm_sub)

        # Exp var
        exp_var = np.cumsum(pca.explained_variance_ratio_)[num_dim-1]

        pc_all_angles = []
        for ai, angle in enumerate(angles):
            angle_tns = response_df[response_df.pole_angle==angle].trialNum.values
            responses_angle = tr_norm_sub.sel(trialNum=angle_tns)
            pc = pca.transform(responses_angle)
            pc_all_angles.append(pc)
        num_groups = len(angles)

        clustering_index_trial = []
        for gi in range(num_groups):
            this_group = pc_all_angles[gi]
            other_group = np.concatenate([pc_all_angles[i] for i in range(num_groups) if i!=gi])
            within_group_dist = squareform(pdist(this_group[:, :num_dim], 'euclidean'))
            between_group_dist = cdist(this_group[:, :num_dim], other_group[:, :num_dim], 'euclidean')
            
            num_within_group = this_group.shape[0]
            num_between_group = other_group.shape[0]
            for i in range(num_shuffle_trials):
                # within_group_inds = np.random.choice(num_within_group-1, num_trials_choose, replace=False)
                # between_group_inds = np.random.choice(num_between_group-1, num_trials_choose, replace=False)
                within_group_inds = np.random.choice(num_within_group, num_trials_choose, replace=False)
                between_group_inds = np.random.choice(num_between_group, num_trials_choose, replace=False)
                within_group_dist_temp = within_group_dist[within_group_inds, :][:, within_group_inds]
                between_group_dist_temp = between_group_dist[within_group_inds, :][:, between_group_inds]
                clustering_index_repeat = []
                for ti in range(num_trials_choose):
                    within_group_mean = within_group_dist_temp[ti,:].sum() / (num_trials_choose-1)
                    between_group_mean = between_group_dist_temp[ti,:].mean()
                    clustering_index_repeat.append((between_group_mean - within_group_mean) / (between_group_mean + within_group_mean))
                clustering_index_trial.append(np.mean(clustering_index_repeat))
        clustering_index_repeats = clustering_index_repeats.append({'clustering_index': np.mean(clustering_index_trial),
                                                                    'num_cells': num_cells_choose,
                                                                    'mean_touch_encoding': mean_touch_encoding,
                                                                    'num_tuned_cells': num_tuned_cells,
                                                                    'mean_tuning_amplitude': mean_tuning_amplitude.astype(float),
                                                                    'exp_var': exp_var}, ignore_index=True)
    clustering_index_repeats.mean_touch_encoding = clustering_index_repeats.mean_touch_encoding.astype(float)
    clustering_index_repeats.mean_tuning_amplitude = clustering_index_repeats.mean_tuning_amplitude.astype(float)
    clustering_index_repeats.num_tuned_cells = clustering_index_repeats.num_tuned_cells.astype(float)
        
    return clustering_index_repeats

def run_ci_vs_na(mouse, volume, session, save_dir):
    save_fn = save_dir / f'JK{mouse:03}_volume{volume}_S{session:02}_clustering_index_vs_neuronal_activity_within_volume.pkl'
    if save_fn.exists():
        print(f'{save_fn} already exists. Skipping...')
        return
    else:
        print(f'Processing {save_fn}...')
        ci_df = clustering_index_vs_neuronal_activity(mouse, volume, session)
        ci_df.to_pickle(save_fn)
        print(f'{save_fn} saved.')


if __name__ == '__main__':
    expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
    use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                    ~expert_mice_df['processing_error'].astype(bool) &
                                    ((expert_mice_df.session_type == 'training') |
                                    (expert_mice_df.session_type.str.contains('test')))]
    use_volume_df = use_mice_df.query('plane in [1, 5]').copy()
    use_volume_df.loc[:, 'volume'] = use_volume_df['plane'].apply(lambda x: 1 if x==1 else 2)
    training_volume_df = use_volume_df.query('session_type == "training"')
    remove_ind = training_volume_df.query('mouse==27 and session=="15"')
    training_volume_df = training_volume_df.drop(remove_ind.index)
    remove_ind = training_volume_df.query('mouse==36 and session=="9"')
    training_volume_df = training_volume_df.drop(remove_ind.index)

    run_df = training_volume_df[['mouse', 'volume', 'session']].drop_duplicates()
    save_dir = results_dir / 'clustering_index_vs_neuronal_activity_within_volume'
    num_processes = max(cpu_count() - 2, 60)

    # # # mouse, volume, session = run_df.iloc[1]
    # mouse = 39
    # volume = 1
    # session = 14
    # run_ci_vs_na(mouse, volume, int(session), save_dir)

    with Pool(processes=num_processes) as pool:
        pool.starmap(run_ci_vs_na, [(mouse, volume, int(session), save_dir) for mouse, volume, session in run_df[['mouse', 'volume', 'session']].values])
    # for i in range(len(run_df)):
    #     mouse, volume, session = run_df.iloc[i]
    #     run_ci_vs_na(mouse, volume, int(session), save_dir)
