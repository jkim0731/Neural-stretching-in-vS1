from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.decomposition import PCA
import utils.matched_roi as mr
from pathlib import Path
import numpy as np


def calculate_clustering_index(response_xr, response_df,
                               num_dims=np.arange(3,12,2),
                               angles=None,
                               pca_specific=True, # when using subset of angles (e.g., 
                               # 45 and 135 from test sessions)
                               balance_trials=True, # Balance the number of trials
                               # between angles
                               num_trials_choose=30, # Used only when balance_trials==True
                               num_repeat=10000):
    pca = PCA()
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
    response_xr = (response_xr - response_xr.mean(axis=0)) / response_xr.std(axis=0)
        
    pca.fit(response_xr)
    
    pc_all_angles = []
    for ai, angle in enumerate(angles):
        angle_tns = response_df[response_df.pole_angle==angle].trialNum.values
        responses_angle = response_xr.sel(trialNum=angle_tns)
        pc = pca.transform(responses_angle)
        pc_all_angles.append(pc)
    num_groups = len(angles)
    clustering_index_dims = []
    if isinstance(num_dims, int):
        num_dims = [num_dims]
    for num_dim in num_dims:
        clustering_index_trial = []
        for gi in range(num_groups):
            this_group = pc_all_angles[gi]
            other_group = np.concatenate([pc_all_angles[i] for i in range(num_groups) if i!=gi])
            within_group_dist = squareform(pdist(this_group[:, :num_dim], 'euclidean'))
            between_group_dist = cdist(this_group[:, :num_dim], other_group[:, :num_dim], 'euclidean')
            
            if balance_trials:
                num_within_group = this_group.shape[0]
                num_between_group = other_group.shape[0]
                for i in range(num_repeat):
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
            else:
                num_trials = this_group.shape[0]
                for ti in range(num_trials):
                    within_group_mean = within_group_dist[ti,:].sum() / (num_trials-1)
                    between_group_mean = between_group_dist[ti,:].mean()
                    clustering_index_trial.append((between_group_mean - within_group_mean) / (between_group_mean + within_group_mean))
                    
        clustering_index_dims.append(np.mean(clustering_index_trial))
    if len(clustering_index_dims) == 1:
        clustering_index_dims = clustering_index_dims[0]
    return clustering_index_dims