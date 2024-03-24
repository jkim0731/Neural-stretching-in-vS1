from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.decomposition import PCA
import utils.matched_roi as mr
from pathlib import Path
import numpy as np


def calculate_clustering_index(response_xr, response_df,
                               num_dims=np.arange(3,12,2),
                               angles=None,
                               pca_specific=False,
                               balance_angles=True,
                               num_repeat=100):
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
            num_trials = this_group.shape[0]
            for ti in range(num_trials):
                within_group_mean = within_group_dist[ti,:].sum() / (num_trials-1)
                between_group_mean = between_group_dist[ti,:].mean()
                clustering_index_trial.append((between_group_mean - within_group_mean) / (between_group_mean + within_group_mean))
        clustering_index_dims.append(np.mean(clustering_index_trial))
    return clustering_index_dims