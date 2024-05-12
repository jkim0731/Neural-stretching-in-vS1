import numpy as np
import pandas as pd
from pathlib import Path
import utils.clustering_index as ci
import socket
host_name = socket.gethostname()
if host_name == 'HNB228-LABPC6':
    base_dir = Path(r'D:\JK\h5')
elif host_name == 'hnb228-jinho':
    # base_dir = Path(r'E:\TPM\JK\h5')
    base_dir = Path(r'C:\JK')
else:
    raise('Incorrect computer')
results_dir = base_dir / 'results'
roi_matching_dir = results_dir / 'roi_matching'
pop_res_dir = results_dir / 'pop_responses' / 'touch_before_answer'


def get_ci_from_matched_rois(mouse, volume, session_nums_comp,
                             num_dims=7, num_repeat=500):
    # roi_matching_dir
    # pop_res_dir

    roi_mapping_fn = roi_matching_dir / 'roi_mapping.pkl'
    roi_mapping_df = pd.read_pickle(roi_mapping_fn)

    # First, get matched ROIs
    for si, session in enumerate(session_nums_comp):
        touch_response_fn = f'JK{mouse:03}_volume{volume}_S{session:02}_ve_0.05_ptf_1.npy'
        touch_response_results = np.load(pop_res_dir / touch_response_fn, allow_pickle=True).item()
        response_xr = touch_response_results['per_touch_response_xr']

        session_ids = response_xr.cell_id.values
        session_mr_ids = roi_mapping_df.query('session_roi_id in @session_ids').master_roi_id.values
        if si == 0:
            matched_mr_ids = session_mr_ids
        else:
            matched_mr_ids = np.intersect1d(matched_mr_ids, session_mr_ids)

        response_xr_fit = touch_response_results['per_touch_response_xr_fit']
        session_ids_fit = response_xr_fit.cell_id.values
        session_mr_ids_fit = roi_mapping_df.query('session_roi_id in @session_ids_fit').master_roi_id.values
        if si == 0:
            matched_mr_ids_fit_both = session_mr_ids_fit
            matched_mr_ids_fit_either = session_mr_ids_fit
        else:
            matched_mr_ids_fit_both = np.intersect1d(matched_mr_ids_fit_both, session_mr_ids_fit)
            matched_mr_ids_fit_either = np.union1d(matched_mr_ids_fit_either, session_mr_ids_fit)

    matched_session_ids = roi_mapping_df.query('master_roi_id in @matched_mr_ids')
    matched_session_ids_fit_both = roi_mapping_df.query('master_roi_id in @matched_mr_ids_fit_both')
    matched_session_ids_fit_either = roi_mapping_df.query('master_roi_id in @matched_mr_ids_fit_either')
    # Then, match back to each session
    # and compute clustering index
    ci_matched = []
    ci_fit_either_matched = []
    ci_fit_both_matched = []
    for session in session_nums_comp:
        touch_response_fn = f'JK{mouse:03}_volume{volume}_S{session:02}_ve_0.05_ptf_1.npy'
        touch_response_results = np.load(pop_res_dir / touch_response_fn, allow_pickle=True).item()

        response_df = touch_response_results['per_touch_response_df']
                
        response_xr = touch_response_results['per_touch_response_xr']
        session_ids_mr_matchback_ind = np.where(np.isin(response_xr.cell_id.values, matched_session_ids))[0]                                                    
        response_xr = response_xr.isel(cell_id=session_ids_mr_matchback_ind)    
        ci_matched.append(ci.calculate_clustering_index(response_xr, response_df,
                                                        num_dims=num_dims, num_repeat=num_repeat))

        session_ids_mr_matchback_ind = np.where(np.isin(response_xr.cell_id.values, matched_session_ids_fit_either))[0]
        response_xr_fit_either = response_xr.isel(cell_id=session_ids_mr_matchback_ind)
        ci_fit_either_matched.append(ci.calculate_clustering_index(response_xr_fit_either, response_df,
                                                                   num_dims=num_dims, num_repeat=num_repeat))

        session_ids_mr_matchback_ind = np.where(np.isin(response_xr.cell_id.values, matched_session_ids_fit_both))[0]
        response_xr_fit_both = response_xr.isel(cell_id=session_ids_mr_matchback_ind)
        ci_fit_both_matched.append(ci.calculate_clustering_index(response_xr_fit_both, response_df,
                                                                 num_dims=num_dims, num_repeat=num_repeat))
    return matched_mr_ids, matched_mr_ids_fit_either, matched_mr_ids_fit_both, \
        ci_matched, ci_fit_either_matched, ci_fit_both_matched


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