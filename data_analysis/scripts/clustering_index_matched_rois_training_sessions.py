import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(r'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Analysis\codes\data_analysis')
import utils.clustering_index as ci
from multiprocessing import Pool
from time import time
import socket
host_name = socket.gethostname()
if host_name == 'HNB228-LABPC6':
    base_dir = Path(r'D:\JK\h5')
    num_processses = 60
elif host_name == 'hnb228-jinho':
    # base_dir = Path(r'E:\TPM\JK\h5')
    base_dir = Path(r'C:\JK')
    num_processes = 18
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
    ci_matched_all = []
    ci_matched_fit_either = []
    ci_matched_fit_both = []
    for session in session_nums_comp:
        touch_response_fn = f'JK{mouse:03}_volume{volume}_S{session:02}_ve_0.05_ptf_1.npy'
        touch_response_results = np.load(pop_res_dir / touch_response_fn, allow_pickle=True).item()

        response_df = touch_response_results['per_touch_response_df']
                
        response_xr = touch_response_results['per_touch_response_xr']
        session_ids_mr_matchback_ind = np.where(np.isin(response_xr.cell_id.values, matched_session_ids))[0]                                                    
        response_xr = response_xr.isel(cell_id=session_ids_mr_matchback_ind)    
        ci_matched_all.append(ci.calculate_clustering_index(response_xr, response_df,
                                                        num_dims=num_dims, num_repeat=num_repeat))

        session_ids_mr_matchback_ind = np.where(np.isin(response_xr.cell_id.values, matched_session_ids_fit_either))[0]
        response_xr_fit_either = response_xr.isel(cell_id=session_ids_mr_matchback_ind)
        ci_matched_fit_either.append(ci.calculate_clustering_index(response_xr_fit_either, response_df,
                                                                   num_dims=num_dims, num_repeat=num_repeat))

        session_ids_mr_matchback_ind = np.where(np.isin(response_xr.cell_id.values, matched_session_ids_fit_both))[0]
        response_xr_fit_both = response_xr.isel(cell_id=session_ids_mr_matchback_ind)
        ci_matched_fit_both.append(ci.calculate_clustering_index(response_xr_fit_both, response_df,
                                                                 num_dims=num_dims, num_repeat=num_repeat))
    return matched_mr_ids, matched_mr_ids_fit_either, matched_mr_ids_fit_both, \
        ci_matched_all, ci_matched_fit_either, ci_matched_fit_both


def save_ci_matched(mouse, volume, session_nums_comp, save_dir,
                    num_repeat=10000):
    assert len(session_nums_comp) == 2
    assert session_nums_comp[0] != session_nums_comp[1]
    save_fn = save_dir / f'JK{mouse:03}_volume{volume}_ci_matched_{session_nums_comp[0]:02}_{session_nums_comp[1]:02}.pkl'
    if save_fn.exists():
        print(f'{save_fn.name} already exists. Skip.')
        return
    else:
        t0 = time()
        print(f'Processing {save_fn.name}')
        matched_mr_ids, matched_mr_ids_fit_either, matched_mr_ids_fit_both, \
            ci_matched, ci_matched_fit_either, ci_matched_fit_both = get_ci_from_matched_rois(mouse, volume, session_nums_comp,
                                                                                                num_repeat=num_repeat)
        save_fn = save_dir / f'JK{mouse:03}_volume{volume}_ci_matched_{session_nums_comp[0]:02}_{session_nums_comp[1]:02}.pkl'
        save_dict = {'mouse': mouse,
                     'volume': volume,
                     'session_1': session_nums_comp[0],
                     'session_2': session_nums_comp[1],
                     'ci_matched_all': [np.array(ci_matched)],
                     'ci_matched_fit_either': [np.array(ci_matched_fit_either)],
                     'ci_matched_fit_both': [np.array(ci_matched_fit_both)],
                     'matched_mr_ids': [matched_mr_ids],
                     'matched_mr_ids_fit_either': [matched_mr_ids_fit_either],
                     'matched_mr_ids_fit_both': [matched_mr_ids_fit_both],
                     }
        save_dict = pd.DataFrame(save_dict)
        save_dict.to_pickle(save_fn)
        t1 = time()
        print(f'Saved {save_fn.name} in {(t1-t0)/60:.1f} min')


def collect_all_results(save_dir):
    save_dir = Path(save_dir)
    save_fn = save_dir / 'clustering_index_matched_training_sessions.pkl'
    load_fns = list(save_dir.glob('JK*_volume*_ci_matched_*_*.pkl'))
    all_results = pd.DataFrame(columns=['mouse', 'volume', 'session_1', 'session_2',
                                        'ci_matched_all', 'ci_matched_fit_either', 'ci_matched_fit_both',
                                        'matched_mr_ids', 'matched_mr_ids_fit_either', 'matched_mr_ids_fit_both'
                                        ])
    for load_fn in load_fns:
        all_results.append(pd.read_pickle(load_fn))
    all_results.to_pickle(save_fn)


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

    save_dir = results_dir / 'clustering_index_matched_roi' / 'training_sessions'
    save_dir.mkdir(parents=True, exist_ok=True)


    # mouse = 25
    # volume = 1
    # sessions = [int(s) for s in training_volume_df.query('mouse==@mouse and volume==@volume').session.values]
    # session_nums_comp = [sessions[0], sessions[1]]
    # save_ci_matched(mouse, volume, session_nums_comp, save_dir, num_repeat=100)

    # 90 min with 18 cores for pairwise comparison
    comp_snum_df = pd.DataFrame(columns=['mouse', 'volume', 'session_nums_comp'])
    for mouse in training_volume_df.mouse.unique():
        for volume in range(1,3):
            sessions = [int(s) for s in training_volume_df.query('mouse==@mouse and volume==@volume').session.values]
            for si in range(len(sessions)-1):
                for sj in range(si+1, len(sessions)):
                    temp_df = pd.DataFrame({'mouse': mouse,
                                            'volume': volume,
                                            'session_nums_comp': [np.array([sessions[si], sessions[sj]])],
                                            })
                    comp_snum_df = comp_snum_df.append(temp_df)

    t0 = time()
    with Pool(num_processes) as pool:
        pool.starmap(save_ci_matched, [(row['mouse'], row['volume'], row['session_nums_comp'], save_dir) \
                                            for _, row in comp_snum_df.iterrows()])
    t1 = time()
    print(f'Elapsed time: {(t1-t0)/60:.1f} min')

    collect_all_results(save_dir)