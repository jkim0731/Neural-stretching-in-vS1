import sys
sys.path.append(r'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Analysis\codes\data_analysis')
import utils.population_activity as pa
from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing.pool import Pool


base_dir = Path(r'E:\TPM\JK\h5')
results_dir = base_dir / 'results' / 'pop_responses'

expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                 ~expert_mice_df['processing_error'].astype(bool) &
                                 ((expert_mice_df.session_type == 'training') |
                                  (expert_mice_df.session_type.str.contains('test')))]
use_volume_df = use_mice_df.query('plane in [1, 5]')


def save_touch_response(base_dir, mouse, top_plane, session, 
                      save_dir_name: str,
                      args={}):
    touch_window = args['touch_window'] if 'touch_window' in args.keys() else 'before_answer'
    spk_norm = args['spk_norm'] if 'spk_norm' in args.keys() else 'std'
    varexp_threshold = args['varexp_threshold'] if 'varexp_threshold' in args.keys() else 0.05
    post_touch_frames = args['post_touch_frames'] if 'post_touch_frames' in args.keys() else 1
    volume = 1 if top_plane == 1 else 2
    print(f'Processing JK{mouse:03} S{session:02} volume {volume}')

    per_touch_response_xr_fit, per_touch_response_df, per_touch_response_xr = \
    pa.get_touch_response_xr_varexp_threshold(base_dir, mouse, top_plane, session, touch_window=touch_window,
                                            spk_norm=spk_norm, varexp_threshold=varexp_threshold,
                                            post_touch_frames=post_touch_frames)
    results = {'per_touch_response_xr_fit': per_touch_response_xr_fit,
               'per_touch_response_df': per_touch_response_df,
               'per_touch_response_xr': per_touch_response_xr}
    save_dir = results_dir / save_dir_name
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    save_fn = save_dir / f'JK{mouse:03}_volume{volume}_S{session:02d}_ve_{varexp_threshold}_ptf_{post_touch_frames}.npy'
    np.save(save_fn, results)


if __name__ == '__main__':
    args = {'touch_window': 'before_answer',
            'spk_norm': 'std',
            'varexp_threshold': 0.05,
            'post_touch_frames': 2}
    with Pool(19) as pool:
        pool.starmap(save_touch_response,
                        [(base_dir, row.mouse, row.plane, int(row.session),
                          'touch_before_answer', args
                          ) for i, row in use_volume_df.iterrows()])
    # Takes about 3 min with 19 pools for touch_before_answer
        

# touch_window='before_answer'
# spk_norm='std'
# varexp_threshold=0.05
# post_touch_frames=0

# for post_touch_frames in [0,1,2]:

#     mouse = 25
#     for volume in [1,2]:
#         top_plane = 1 if volume==1 else 5

#         sessions = [int(s) for s in use_mice_df.query('mouse == @mouse and plane==@top_plane').session.unique()]
#         for session in sessions:
#             per_touch_response_xr_fit, per_touch_response_df, per_touch_response_xr = \
#                 pa.get_touch_response_xr_varexp_threshold(base_dir, mouse, top_plane, session,
#                                                         touch_window=touch_window,
#                                                         spk_norm=spk_norm,
#                                                         varexp_threshold=varexp_threshold,
#                                                         post_touch_frames=post_touch_frames)
#             results = {'per_touch_response_xr_fit': per_touch_response_xr_fit,
#                     'per_touch_response_df': per_touch_response_df,
#                     'per_touch_response_xr': per_touch_response_xr}
#             save_dir = results_dir / 'touch_before_answer'
#             if not save_dir.exists():
#                 save_dir.mkdir(parents=True)
#             save_fn = save_dir / f'JK{mouse:03}_volume{volume}_S{session:02d}_ve_{varexp_threshold}_ptf_{post_touch_frames}.npy'
#             np.save(save_fn, results)

        
