import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from time import time
from multiprocessing import Pool
import sys
sys.path.append(r'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Analysis\codes\data_analysis')
import utils.neural_stretching_test_sessions as nstest
from sklearn.decomposition import PCA

base_dir = Path(r'E:\TPM\JK\h5')

expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
test_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                 ~expert_mice_df['processing_error'].astype(bool) &
                                 ((expert_mice_df.session_type.str.contains('test')))]

def save_lda_test_sessions(mouse, top_plane, session, 
                           touch_window='before_answer',
                           spk_norm='std', 
                           varexp_threshold=0.05,
                           post_touch_frames=1):
    volume = 1 if top_plane == 1 else 2
    per_touch_response_xr_fit, per_touch_response_df, per_touch_response_xr = \
    nstest.get_touch_response_xr_varexp_threshold(base_dir, mouse, top_plane, session, touch_window=touch_window,
                                            spk_norm=spk_norm, varexp_threshold=varexp_threshold,
                                            post_touch_frames=post_touch_frames)
    pca = PCA()
    pca.fit_transform(per_touch_response_xr_fit)
    dim_nums = [*np.arange(2, 11), *np.arange(20, pca.components_.shape[0], 10)]
    lda_performances = []
    for di in dim_nums:
        X_pca = pca.transform(per_touch_response_xr_fit)[:,:di]
        lda_performances.append(nstest.get_lda_accuracies(X_pca, per_touch_response_df.pole_angle.values))
    
    save_dir = base_dir / 'results' / 'neural_stretching' / 'lda_performances'
    save_dir.mkdir(exist_ok=True, parents=True)
    save_fn = save_dir / f'JK{mouse:03}_volume{volume}_S{session:02d}_lda_performances_{touch_window}.npy'
    results = {'dim_nums': dim_nums, 'lda_performances': lda_performances}
    np.save(save_fn, results)


if __name__ == '__main__':
    t0 = time()
    test_df = test_mice_df.query('plane in [1, 5]')
    print(len(test_df))
    with Pool(19) as pool:
        pool.starmap(save_lda_test_sessions, 
                        [(row.mouse, row.plane, int(row.session)) for i, row in test_df.iterrows()])
    # for i, row in test_df.iterrows():
    #     if i < test_df.index.values[1]:
    #         print(f'Processing JK{row.mouse:03} S{int(row.session):02} plane {row.plane}')
    #         save_lda_test_sessions(row.mouse, row.plane, int(row.session))
    t1 = time()
    print(f'Elapsed time: {(t1-t0)/60:.1f} min')
                     
    # All expert mice test sessions took 35 min (with 19 processes)


