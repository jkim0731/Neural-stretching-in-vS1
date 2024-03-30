import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from multiprocessing import Pool
import sys
sys.path.append(r'C:\Users\shires\Dropbox\Works\Projects\2020 Neural stretching in S1\Analysis\codes\data_analysis')
import utils.lda_angle_discrim as lda_angle
import utils.population_activity as pa
from sklearn.decomposition import PCA

base_dir = Path(r'E:\TPM\JK\h5')

expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                 ~expert_mice_df['processing_error'].astype(bool) &
                                 ((expert_mice_df.session_type == 'training') |
                                  (expert_mice_df.session_type.str.contains('test')))]

def save_lda_45_135(mouse, top_plane, session, 
                    touch_window='before_answer',
                    spk_norm='std', 
                    varexp_threshold=0.05,
                    post_touch_frames=1):
    volume = 1 if top_plane == 1 else 2
    print(f'Processing JK{mouse:03} S{session:02} volume {volume}')

    per_touch_response_xr_fit, per_touch_response_df, per_touch_response_xr = \
    pa.get_touch_response_xr_varexp_threshold(base_dir, mouse, top_plane, session, touch_window=touch_window,
                                            spk_norm=spk_norm, varexp_threshold=varexp_threshold,
                                            post_touch_frames=post_touch_frames)
    
    per_touch_response_df_45_135 = per_touch_response_df.query('pole_angle == 45 or pole_angle == 135')
    tn_45_135 = per_touch_response_df_45_135.trialNum.values
    per_touch_response_xr_fit_45_135 = per_touch_response_xr_fit.sel(trialNum=tn_45_135)
    # check the order is the same
    assert np.all(per_touch_response_xr_fit_45_135.trialNum.values == per_touch_response_df_45_135.trialNum.values)

    pca = PCA()
    pca.fit_transform(per_touch_response_xr_fit_45_135)
    dim_nums = [*np.arange(3, 17, 2)]
    lda_performances = []
    for di in dim_nums:
        X_pca = pca.transform(per_touch_response_xr_fit_45_135)[:,:di]
        lda_performances.append(lda_angle.get_lda_accuracies(X_pca, per_touch_response_df_45_135.pole_angle.values))

    save_dir = base_dir / 'results' / 'neural_stretching' / 'lda_performances'
    save_dir.mkdir(exist_ok=True, parents=True)
    save_fn = save_dir / f'JK{mouse:03}_volume{volume}_S{session:02d}_lda_performances_45_135_{touch_window}.npy'
    results = {'dim_nums': dim_nums, 'lda_performances': lda_performances}
    np.save(save_fn, results)


if __name__ == '__main__':
    t0 = time()
    test_df = use_mice_df.query('plane in [1, 5]')
    print(len(test_df))
    # with Pool(19) as pool:
    #     pool.starmap(save_lda_45_135, 
    #                     [(row.mouse, row.plane, int(row.session)) for i, row in test_df.iterrows()])
    for i, row in test_df.iterrows():
        if i < test_df.index.values[1]:
            print(f'Processing JK{row.mouse:03} S{int(row.session):02} plane {row.plane}')
            save_lda_45_135(row.mouse, row.plane, int(row.session))
    t1 = time()
    print(f'Elapsed time: {(t1-t0)/60:.1f} min')                     


