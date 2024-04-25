import fit_glm_whisker_combined
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import time

import socket
hostname = socket.gethostname()

if __name__ == '__main__':
    if hostname == 'HNB228-LABPC6':
        base_dir = Path(r'D:\JK\h5')
    else:
        base_dir = Path(r'E:\TPM\JK\h5')

    ridge_dir = base_dir / 'results/neuron_glm/ridge/whisker_combined'

    expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
    use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                    ~expert_mice_df['processing_error'].astype(bool) &
                                    ((expert_mice_df.session_type == 'training') |
                                    (expert_mice_df.session_type.str.contains('test')))]
    t0 = time.time()
    with Pool(processes=60) as pool:
        pool.starmap(fit_glm_whisker_combined.run_lasso_glm_and_save, [(mouse, plane, int(session), base_dir, ridge_dir, 'whisker_combined')
                                                for mouse, plane, session
                                                in zip(use_mice_df.mouse.values, use_mice_df.plane.values, use_mice_df.session.values)])
    with Pool(processes=60) as pool:
        pool.starmap(fit_glm_whisker_combined.run_glm_posthoc, [(mouse, plane, int(session), base_dir, 'whisker_combined', 'lasso')
                                                for mouse, plane, session
                                                in zip(use_mice_df.mouse.values, use_mice_df.plane.values, use_mice_df.session.values)])
    # i = 0
    # row = use_mice_df.iloc[i]    
    # fit_glm_whisker_combined.run_lasso_glm_and_save(row.mouse, row.plane, int(row.session), base_dir,
    #                                                 ridge_dir=ridge_dir, glm_type='whisker_combined')
    # fit_glm_whisker_combined.run_glm_posthoc(row.mouse, row.plane, int(row.session), base_dir,
    #                                          glm_type='whisker_combined',
    #                                          model='lasso')
    t1 = time.time()
    print(f'{(t1-t0)/60:.1f} min elapsed')
## about 11.5 hours for all expert use mice (with 18 processes, ridge, no posthoc)

