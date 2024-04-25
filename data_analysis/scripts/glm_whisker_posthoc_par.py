import fit_glm_whisker_combined
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

if __name__ == '__main__':
    # base_dir = Path(r'E:\TPM\JK\h5')
    # base_dir = Path(r'C:\JK')
    base_dir = Path(r'D:\JK\h5')

    expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
    use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                    ~expert_mice_df['processing_error'].astype(bool) &
                                    ((expert_mice_df.session_type == 'training') |
                                    (expert_mice_df.session_type.str.contains('test')))]
    t0 = time.time()
    # num_processes = cpu_count() - 2
    num_processes = 50
    print(f'Running with {num_processes} processes.')
    with Pool(processes=num_processes) as pool:
        pool.starmap(fit_glm_whisker_combined.run_glm_posthoc, [(mouse, plane, int(session), base_dir, 'whisker_combined')
                                                for mouse, plane, session
                                                in zip(use_mice_df.mouse.values, use_mice_df.plane.values, use_mice_df.session.values)])
    # i = 108
    # row = use_mice_df.iloc[i]
    # fit_glm_whisker_combined.run_glm_posthoc(row.mouse, row.plane, int(row.session), base_dir, 'whisker_combined')
    t1 = time.time()
    print(f'{(t1-t0)/60:.1f} min elapsed')

# Took 464.3 min (7.7 hours) for 50 processes (all expert use mice, whisker combined, ridge)
# For 469 planes (total 703 planes)
