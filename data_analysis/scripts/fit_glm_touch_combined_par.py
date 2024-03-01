import fit_glm_touch_combined
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import time

if __name__ == '__main__':
    base_dir = Path(r'E:\TPM\JK\h5')

    expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
    use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                    ~expert_mice_df['processing_error'].astype(bool) &
                                    ((expert_mice_df.session_type == 'training') |
                                    (expert_mice_df.session_type.str.contains('test')))]
    t0 = time.time()
    with Pool(processes=18) as pool:
        pool.starmap(fit_glm_touch_combined.run_glm_and_save, [(mouse, plane, int(session), base_dir, 'touch_combined')
                                                for mouse, plane, session
                                                in zip(use_mice_df.mouse.values, use_mice_df.plane.values, use_mice_df.session.values)])
    t1 = time.time()
    print(f'{(t1-t0)/60:.1f} min elapsed')
## about 10 hours for all expert use mice
