import design_matrix_whisker_combined as dm_whisker_combined
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import time


def make_and_save_design_matrix_whisker_combined(mouse, plane, session, base_dir):
    print(f'processing {mouse:03} plane {plane} session {session:03}')
    design_df, _ = dm_whisker_combined.make_design_dataframe(mouse, plane, session, base_dir)
    save_dir = base_dir / f'{mouse:03}/plane_{plane}/{session:03}/plane0/roi/glm/whisker_combined'
    save_dir.mkdir(exist_ok=True, parents=True)
    design_df.to_pickle(save_dir / 'design_whisker_combined.pkl')
    # Temporarily, remove the previous design matrix
    (save_dir / 'design.pkl').unlink(missing_ok=True)

if __name__ == '__main__':
    base_dir = Path(r'E:\TPM\JK\h5')

    expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
    use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                    ~expert_mice_df['processing_error'].astype(bool) &
                                    ((expert_mice_df.session_type == 'training') |
                                    (expert_mice_df.session_type.str.contains('test')))]
    t0 = time.time()
    with Pool(processes=18) as pool:
        pool.starmap(make_and_save_design_matrix_whisker_combined, [(mouse, plane, int(session), base_dir)
                                                for mouse, plane, session
                                                in zip(use_mice_df.mouse.values, use_mice_df.plane.values, use_mice_df.session.values)])
    # i = 0
    # mouse = use_mice_df.mouse.values[i]
    # plane = use_mice_df.plane.values[i]
    # session = int(use_mice_df.session.values[i])
    # make_and_save_design_matrix_touch_combined(mouse, plane, session, base_dir)
    t1 = time.time()
    print(f'{(t1-t0)/60:.1f} min elapsed')
    # took 7.7 min for all expert use mice
    
