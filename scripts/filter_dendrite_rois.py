import numpy as np
import pandas as pd
from pathlib import Path

def pix_to_eq_diameter(npix, pix_size):
    return np.sqrt(npix/np.pi) * 2 * pix_size

base_dir = Path(r'E:\TPM\JK\h5')
expert_mice_df = pd.read_csv(base_dir / 'expert_mice.csv', index_col=0)
use_mice_df = expert_mice_df.loc[expert_mice_df['depth_matched'].astype(bool) & 
                                 ~expert_mice_df['processing_error'].astype(bool) &
                                 ((expert_mice_df.session_type == 'training') |
                                  (expert_mice_df.session_type.str.contains('test')))]
mice = use_mice_df.mouse.unique()


diameter_threshold = 7

planes = np.arange(1,9)
for mouse in mice:
    for plane in planes:
        plane_dir = base_dir / f'{mouse:03}' / f'plane_{plane}'
        session_names = use_mice_df.query('mouse==@mouse and plane==@plane').session.unique()
        for session_name in session_names:        
            session_dir = plane_dir / f'{session_name}' / 'plane0'
            roi_dir = session_dir / 'roi'
            ops = np.load(roi_dir / 'ops.npy', allow_pickle=True).item()
            pix_size = ops['umPerPix']
            iscell = np.load(roi_dir / 'iscell.npy', allow_pickle=True)
            stats = np.load(roi_dir / 'stat_refined.npy', allow_pickle=True)
            for i in range(len(stats)):
                if pix_to_eq_diameter(stats[i]['npix'], pix_size) < diameter_threshold:
                    iscell[i,:] = 0
            np.save(roi_dir / 'iscell.npy', iscell)