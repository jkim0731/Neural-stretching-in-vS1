import os, mat73, tqdm, glob
import pandas as pd
from multiprocessing import Pool
from itertools import repeat
from pathlib import Path

def build_whisker_dataframe(w_dir, session_folder):
    session_name = session_folder.split('\\')[-1]
    wf_fn = w_dir / session_name / f'{session_name}_whisker_final_h5.mat'
    whisker_pkl_fn = w_dir / session_name / f'{session_name}_whisker.pkl'
    # if not os.path.isfile(whisker_pkl_fn):
    if os.path.isfile(wf_fn):
        print(f'Processing {session_name}', flush=True)
        w_data = mat73.loadmat(wf_fn)
        print(f'{session_name} data loaded.', flush=True)
        dict_keys = list(w_data['hw']['trials'][1].keys())
        whisker_df = pd.DataFrame(columns=dict_keys)
        num_trials = len(w_data['hw']['trials'])
        for i in range(num_trials):
            trial_data = w_data['hw']['trials'][i]
            trial_df = pd.DataFrame({})
            for key in dict_keys:
                trial_df[key] = [trial_data[key]]
            whisker_df = whisker_df.append(trial_df)
        whisker_df['mouse_name'] = w_data['hw']['mouse_name']
        whisker_df['session_name'] = w_data['hw']['session_name']
        whisker_df.rename(columns={'time': 'whisker_time'}, inplace=True)
        whisker_df['trialNum'] = whisker_df['trialNum'].astype(int)
        whisker_df['poleAngle'] = whisker_df['poleAngle'].astype(int)
        whisker_df['poleDistance'] = whisker_df['poleDistance'].astype(int)
        whisker_df.set_index('trialNum', inplace=True)
        whisker_df.to_pickle(whisker_pkl_fn)
        print(f'{session_name} Done.', flush=True)
    
if __name__ == '__main__':
    mice =          [25,    27,   30,   36,     37,     38,     39,     41,     52,     53,     54,     56]
    w_dir = Path('D:/WhiskerVideo/')
    for mouse in tqdm.tqdm(mice):
        session_folder_list = glob.glob(str(w_dir / f'JK{mouse:03d}S*'))
        with Pool() as pool:
            pool.starmap(build_whisker_dataframe, zip(repeat(w_dir), session_folder_list))