# from suite2p.run_s2p import run_s2p, default_ops
from suite2p.run_s2p import run_s2p
from scipy.io import savemat
import sys

def run_s2p_parallel(db, ops):
    print(f'Starting {db["h5py"]}')
    sys.stdout.flush()
    opsEnd = run_s2p(ops, db)
    print(f'Saving {db["h5py"]}')
    sys.stdout.flush()
    saveFn = db['matSaveFn']
    savemat(f'{saveFn}', {'mimg':opsEnd['meanImg'].transpose()})
    print(f'Done! {db["h5py"]}')
    sys.stdout.flush()