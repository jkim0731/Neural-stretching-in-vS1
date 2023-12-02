from pathlib import Path
import multiprocessing
import numpy as np
from scripts import roi_remove_border as rrb
from matplotlib import pyplot as plt
import time


def save_border_filter_mask(session_tuple, base_dir):

    mouse = session_tuple[0]
    plane = session_tuple[1]
    session = session_tuple[2]

    print(f'Processing {mouse:03} plane {plane} session {session}...')
    final_mask = rrb.border_filter_mask(session_tuple, base_dir)
    final_mask_fn = base_dir / f'{mouse:03}/plane_{plane}/{session}/plane0/roi/final_mask.npy'
    np.save(final_mask_fn, final_mask)

    ops = np.load(base_dir / f'{mouse:03}/plane_{plane}/{session}/plane0/ops.npy', allow_pickle=True).item()
    mean_img = ops['meanImg']
    fig, ax = plt.subplots(figsize=(mean_img.shape[0]*0.02, mean_img.shape[1]*0.02))
    ax.imshow(mean_img)
    for i in range(final_mask.shape[-1]):
        ax.contour(final_mask[:,:,i], colors='r', linewidths=0.5)

    save_dir = base_dir / f'{mouse:03}/plane_{plane}' / 'roi_collection_test'
    save_dir.mkdir(exist_ok=True)
    fig.savefig(save_dir / f'{mouse:03}_plane_{plane}_{session}.png', dpi=300)
    plt.close(fig)
    print(f'Finished processing {mouse:03} plane {plane} session {session}')


if __name__ == '__main__':
    t0 = time.time()
    base_dir = Path(r'E:\TPM\JK\h5')
    mice = [25,27,30,36,39,52]
    planes = range(1,9)
    training_session_tuples = []
    for mouse in mice:
        for plane in planes:
            plane_dir = base_dir / f'{mouse:03}/plane_{plane}'
            sessions = [x.name for x in plane_dir.iterdir() if \
                        x.is_dir() and x.name[0].isdigit() and len(x.name)==3]
            # list of (mouse,plane,session) tuples
            for session in sessions:
                training_session_tuples.append((mouse,plane,session))

    num_cores = multiprocessing.cpu_count() - 2
    print(f'Number of cores = {num_cores}')
    with multiprocessing.Pool(num_cores) as pool:
        pool.starmap(save_border_filter_mask, zip(training_session_tuples, [base_dir]*len(training_session_tuples)))
        
    t1 = time.time()
    print(f'Time elapsed: {(t1-t0)/60:.2f} minutes')
    # 215.81 min