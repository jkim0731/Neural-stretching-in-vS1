from scipy.io import savemat
import random
import sys

def par_test(itest):
    print('Start')
    sys.stdout.flush()
    text = random.randint(1, 101)
    savefn = f'C:/Users/shires/Dropbox/Works/Projects/2020 Neural stretching in S1/Analysis/codes/test/save_{text}.mat'
    savemat(f'{savefn}', {'text':text})
    print('Finish')
    sys.stdout.flush()