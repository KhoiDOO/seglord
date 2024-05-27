from glob import glob
from tqdm import tqdm

import os
import shutil


if __name__ == '__main__':
    run_dir = os.getcwd() + '/runs'

    if not os.path.exists(run_dir):
        exit(0)
    else:
        svdirs = glob(run_dir + '/*')

        with tqdm(total=len(svdirs)) as pbar:
            pbar.set_description('Cleaning')
            for svdir in svdirs:

                done_path = svdir + '/done.txt'

                if not os.path.exists(done_path):
                    shutil.rmtree(svdir)
                
                pbar.update(1)