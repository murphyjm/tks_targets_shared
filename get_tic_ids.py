'''
A short script to get a list of TIC IDs from the newest TOI+ list, break up the
TIC IDs into .txt files of <= EXOFOP_MAX_LEN rows each, and save the .txt files
to the data/tic_id_lists/ folder (by default).

Once these files are saved:
Go to the ExoFOP-TESS Search page: https://exofop.ipac.caltech.edu/tess/search.php
- Change coordinates to Decimal Degrees (is this necessary?)
- Under the Magnitudes tab, select "Include in Output" for V, J, H, K
- Upload each .txt file and click the search button
- Save the output to the data/exofop folder
'''
import argparse

import pandas as pd
import numpy as np
from math import ceil

from X_ranking import get_newest_csv

# Command line arguments. For most uses, the defaults will be fine.
parser = argparse.ArgumentParser(description='Extract and save TIC IDs in .txt files that ExoFOP-TESS can handle.')
parser.add_argument('save_fname_base', type=str, default=None, help='File save path for output. Numbers will be appended to \
keep the total number of lines less than 1000, which is the max length of a file that the ExoFOP page can read in at one time.')
parser.add_argument('--toi_folder', type=str, default='data/toi/', help='Folder with toi+ lists.')
parser.add_argument('--output_dir', type=str, default='data/tic_id_lists/', help='Folder to save the lists of the TIC IDs to.')

# The maximum number of rows in a .txt file that can be uploaded for a single ExoFOP query.
EXOFOP_MAX_LEN = 1000

def get_tic_ids(toi_folder):
    '''
    A short function to get a list of the TIC IDs that are contained in the TOI+
    list.
    '''
    toi_fname = get_newest_csv(toi_folder)
    toi_df = pd.read_csv(toi_fname, comment='#')
    return toi_df['TIC'].values.copy()

def save_tic_lists(tic_ids, save_fname_base, output_dir):
    '''
    Break up the list of TIC IDs into chunks that the ExoFOP search tool can handle.
    Save the lists to .txt files.
    '''
    num_tics = len(tic_ids)
    print(f"There are {num_tics} TIC IDs.")

    if num_tics >= EXOFOP_MAX_LEN:
        num_files = int(ceil(num_tics / EXOFOP_MAX_LEN))

        for i in range(num_files):
            fname = save_fname_base + f"_{i+1}of{num_files}.txt"
            print(f'Saving file {i+1} of {num_files}...')
            np.savetxt(output_dir + fname, tic_ids[i*EXOFOP_MAX_LEN:(i+1)*EXOFOP_MAX_LEN], fmt='%d')
    else:
        fname = save_fname_base + f"_1of1.txt"
        np.savetxt(output_dir + fname, tic_ids, fmt='%d')

    print(f'The lists of TIC IDs are stored in {output_dir}.')

if __name__ == "__main__":

    args = parser.parse_args()
    save_fname_base = args.save_fname_base
    toi_folder = args.toi_folder
    output_dir = args.output_dir

    tic_ids = get_tic_ids(toi_folder)
    save_tic_lists(tic_ids, save_fname_base, output_dir)
