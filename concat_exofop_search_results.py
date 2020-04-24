"""
Concatenate the exofop search results stored in the data/exofop folder, as created
by ExoFOP searches using the TIC IDs resulting from the get_tic_ids script. See
the file docstring in get_tic_ids.py for more information on how to exofop search
results are queried.
"""
import glob
import argparse

import pandas as pd

# Command line arguments. For most uses, the defaults will be fine.
parser = argparse.ArgumentParser(description='Merge multiple exofop search results into a single .csv file.')
parser.add_argument('fname_base', type=str, default=None, help='File save path for output. Numbers will be appended to \
keep the total number of lines less than 1000, which is the max length of a file that the ExoFOP page can read in at one time.')
parser.add_argument('--exofop_folder', type=str, default='data/exofop/', help='Folder with ExoFOP search results.')
parser.add_argument('--save_fname', type=str, default=None, help='Name of file to save output to. By default it matches the str in fname_base')

def get_files_like(fname_base, exofop_folder):
    '''
    Like get_newest_csv in X_ranking.py, but match the format of fname_base.
    '''
    fname_like = exofop_folder + fname_base + '*'
    list_of_files_with_base = glob.glob(fname_like)
    return list_of_files_with_base

def concat_exofop_search_results(fname_base, exofop_folder):
    """
    Concatenate the exofop search results stored in the data/exofop folder, as created
    by ExoFOP searches using the TIC IDs resulting from the get_tic_ids script. See
    the file docstring in get_tic_ids.py for more information on how to exofop search
    results are queried.
    """

    files = get_files_like(fname_base, exofop_folder)
    print(f"Concatenating {len(files)} files into a single .csv")
    return pd.concat([pd.read_csv(f, comment='#') for f in files], sort=False)

if __name__ == "__main__":

    args = parser.parse_args()
    fname_base = args.fname_base
    exofop_folder = args.exofop_folder

    save_fname = args.save_fname
    if save_fname is None:
        save_fname = exofop_folder + args.fname_base + '.csv'

    fname_base += '_' # This prevents us from concatenating a newly saved output .csv file to itself.

    exofop_df = concat_exofop_search_results(fname_base, exofop_folder)
    exofop_df.to_csv(save_fname, index=False)

    print(f"The concatenated ExoFOP .csv file was saved to {save_fname}.")
