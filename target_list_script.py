'''
This script uses functions from X_ranking.py to create an initial target list to
input to the TKS prioritization algorithm. It ranks targets based on the "X" metric,
which is the ratio of the target's TSM and t_HIRES, the expected total exposure
time needed to achieve a 5-sigma mass measurement on the system.

Notes:
    - Right now (03/24/20), the program does not incorporate information from Jump.
        Though, information that would be useful to have from Jump includes:
        - The number of observations a target already has, so we can deduct the
            proper amount of time from t_HIRES since these targets have a "head start"
        - Parameters used for prior observations e.g.
            - N_shots (number of shots be observation)
            - exposure_meter_target (target number of exposure meter counts)
        - If a target already has RVs, an estimate of the typical RV measurement
            precision, measured from the error bars on the RV points themselves.
        - If a target already has a mass measurement from RadVel, an updated calculation
            of t_HIRES, extrapolated from the number of current observations and
            SNR of the mass measurement.
            - The first two bullet points can be gathered from data using an SQL
                query to Jump. Getting the actual mass precision would be a little
                more involved because it's not already in a Jump database. Maybe,
                since there aren't all that many prioritized targets, these measurements
                could be collated by hand in a .csv somewhere.
'''
# System
import os
import sys
import argparse

from X_ranking import *

# Command line arguments. For most uses, the defaults will be fine.
parser = argparse.ArgumentParser(description='Rank targets in selected_TOIs.')
parser.add_argument('save_fname', type=str, default=None, help='File save path for output.')
parser.add_argument('--save_folder', type=str, default='data/sc3_target_lists/', help='Folder where the output will be saved.')
parser.add_argument('--toi_folder', type=str, default='data/toi/', help='Folder with toi+ lists.')
parser.add_argument('--tic_folder', type=str, default='data/exofop/', help='Folder with TIC info.')
parser.add_argument('--selected_TOIs_folder', type=str, default='data/TKS/', help='Folder with selected_TOIs csv.')
parser.add_argument('--include_qlp', type=str, default='False', help='Include QLP TOIs in ranking algorithm?')
parser.add_argument('--verbose', type=str, default='True', help='Print additional messages during target list generation?')
parser.add_argument('--num_to_rank', type=str, default='5', help='Number of targets to assign priorities to per bin.')
parser.add_argument('--k_amp_cut', type=float, default=2., help='Minimum expected K-amplitude that targets must have to make the final list.')
parser.add_argument('--min_TSM', type=float, default=0., help='Minimum TSM value that targets must have to make the final list.')

def save_to_csv(df, save_fname, save_folder):
    '''
    Save the df to file.
    '''
    df.to_csv(save_folder + save_fname)
    print('The target list was saved to {}.'.format(save_fname))

def mark_vip_targets(df, vip_fname):
    '''
    Add a column to the DataFrame signifying a VIP priority for certain targets that
    we want to send directly to the top of our priority list, regardless of other
    sorting methods.
    '''
    if vip_fname is not '':
        vip_list = np.loadtxt(vip_fname, dtype=str)
        data = zip(vip_list, np.arange(1, len(vip_list)+1))
        cols = ['cps', 'vip_rank']
        vip_df = pd.DataFrame(data, columns=cols)

        # Merge the X_tois_df with the vip_df while preserving the indexes on the X_tois_df, which contain binning info
        return df.reset_index().merge(vip_df, how='left', left_on='cps', right_on='cps').set_index(df.index.names)
    else:
        df['vip_rank'] = np.nan # Make vip_rank column empty is no file given
        return df

if __name__ == '__main__':

    args = parser.parse_args()
    save_fname = args.save_fname
    save_folder = args.save_folder
    toi_folder = args.toi_folder
    tic_folder = args.tic_folder
    selected_TOIs_folder = args.selected_TOIs_folder
    num_to_rank = int(args.num_to_rank)
    k_amp_cut = float(args.k_amp_cut)
    min_TSM = float(args.min_TSM)

    # Convert these optional arguments to other data types, if they're specified.
    include_qlp_str = args.include_qlp
    assert include_qlp_str.lower() in ['false', 'true'], '--include_qlp must be either True or False'
    include_qlp = False
    if include_qlp_str.lower() == 'false':
        include_qlp = False
    elif include_qlp_str.lower() == 'true':
        include_qlp = True

    verbose_str = args.verbose
    assert verbose_str.lower() in ['false', 'true'], '--verbose must be either True or False'
    verbose = None
    if verbose_str.lower() == 'false':
        verbose = False
    elif verbose_str.lower() == 'true':
        verbose = True

    # Get the initial target list
    print('Generating initial target list...')
    print('')
    X_tois_df = get_target_list(save_fname=None, toi_folder=toi_folder, tic_folder=tic_folder, selected_TOIs_folder=selected_TOIs_folder, include_qlp=include_qlp, verbose=verbose, num_to_rank=num_to_rank, k_amp_cut=k_amp_cut, min_TSM=min_TSM)
    print('----------')

    # Add VIP targets
    print('Would you like to mark VIP targets that will get selected first? y/n')
    vip_yn_valid = False
    while not vip_yn_valid:
        vip_yn = input().lower()
        if vip_yn in ['yes', 'y']:
            vip_yn_valid = True
            sys.stdout.write("Provide the path to the .txt file with your VIP targets' CPS IDs listed one per row, in order of highest VIP to lowest: ")
            vip_path_valid = False
            while not vip_path_valid:
                vip_fname = input()
                if os.path.exists(vip_fname):
                    vip_path_valid = True
                else:
                    print('')
                    print('That is not a valid path, enter another...')
            X_tois_df = mark_vip_targets(X_tois_df, vip_fname)
            save_to_csv(X_tois_df, save_fname, save_folder)
        elif vip_yn in ['no', 'n']:
            vip_yn_valid = True
            X_tois_df = mark_vip_targets(X_tois_df, '') # Add the vip_rank column but leave it empty
            save_to_csv(X_tois_df, save_fname, save_folder)
        else:
            print('Please enter yes or no...')
    print('----------')
