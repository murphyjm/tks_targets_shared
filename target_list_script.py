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
parser.add_argument('--toi_folder', type=str, default='data/toi/', help='Folder with toi+ lists.')
parser.add_argument('--tic_folder', type=str, default='data/exofop/', help='Folder with TIC info.')
parser.add_argument('--selected_TOIs_folder', type=str, default='data/TKS/', help='Folder with selected_TOIs csv.')
parser.add_argument('--verbose', type=bool, default=True, help='Print additional messages during target list generation?')

if __name__ == '__main__':

    args = parser.parse_args()
    verbose = args.verbose

    # Get the initial target list
    print('Generating initial target list...')
    print('')
    X_tois_df = get_target_list(save_fname=args.save_fname, toi_folder=args.toi_folder, tic_folder=args.tic_folder, selected_TOIs_folder=args.selected_TOIs_folder, verbose=args.verbose)
    print('----------')
