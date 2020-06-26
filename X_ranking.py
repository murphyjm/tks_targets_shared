'''
This combines multiple cells from the prioritization_TSM_t_HIRES_ratio notebook into a single script,
so that it outputs a simple dataframe with rankings for both methods.
'''
# System
import os
import sys
import glob

# Basic analysis
import numpy as np
import pandas as pd

# Util functions
from priority_tools import * # Implementation details and comments can be found in this file

# Plotting
import matplotlib
matplotlib.use('TkAgg') # To fix annoying "python is not installed as a framework" error
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def get_newest_csv(folder_path):
    '''
    Get the filename of csv file in folder_path that is the most recent.

    Taken from: https://stackoverflow.com/questions/39327032/how-to-get-the-latest-file-in-a-folder-using-python

    Args
    ----------
    folder_path (str): A valid path to the folder that contains the .csv file you
        want to search for.

    Returns
    ----------
    str: Path to the .csv file that was most recently changed in folder_path.
    '''
    folder_path += '*.csv'
    list_of_files = glob.glob(folder_path) # * means all if need specific format then *.csv
    return max(list_of_files, key=os.path.getctime)

def get_X_ranked_df(toi_path, tic_path, kp_file=r'data/known_planets/known_planets.csv', include_qlp=False, num_to_rank=5, dec_cut=-20, k_amp_cut=2):
    '''
    This function bascially replicates what the Priority-Tools-Tutorial notebook does
    but with the new X metric--the ratio of the TSM and the expected total exposure
    time on HIRES to achieve a 5-sigma mass.

    Args
    ----------
    toi_path (string): Path to the TOI+ list file to use.
    tic_path (string): Path to the TIC star information file to use.
    include_qlp (bool): Optionally consider QLPs in the prioritization ranking.
    num_to_rank (int): Optionally change the number of targest ranked per bin. Default
        is 3.

    Returns
    ----------
    DataFrame: DataFrame that contains useful columns, rankings from both
        prioritization methods. See prioritization_TSM_t_HIRES_ratio.ipynb for a
        tutorial.

    DataFrame: Subset of the first dataframe, but contains only the targets that
        changed ranking between the two methods.
    '''
    rp_key,ms_key,pp_key,Ts_key,ars_key,Jmag_key,\
    Vmag_key,rs_key,dec_key,Fp_key,mp_key,\
    mp_units,id_key = tess_colnames()

    # Load the TOI+ list
    toiplus = pd.read_csv(toi_path, delimiter=',',header=4) # Latest: data/toi/toi+-2020-02-20.csv

    # Load the TIC star info
    TIC_info = pd.read_csv(tic_path, comment='#')

    # Run the data cleaning function
    tess = clean_tess_data(toiplus, TIC_info, include_qlp=include_qlp, dec_cut=dec_cut, k_amp_cut=k_amp_cut)

    # Manual TOI info
    # tess = add_extra_info() # Add some information manually if it isn't correct in the TOI+ table e.g. TOI-509.01 and TOI-509.02

    # Load the known planets table, and merge it
    kps = pd.read_csv(kp_file)

    assert len(kps.index[pd.isnull(kps['Stellar Mass'])]) == 0, "Rows in known planets table with no stellar mass"
    assert len(kps.index[pd.isnull(kps['Star Radius Value'])]) == 0, "Rows in known planets table with no stellar radius"
    assert len(kps.index[pd.isnull(kps['Effective Temperature Value'])]) == 0, "Rows in known planets table with no Teff"
    assert len(kps.index[pd.isnull(kps['pl_masses'])]) == 0, "Rows in known planets table with no planet mass"

    # Check if these values are missing, if they are, fill them in...
    # for index, row in kps.iterrows():
    #     if np.isnan(row['K_amp']):
    #         try:
    #             row['K_amp'] = k_amp_finder(row['Stellar Mass'], row['Star Radius Value'], row['pl_masses'], row['Ars'])
    #         except:
    #             continue
    #     if np.isnan(row['TSM']):
    #         try:
    #             row['TSM'] = get_TSM(row['Planet Radius Value'], row['Star Radius Value'], row['Effective Temperature Value'], row['J mag'], row['pl_masses'], row['Ars'])
    #         except:
    #             continue


    df = tess.append(kps[np.logical_and(kps['K_amp'] > 1.5, kps['TSM'] > 10)],sort=False)


    n_counts = 250 # This might change in the future
    df['t_HIRES'] = t_HIRES_plavchan(df['V mag'], n_counts, df['K_amp'])
    df['X'] = df['TSM']/df['t_HIRES']

    # Defining the bins:
    # Log-uniform in radius and Fp, uniform in Teff. Right now,
    # you need 6 bin edges for radius and Fp and 4 for Teff;
    # we can change that by editing the "labels" in binning_function
    rad_bins = 10**(np.linspace(0,1,6))
    rad_bins[-1] = 11.2 #want to include up to Jupiter radius
    fpl_bins = 10**(np.linspace(-1,4,6))
    tef_bins = np.array([2500,3900,5200,6500])
    bins = [rad_bins, fpl_bins, tef_bins]

    # The two different ranking methods:
    # Sort things by TSM and then sort the top three in TSM by their Vmag
    binned_TSM_Vmag_df = binning_function(df, bins)
    binned_TSM_Vmag_df['priority'] = binned_TSM_Vmag_df['priority'].replace(0., np.nan) # Replace 0-rankings with nans to make filtering easier

    # Sort things by the ratio of TSM and the total exposure time needed to get a 5-sigma mass
    binned_X_df = binning_function_X(df, bins, num_to_rank=num_to_rank)
    binned_X_df['priority'] = binned_X_df['priority'].replace(0., np.nan) # Replace 0-rankings with nans to make filtering easier

    # Cut down on some of the extraneous columns
    useful_cols = ['Source Pipeline', 'Full TOI ID', rp_key, pp_key, Ts_key, Fp_key,
                   'Planet Equilibrium Temperature (K) Value', 'mass_flag', 'V mag','K_amp', 'TSM', 't_HIRES', 'X']
    compare_df = pd.DataFrame()
    compare_df[useful_cols] = binned_TSM_Vmag_df[useful_cols]
    compare_df['TSM_Vmag_priority'] = binned_TSM_Vmag_df['priority']
    compare_df = compare_df.merge(binned_X_df[useful_cols + ['priority']], on=useful_cols, left_index=True)
    compare_df = compare_df.rename(columns={'priority':'X_priority'})

    # Look at the rows where the two methods give different priority rankings
    compare_diff_df = compare_df[compare_df['TSM_Vmag_priority'] != compare_df['X_priority']]

    return compare_df, compare_diff_df

def get_target_list(save_fname=None, toi_folder='data/toi/', tic_folder='data/exofop/', selected_TOIs_folder='data/TKS/', include_qlp=False, verbose=True, num_to_rank=5, k_amp_cut=2., min_TSM=0.):
    '''
    Get a target list that incorporates information from selected_TOIs and Jump.

    The output of this function should be a good starting place for choosing targets
    for Ashley's prioritization algorithm.

    Args
    ----------
    save_fname (optional, str): Default=None. If give, save output to this filename.
    toi_folder (optional, str): Default='data/toi/'. Folder containing the TOI+ lists.
    tic_folder (optional, str): Default='data/exofop'. Folder containing the exofop
        TIC star information.
    selected_TOIs_folder (optional, str): Default='data/TKS/'. Folder containing
        downloaded versions of the selected_TOIs Google sheet.
    include_qlp (bool): Optionally consider QLPs in the prioritization ranking.
    verbose (optional, bool): Default=True. Print out additional information about
        the merging/filtering process while the call is executing.
    num_to_rank (int): Optionally change the number of targest ranked per bin. Default
        is 5.
    k_amp_cut (float): Optionally drop targets from the final list with an expected
        K-amp below this value. Default is 2, based on nominal 1 m/s HIRES RV precision.
    min_TSM (float): Optionally drop targets from the final list with TSM below
        this value. Default is 0, so no targets are dropped.

    Returns
    ----------
    DataFrame: Intersection of DataFrame from get_X_ranked_df() and selected_TOIs,
        with some filtering.
    '''

    # Get the latest versions of the data
    toi_path = get_newest_csv(toi_folder)
    tic_path = get_newest_csv(tic_folder)
    selected_TOIs_path = get_newest_csv(selected_TOIs_folder)

    # Check to make sure these are the files that you want to use
    if verbose:
        print('TOI+ list file used: \t {}'.format(toi_path))
        print('TIC ExoFOP file used: \t {}'.format(tic_path))
        print('selected_TOIs file used: {}'.format(selected_TOIs_path))
        print('')

    # Generate the binned data frame with the X rankings.
    print('Binning targets...')
    X_df, __________ = get_X_ranked_df(toi_path, tic_path, include_qlp=include_qlp, num_to_rank=num_to_rank, k_amp_cut=k_amp_cut) # Don't really need the second dataframe that's returned
    selected_TOIs_df = pd.read_csv(selected_TOIs_path)
    print("The X_df dataframe has {} rows.".format(len(X_df)))
    print("The selected_TOIs_df dataframe has {} rows.".format(len(selected_TOIs_df)))
    print('')

    # Merge X_df with selected_TOIs
    X_tois_df = merge_with_selected_TOIs(X_df, selected_TOIs_df, verbose=verbose, num_to_rank=num_to_rank, min_TSM=min_TSM)

    # TODO: Decide what information from Jump will be relevant
    # Incorporate information from Jump
    # X_tois_df = use_jump_info(X_tois_df)

    # Save the output if given a filename.
    if save_fname is not None:
        X_tois_df.to_csv(save_fname)
        print('The target list was saved to {}.'.format(save_fname))

    return X_tois_df

def merge_with_selected_TOIs(X_df, selected_TOIs_df, verbose=True, num_to_rank=5, min_TSM=0.):
    '''
    Merge the binned, ranked dataframe of targets with those in selected_TOIs. Filter
    out targets from the merged dataframe that failed vetting in selected_TOIs,
    are known planets, or otherwise have 'hires_prv' = 'no'.

    Args
    ----------
    X_df (DataFrame): DataFrame containing binned, ranked targets using the X metric.
        See get_X_ranked_df() for details.
    selected_TOIs_df (DataFrame): DataFrame of the selected_TOIs Google spreadsheet.
    num_to_rank (int): Optionally change the number of targest ranked per bin. Default
        is 5.
    min_TSM (float): Optionally drop targets from the final list with TSM below
        this value. Default is 0, so no targets are dropped.

    Returns
    ----------
    DataFrame: X_tois_df, the result of cross referencing our binned, ranked targets
        with those in selected_TOIs.
    '''

    selected_TOIs_IDs = selected_TOIs_df['toi'].values
    X_tois_df = X_df[X_df['Full TOI ID'].isin(selected_TOIs_IDs)] # Get the intersection of X_df and selected_TOIs_df
    if verbose:
        print("The intersection of X_df and the selected_TOIs_df has {} rows...".format(len(X_tois_df)))

    # Use these columns to filter out systems that failed vetting in some way/are no longer being observed/are known planets
    # The cps column will also be useful for comparing the targets to Jump database tables.
    selected_TOIs_cols = ['toi', 'tic', 'cps', 'disp', 'vetting', 'ao_vet', 'hires_prv', 'apf_prv']

    # This long line merges X_tois_df and selected_TOIs_df with helpful filtering columns from selected_TOIs,
    # while preserving the indices in X_tois_df, which contain binning information
    X_tois_df = X_tois_df.reset_index().merge(selected_TOIs_df[selected_TOIs_cols],
                                              how='left', left_on='Full TOI ID', right_on='toi').set_index(X_tois_df.index.names)

    # Filter out targets that are known planets, fail spectroscopic vetting, or otherwise have "no" for their hires_prv column.
    X_tois_df = X_tois_df[X_tois_df['disp'] != 'KP']
    if verbose:
        print("After filtering out targets that are known planets, {} rows remain...".format(len(X_tois_df)))
    X_tois_df = X_tois_df[X_tois_df['vetting'].isin(['passed', 'do observe'])]
    if verbose:
        print("After filtering out targets that failed spectroscopic vetting, {} rows remain...".format(len(X_tois_df)))
    X_tois_df = X_tois_df[X_tois_df['ao_vet'] != 'failed']
    if verbose:
        print("After filtering out targets that failed AO vetting, {} rows remain...".format(len(X_tois_df)))
    X_tois_df = X_tois_df[X_tois_df['hires_prv'] != 'no']
    if verbose:
        print("After filtering out targets whose hires_prv plan is 'no', {} rows remain.".format(len(X_tois_df)))

    # Drop low TSM targets
    low_TSM_df = X_tois_df[X_tois_df['TSM'] < min_TSM]
    print(f'Dropping {len(low_TSM_df)} targets with TSM below {min_TSM}...')
    low_TSM_tois = low_TSM_df['Full TOI ID'].values
    print(f'Targets dropped for low TSM values: {low_TSM_tois}')
    X_tois_df = X_tois_df[X_tois_df['TSM'] >= min_TSM]

    # Of the targets remaining, how many actually have a 1, 2, 3, etc. priority ranking in their bin?
    print('')
    num_list = [len(X_tois_df[X_tois_df['X_priority'] == i]) for i in range(1, num_to_rank+1)]
    tot_num_p = sum(num_list)
    sys.stdout.write("Of the {} targets with priorities, ".format(tot_num_p))
    for i in range(1,num_to_rank+1):
        sys.stdout.write('{} are Priority {}'.format(num_list[i-1], i))
        if i in range(1,num_to_rank):
            sys.stdout.write(', ')
        else:
            sys.stdout.write('.')
    print('')

    return X_tois_df

# def use_jump_info(X_tois_df):
#     '''
#     Once we have a list of targets that survived the initial prioritization, update
#     their X metrics using information from Jump.
#     '''
#
#     return None


def summary_plot(sc3_df, pick_tois_df, benchmark_targets=None, id_key='Full TOI ID', hist_bin_num=10):
    '''
    Create a summary plot of where the ranked targets in sc3_df fall in V magnitude,
    K-amplitude, planet radius, and priority. Optionally include the IDs of systems
    that are benchmarks for reference.

    Args
    ----------
    sc3_df (DataFrame): DataFrame like the output of get_X_ranked_df(), containing targets
        with an associated X metric.
    pick_tois_df (DataFrame): DataFrame containing information on all of the targets that were
        picked by the selection algorithm.
    benchmark_targets (optional, array_like): Default=None. A list of target id_keys
        that will be highlighted in the scatter plots as reference points.
    id_key (optional, str): Default='Full TOI ID'. The column to use to identify
        the targets in sc3_df.
    hist_bin_num (optional, int): Default=10. The number of histogram to use for
        the histograms.

    Returns
    ----------
    figure: A matplotlib figure containing the summary plot.
    axes: An array of the figure's subplot axes.
    '''

    # def plot_benchmark_targets(planet_list, col='blue'):
    #     for i,planet in enumerate(planet_list):
    #         curr_row = sc3_df[sc3_df[id_key] == planet]
    #
    #         if type(planet) != str:
    #             planet = str(planet)
    #
    #         if i == 0:
    #             ax_vmag.plot(curr_row['V mag'].values, curr_row['X'].values, '.', color=col, alpha=0.7)
    #         else:
    #             ax_vmag.plot(curr_row['V mag'].values, curr_row['X'].values, '.', color=col, alpha=0.7)
    #         ax_vmag.text(curr_row['V mag'].values[0] + 0.1, curr_row['X'].values[0] * 1.25, planet, fontsize=12)
    #
    #         ax_kamp.plot(curr_row['K_amp'].values, curr_row['X'].values, '.', color=col, alpha=0.7)
    #         ax_kamp.text(curr_row['K_amp'].values[0] * 1.025, curr_row['X'].values[0] * 1.25, planet, fontsize=12)
    #
    #         ax_rad.plot(curr_row['Planet Radius Value'].values, curr_row['X'].values, '.', color=col, alpha=0.7)
    #         ax_rad.text(curr_row['Planet Radius Value'].values[0] * 1.01,
    #                     curr_row['X'].values[0] * 1.25, planet, fontsize=12)

    fig, axes = plt.subplots(figsize=(18,10), ncols=3, nrows=2, sharey='row', sharex='col')
    ax_vmag, ax_kamp, ax_rad, ax_p_vmag, ax_p_kamp, ax_p_rad = axes.flatten()

    # Map priority level to a color
    colors = plt.cm.viridis(np.linspace(0., 1, 6))

    for ax, x_axis_key in zip([ax_vmag, ax_kamp, ax_rad], ['V mag', 'K_amp', 'Planet Radius Value']):
        for index, row in sc3_df.iterrows():

            priority = None
            try:
                priority = int(row['X_priority'])
            except:
                priority = 6
            color = colors[priority - 1]

            ax.plot(row[x_axis_key], row['X'], '.', alpha=0.7, color=color)
            # ax.annotate(row['Full TOI ID'],
            #             (row[x_axis_key], row['X']),
            #              textcoords='offset points',
            #              xytext=(-5,5),
            #              ha='left',
            #              clip_on=True)
        ax.plot(row[x_axis_key], row['X'], '.', alpha=0.7, label='Picked by SC3') # Plot the last value again just to get the label

    for ax, x_axis_key in zip([ax_vmag, ax_kamp, ax_rad], ['vmag', 'k', 'rp']):
        ax.plot(pick_tois_df[x_axis_key], pick_tois_df['X'], '.', alpha=0.3, color='gray', label='Full Sample')
    # ax_vmag.plot(picks_df['vmag'], picks_df['X'], '.', alpha=0.5, color='gray', label='Full Sample') # Need to compute X metric for all of the TOIs if we want to do this

    vmag_low, vmag_high = ax_vmag.get_xlim()
    # ax_vmag.plot(0, 0, '.', label='Picked by SC3') # Foo value to use just for labeling
    ax_vmag.set_xlim([vmag_high, vmag_low]) # Invert x axis
    ax_vmag.set_xlabel('$V$ [mag]', fontsize=14)

    # Plot as a function of K-amplitude
    # ax_kamp.plot(sc3_df['K_amp'], sc3_df['X'], '.', alpha=0.7)
    # ax_kamp.plot(picks_df['K_amp'], picks_df['X'], '.', alpha=0.5, color='gray')
    ax_kamp.set_xlabel('$K$ [m s$^{-1}$]', fontsize=14)

    # Plot as a function of radius (though I think this is the same information as K-amplitude)
    # ax_rad.plot(sc3_df['Planet Radius Value'], sc3_df['X'], '.', alpha=0.7)
    # ax_rad.plot(picks_df['Planet Radius Value'], picks_df['X'], '.', alpha=0.5, color='gray')
    ax_rad.set_xlabel(r'$R$ [$R_\oplus$]', fontsize=14)

    # # Mark some notable planets for context
    # if benchmark_targets is not None:
    #     assert all([target in sc3_df[id_key].values for target in benchmark_targets]), \
    #     'One of the benchmark targets is not contained in the DataFrame.'
    #     plot_benchmark_targets(benchmark_targets)

    ax_vmag.legend(fancybox=True, fontsize=12, loc='upper left')

    # Plot histograms of p1, p2, p3, p4, p5 targets
    hist_axes = [ax_p_vmag, ax_p_kamp, ax_p_rad]
    hist_keys = ['V mag', 'K_amp', 'Planet Radius Value']
    for ax, key in zip(hist_axes, hist_keys):
        bins = None
        if key == 'V mag':
            bins = np.linspace(sc3_df[key].min(), sc3_df[key].max(), hist_bin_num)
        elif key == 'K_amp':
            bins = np.logspace(0, np.log10(50), hist_bin_num)
        elif key == 'Planet Radius Value':
            bins = np.logspace(0, np.log10(sc3_df[key].max()), hist_bin_num)

        for i in range(1, 6):
            histtype = None
            linestyle = '-'
            hatch = None
            alpha = 0.7
            if i == 1 or i == 2 or i == 3:
                histtype='stepfilled'
            elif i == 4:
                histtype='step'
                linestyle='--'
                alpha = 1.
            elif i == 5:
                histtype='step'
                hatch='/'
                alpha = 1.
            ax.hist(sc3_df[sc3_df['X_priority'] == i][key].dropna().values,
            bins=bins, histtype=histtype, alpha=alpha, color=colors[i-1], linestyle=linestyle, hatch=hatch, label='Priority {}'.format(i))
        if key == 'Vmag':
            ax.legend(fancybox=True, fontsize=12)

    ax_p_vmag.set_ylabel('N', fontsize=14)
    ax_p_vmag.set_xlabel('$V$ [mag]', fontsize=14)
    ax_p_kamp.set_xlabel('$K$ [m s$^{-1}$]', fontsize=14)
    ax_p_rad.set_xlabel(r'$R$ [$R_\oplus$]', fontsize=14)

    # Log scales needed to see structure
    ax_vmag.set_yscale('log')
    ax_kamp.set_xscale('log')
    ax_rad.set_xscale('log')

    # Add label to label that plots X metric
    ax_vmag.set_ylabel(r'X [arbitrary units]', fontsize=14)

    # Make axis labels visible for top row
    for ax in axes.flatten():
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.tick_params(axis='both', labelsize=14)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    return fig, axes, colors
