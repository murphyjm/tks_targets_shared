'''
This combines multiple cells from the prioritization_TSM_t_HIRES_ratio notebook into a single script,
so that it outputs a simple dataframe with rankings for both methods.
'''
# Basic analysis
import numpy as np
import pandas as pd

# Util functions
from priority_tools import * # Implementation details and comments can be found in this file

def get_X_ranked_df(toi_path, tic_path):
    '''
    Args
    ----------
    toi_path (string): Path to the TOI+ list file to use.
    tic_path (string): Path to the TIC star information file to use.

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
    # NOTE: This pd.read_csv() call is being finicky for some reason.... - Joey, 03/23/20
    #TIC_info = pd.read_csv(tic_path, delimiter=',', comment='#') # Latest: data/exofop/TIC_star_info_2020-02-20.csv
    TIC_info = pd.read_csv(tic_path, delimiter=',', comment='#', header=1, usecols=[0,1,2,3,4,5,6])

    # Run the data cleaning function
    tess = clean_tess_data(toiplus, TIC_info, include_qlp=False)

    # Load the known planets (and Kepler PCs) table, and merge it
    kps = pd.read_csv(r'data/kp-k14_pc-v14.csv')
    df = tess.append(kps[np.logical_and(kps['K_amp'] > 1.5, kps['TSM'] > 10)],sort=False)

    n_counts = 250 # This might change in the future
    df['t_HIRES'] = t_HIRES(df['V mag'], n_counts, df['K_amp'])
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

    # Sort things by the ratio of TSM and the total exposure time needed to get a 5-sigma mass
    binned_X_df = binning_function_X(df, bins)

    # Cut down on some of the extraneous columns
    useful_cols = ['Full TOI ID', rp_key, pp_key, Ts_key, Fp_key,
                   'Planet Equilibrium Temperature (K) Value', 'V mag','K_amp', 'TSM', 't_HIRES', 'X']
    compare_df = pd.DataFrame()
    compare_df[useful_cols] = binned_TSM_Vmag_df[useful_cols]
    compare_df['TSM_Vmag_priority'] = binned_TSM_Vmag_df['priority']
    compare_df = compare_df.merge(binned_X_df[useful_cols + ['priority']], on=useful_cols, left_index=True)
    compare_df = compare_df.rename(columns={'priority':'X_priority'})

    # Look at the rows where the two methods give different priority rankings
    compare_diff_df = compare_df[compare_df['TSM_Vmag_priority'] != compare_df['X_priority']]

    return compare_df, compare_diff_df

def summary_plot(df, benchmark_targets=None, id_key='Full TOI ID', hist_bin_num=10):
    '''
    Create a summary plot of where the targets in df fall in V magnitude, K-amplitude,
    planet radius, and priority. Optionally include the IDs of systems that are
    benchmarks for reference.
    '''

    def plot_benchmark_targets(planet_list):
        for planet in planet_list:
            curr_row = df[df[id_key] == planet]

            if type(planet) != str:
                planet = str(planet)

            ax_vmag.plot(curr_row['V mag'].values, curr_row['X'].values, '.', color='red', alpha=0.7)
            ax_vmag.text(curr_row['V mag'].values[0] + 0.1, curr_row['X'].values[0] * 1.5, planet, fontsize=12)

            ax_kamp.plot(curr_row['K_amp'].values, curr_row['X'].values, '.', color='red', alpha=0.7)
            ax_kamp.text(curr_row['K_amp'].values[0] * 1.05, curr_row['X'].values[0] * 1.5, planet, fontsize=12)

            ax_rad.plot(curr_row['Planet Radius Value'].values, curr_row['X'].values, '.', color='red', alpha=0.7)
            ax_rad.text(curr_row['Planet Radius Value'].values[0] * 1.025,
                        curr_row['X'].values[0] * 1.5, planet, fontsize=12)

    fig, axes = plt.subplots(figsize=(18,10), ncols=3, nrows=2, sharey='row', sharex='col')
    ax_vmag, ax_kamp, ax_rad, ax_p_vmag, ax_p_kamp, ax_p_rad = axes.flatten()

    # Plot as a function of V magnitude
    ax_vmag.plot(df['V mag'], df['X'], '.', alpha=0.3)
    vmag_low, vmag_high = ax_vmag.get_xlim()
    ax_vmag.set_xlim([vmag_high, vmag_low]) # Invert x axis
    ax_vmag.set_xlabel('$V$ [mag]', fontsize=14)

    # Plot as a function of K-amplitude
    ax_kamp.plot(df['K_amp'], df['X'], '.', alpha=0.3)
    ax_kamp.set_xlabel('$K$ [m s$^{-1}$]', fontsize=14)

    # Plot as a function of radius (though I think this is the same information as K-amplitude)
    ax_rad.plot(df['Planet Radius Value'], df['X'], '.', alpha=0.3)
    ax_rad.set_xlabel(r'$R$ [$R_\oplus$]', fontsize=14)

    # Mark some notable planets for context
    if benchmark_targets is not None:
        assert all([target in df[id_key].values for target in benchmark_targets]), \
        'One of the benchmark targets is not contained in the DataFrame.'
        plot_benchmark_targets(benchmark_targets)

    # Plot histograms of p1, p2, and p3 targets
    hist_axes = [ax_p_vmag, ax_p_kamp, ax_p_rad]
    hist_keys = ['V mag', 'K_amp', 'Planet Radius Value']
    for ax, key in zip(hist_axes, hist_keys):
        bins = None
        if key == 'V mag':
            bins = np.linspace(df[key].min(), df[key].max(), hist_bin_num)
        else:
            bins = np.logspace(np.log10(df[key].min()), np.log10(df[key].max()), hist_bin_num)
        colors = ['blue', 'black', 'red']
        for i in range(1, 4):
            histtype = None
            if i == 1 or i == 2:
                histtype='stepfilled'
            elif i == 3:
                histtype='step'
            ax.hist(df[df['X_priority'] == i][key].dropna().values,
            bins=bins, histtype=histtype, alpha=0.7, color=colors[i-1], label='Priority {}'.format(i))
        ax.legend(fancybox=True, fontsize=12)

    ax_p_vmag.set_ylabel('N', fontsize=14)
    ax_p_vmag.set_xlabel('$V$ [mag]', fontsize=14)
    ax_p_kamp.set_xlabel('$K$ [m s$^{-1}$]', fontsize=14)
    ax_p_rad.set_xlabel(r'$R$ [$R_\oplus$]', fontsize=14)

    # Log scales needed to see structure
    ax_vmag.set_yscale('log')
    ax_kamp.set_xscale('log')
    ax_rad.set_xscale('log')

    # Plot housekeeping
    ax_vmag.set_ylabel(r'X [arbitrary units]', fontsize=14)

    for ax in axes.flatten():
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.tick_params(axis='both', labelsize=14)

    return fig, axes
