'''
This script will handle the construction of the known planets data frame that will
be used when binning and ranking the TOI targets.

We're looking for known, transiting planets with a reliable mass measurement and
J-mag < 12 (JWST limit) or Ks-mag < 14. (When J-mag is not available, use
Ks as a proxy, and add a generous 2 magnitude buffer to the cutoff. Ks is the default
Near IR magnitude that is reported in the composite table on Exoplanet archive.
'''

import argparse

import numpy as np
import pandas as pd

from priority_tools import star_mass_from_logg_r, ars_from_t, get_TSM, get_insol_flux, k_amp_finder
from X_ranking import get_newest_csv

# Command line arguments. For most uses, the defaults will be fine.
parser = argparse.ArgumentParser(description='Create the known planets .csv files with a simple script.')

parser.add_argument('save_fname', type=str, default=None, help='Save name for the resulting .csv file.')
parser.add_argument('--output_dir', type=str, default='data/known_planets/', help='Folder to save output file to.')
parser.add_argument('--kp_folder', type=str, default='data/known_planets/', help='Folder with the known planet \
data from the Exoplanet Archive.')
parser.add_argument('--composite_fname', type=str, default='composite.csv', help='Name of the .csv file containing \
Exoplanet Archive data from the composite table, subject to the filertering described in README.md')
parser.add_argument('--confirmed_fname', type=str, default='confirmed_jmag.csv', help='Name of the .csv file containing \
Exoplanet Archive data from the confirmed table, usually just the Planet Name and J magnitude.')

# Note: Need to confirm with Nicholas and/or what goes in to creating these tables and what filters we use for them.
# Excluding these for now, since none of them would have masses anyway.
# parser.add_argument('--koi_fname', type=str, default='koi_above_v14.csv', help='Name of the .csv file containing \
# Exoplanet Archive data from the KOI table with dispositions of CANDIDATE.')
# parser.add_argument('--k2_fname', type=str, default='k2_above_v14.csv', help='Name of the .csv file containing \
# Exoplanet Archive data from the K2 table.')

def kp_match_mags(composite_df, confirmed_df):
    '''
    Using information from the confirmed data table, if we can replace the Ks mags
    from the composite table with J mags from the confirmed table, do so. If not,
    leave the Ks mags. Make a note in a new column to say which mag is being used,
    since it goes into calculating the TSM values.

    Do the same for V mags from the confirmed data table and the optical magnitudes 
    that are included in the composite table (most of which would be V in any case).

    Args
    ----------
    composite_df (DataFrame): Data from Exoplanet Archive composite table, subject
        to filtering as described in README.md.
    confirmed_df (DataFrame): Data from Exoplanet Archive confirmed table, but is
        basically just the host star names and J mag values, excluding those with
        null J mag values on Exoplanet Archive.

    Returns
    ----------
    DataFrame: The known planets data from Exoplanet Archive's composite table,
    now supplemented with the J mag values from the confirmed table, and in a few
    cases e.g. WASP-91, WASP-105 (see data/known_planets/simbad_jmag_notes.txt)
    J mag values from Simbad.

    Also add in V mag values from the confirmed table.
    '''

    kp_df = composite_df.merge(confirmed_df, how='left', left_on='fpl_name', right_on='pl_name')

    # From testing, there are only 13 stars in the composite table that don't have
    # a J-mag in the confirmed table. Of these, all of them also don't have a K-mag.
    # I looked for J-mags manually on Simbad for these 13 stars and stored their values in data/known_planets/simbad_jmag.csv
    # We'll load in the values that we found.
    # For the remaining systems that we can't find a J mag for, we'll just drop them.

    no_j_mag_inds = kp_df.index[pd.isnull(kp_df['st_j'])]
    no_j_mag_and_no_k_mag_inds = kp_df.index[np.logical_and(pd.isnull(kp_df['fst_nirmag']), pd.isnull(kp_df['st_j']))]
    assert len(no_j_mag_inds) == len(no_j_mag_and_no_k_mag_inds), "There are some stars that have a K-mag but no J-mag"

    # This filename is hardcoded in
    simbad_jmag_df = pd.read_csv('data/known_planets/simbad_jmag.csv')

    # This will fill in the miss J mag values. Hints taken from here: https://stackoverflow.com/questions/56842140/pandas-merge-dataframes-with-shared-column-fillna-in-left-with-right
    kp_df['j_mag'] = kp_df['st_j'].copy() # This is ok because if a star doesn't have a J mag, we know by the assert statement above that it doesn't have a K mag.
    kp_df = kp_df.merge(simbad_jmag_df, how='left', left_on=['fpl_hostname'], right_on=['fpl_hostname'])
    kp_df['j_mag'] = kp_df['j_mag_x'].fillna(kp_df['j_mag_y'])
    kp_df = kp_df.drop(columns=['j_mag_x', 'j_mag_y'], axis=1)

    # Now drop the rows that don't have a J mag, at least as far as we can tell
    # This drops 6 systems:
    # HD 202772 A, OGLE-TR-056, OGLE-TR-10, OGLE-TR-113, OGLE-TR-132, SWEEPS-11
    # See simbad_jmag_notes.txt for more info
    kp_df = kp_df.drop(index=kp_df.index[pd.isnull(kp_df['j_mag'])]).reset_index(drop=True)

    return kp_df

def fill_in_missing_info(kp_df):
    '''
    Some rows still have missing (and necessary) information:
    e.g. Kepler 60 b, c, d, WASP-157 b, and WASP-85 A b.
    I'm going to look for this information manually on Exoplanet Archive, enter
    it into a .csv file, a load it manually, similar to what we did for the missing
    J mags.
    '''
    missing_info_df = pd.read_csv('data/known_planets/missing_info.csv')

    kp_df = kp_df.merge(missing_info_df, how='left', left_on='fpl_name', right_on='fpl_name')

    # Fill in the missing information
    kp_df['fst_mass'] = kp_df['fst_mass_x'].fillna(kp_df['fst_mass_y'])
    kp_df = kp_df.drop(columns=['fst_mass_x', 'fst_mass_y'], axis=1)

    kp_df['fst_rad'] = kp_df['fst_rad_x'].fillna(kp_df['fst_rad_y'])
    kp_df = kp_df.drop(columns=['fst_rad_x', 'fst_rad_y'], axis=1)

    kp_df['fst_teff'] = kp_df['fst_teff_x'].fillna(kp_df['fst_teff_y'])
    kp_df = kp_df.drop(columns=['fst_teff_x', 'fst_teff_y'], axis=1)

    kp_df['fpl_bmasse'] = kp_df['fpl_bmasse_x'].fillna(kp_df['fpl_bmasse_y'])
    kp_df = kp_df.drop(columns=['fpl_bmasse_x', 'fpl_bmasse_y'], axis=1)

    # After this, it looks like Kepler-60 d (which only has an upper limit on a mass)
    # is still missing information. For any other rows that are missing a stellar mass,
    # radius, Teff, or planet mass, drop them.
    kp_df = kp_df.drop(index=kp_df.index[pd.isnull(kp_df['fst_mass'])]).reset_index(drop=True)
    assert len(kp_df.index[pd.isnull(kp_df['fst_mass'])]) == 0, "Still rows with no stellar mass"
    assert len(kp_df.index[pd.isnull(kp_df['fst_rad'])]) == 0, "Still rows with no stellar radius"
    assert len(kp_df.index[pd.isnull(kp_df['fst_teff'])]) == 0, "Still rows with no Teff"
    assert len(kp_df.index[pd.isnull(kp_df['fpl_bmasse'])]) == 0, "Still rows with no planet mass"

    return kp_df

def add_insol_flux(kp_df):
    '''
    Almost half of all insolation flux column values are missing, so fill them in
    where needed.
    '''
    flux_df = pd.DataFrame(data={'fpl_name':kp_df['fpl_name'].copy(), 'Effective Stellar Flux Value':get_insol_flux(kp_df['Ars'], kp_df['fst_rad'], kp_df['fst_teff'])})

    # Fill in insolation flux values where missing
    kp_df = kp_df.merge(flux_df, how='left', left_on='fpl_name', right_on='fpl_name')
    kp_df['fpl_insol'] = kp_df['fpl_insol'].fillna(kp_df['Effective Stellar Flux Value'])
    kp_df = kp_df.drop(columns=['Effective Stellar Flux Value'], axis=1)

    return kp_df

# def rename_and_cut_columns(kp_df):
#     '''
#     Rename columns to match the column names in the TOI+ list. Cut down to only
#     the columns that we need.
#     '''
#
#     output_df = pd.DataFrame(data={
#     'Planet Radius Value':kp_df['fpl_rade'],
#     'Stellar Mass':kp_df['fst_mass'],
#     'Orbital Period Value':kp_df['fpl_orbper'],
#     'Star Radius Value':kp_df['fst_rad'],
#     'Effective Temperature Value':kp_df['fst_teff'],
#     'J mag':kp_df['j_mag'],
#     # 'V mag':kp_df['fst_optmag'], # Some of these are V mags and some are G. We don't use them for the known planets, so don't include for now.
#     'TIC Declination':kp_df['dec'],
#     'Effective Stellar Flux Value':kp_df['fpl_insol'],
#     'pl_masses':kp_df['fpl_bmasse'], # Can probably change the name of this column to something other than pl_masses?
#     'mass_flag':kp_df['mass_flag'],  # The mass flag should be unneccessary at some point
#     'Ars':kp_df['Ars'],
#     'K_amp':kp_df['K_amp'],
#     'TSM':kp_df['TSM'],
#     'Full TOI ID':kp_df['fpl_name']
#     })
#
#     return output_df

def rename_columns(kp_df):
    '''
    Rename columns to match the column names in the TOI+ list.
    '''
    
    output_df = kp_df.rename(mapper={
    'fpl_rade':'Planet Radius Value',
    'fst_mass':'Stellar Mass',
    'fpl_orbper':'Orbital Period Value',
    'fst_rad':'Star Radius Value',
    'fst_teff':'Effective Temperature Value',
    'j_mag':'J mag',
    'fst_optmag':'Opt mag',
    'dec':'TIC Declination',
    'fpl_insol':'Effective Stellar Flux Value',
    'fpl_bmasse':'pl_masses', # Can probably change the name of this column to something other than pl_masses?
    'mass_flag':'mass_flag',  # The mass flag should be unneccessary at some point
    'Ars':'Ars',
    'K_amp':'K_amp',
    'TSM':'TSM',
    'fpl_name':'Full TOI ID'
    }, axis=1)

    # Uncommment these lines to prepare a known_planets.csv file for the TKS target selection algorithm
    # These lines rename columns to match the column headers in TOIs_perfect and remove extraneous columns.
    
    # output_df = kp_df.rename(mapper={
    # 'fpl_rade':'rp',
    # 'fst_mass':'Stellar Mass',
    # 'fpl_orbper':'period',
    # 'fst_rad':'r_s',
    # 'fst_teff':'teff',
    # 'j_mag':'jmag',
    # 'fst_optmag':'Opt mag',
    # 'dec':'TIC Declination',
    # 'fpl_insol':'sinc',
    # 'fpl_bmasse':'mp', 
    # 'mass_flag':'mass_flag',  # The mass flag should be unneccessary at some point
    # 'Ars':'a_to_R',
    # 'K_amp':'K_amp',
    # 'TSM':'TSM',
    # 'fpl_name':'fpl_name'
    # }, axis=1)

    # output_df = output_df.loc[pd.notnull(output_df['TSM'])]
    # return output_df[['fpl_name', 'rp', 'teff', 'sinc', 'mp', 'period', 'TSM']]
    
    return output_df


def get_kp_df(kp_folder, composite_fname='composite.csv', confirmed_fname='confirmed_jmag.csv'):
    # Excluding KOI and K2 candidates for now because they don't have masses anyway.
    # koi_fname='koi_above_v14.csv',
    # k2_fname='k2_above_v14.csv'):

    '''
    Return a dataframe containing information on known planets from the Exoplanet
    Archive. Use information from both the composite and confirmed table. The confirmed
    table information is just to fill in J-magnitudes where we can, since the Near
    IR magnitude available in the composite table is Ks. If no J-mag exists for
    an entry, just leave their K-mag but make a note of it somewhere.

    Args
    ----------
    kp_folder (str): Path to folder containing Exoplanet Archive data. This folder
    is assumed to have files named composite.csv and confirmed_jmag.csv by default
    '''
    # Read in data from composite and confirmed table
    composite_df = pd.read_csv(kp_folder + composite_fname, comment="#")
    confirmed_df = pd.read_csv(kp_folder + confirmed_fname, comment="#")

    kp_df = kp_match_mags(composite_df, confirmed_df)

    # 5 rows still have missing (and necessary) information for some reason:
    # Kepler 60 b, c, d, WASP-157 b, and WASP-85 A b.
    # I'm going to look for this information manually on Exoplanet Archive, enter
    # it into a .csv file, a load it manually, similar to what we did for the missing
    # J mags.
    kp_df = fill_in_missing_info(kp_df)

    # All of the masses in the kp_df will be actual reliable measurements, based
    # on how we filtered the table when we downloaded it.
    kp_df['mass_flag'] = np.ones(len(kp_df)) # Mass flag = 1 means reliable mass

    # Need the ratio of the planet semi-major axis to the star's radius to calculate TSM
    # as well as the K-amplitude of the RV signal
    kp_df['Ars'] = ars_from_t(kp_df['fpl_orbper'], kp_df['fst_mass'], kp_df['fst_rad'])
    kp_df['K_amp'] = k_amp_finder(kp_df['fst_mass'], kp_df['fst_rad'], kp_df['fpl_bmasse'], kp_df['Ars'])

    # Calculate the TSM
    assert len(kp_df.index[pd.isnull(kp_df['fst_teff'])]) == 0, "Some stars in the known planets dataframe are missing T_eff."
    kp_df['TSM'] = get_TSM(kp_df['fpl_rade'], kp_df['fst_rad'], kp_df['fst_teff'], kp_df['j_mag'], kp_df['fpl_bmasse'], kp_df['Ars'])

    # Get insolation fluxes where missing
    kp_df = add_insol_flux(kp_df)

    # Rename columns to match those in the TOI+ table, cut extraneous columns
    # kp_df = rename_and_cut_columns(kp_df)
    kp_df = rename_columns(kp_df) # No need to cut the extra info. - Joey, 05/04/20

    # Read in data from KOI and K2 candidates
    # TODO: Ask Natalie: Do we really need to use KOI and K2 candidates? They don't
    # have masses, so we wouldn't include them in a transmission spectroscopy survey, right?
    # Pass on including them for now.
    # koi_df = pd.read_csv(kp_folder + koi_fname, comment="#")
    # k2_df  = pd.read_csv(kp_folder + k2_fname, comment="#")
    # k2_df["Star Mass Value"] = star_mass_from_logg_r(k2_df['st_logg'], k2_df['st_rad'])

    return kp_df

if __name__ == '__main__':

    args = parser.parse_args()
    save_fname = args.save_fname

    kp_folder  = args.kp_folder
    composite_fname = args.composite_fname
    confirmed_fname = args.confirmed_fname

    # Excluding KOI and K2 candidates for now
    # koi_fname = args.koi_fname
    # k2_fname  = args.k2_fname

    output_dir = args.output_dir

    # Load data from the Exoplanet Archive using composite and confirmed tables
    kp_df = get_kp_df(kp_folder, composite_fname=composite_fname, confirmed_fname=confirmed_fname)

    # Save the output to the specified file
    kp_df.to_csv(output_dir + save_fname, index=False)

    print(f"The known planets table has {len(kp_df)} rows.")
    print(f"The table was saved to {output_dir + save_fname}.")
