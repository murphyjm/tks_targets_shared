# Command line input
import argparse

# Files
import os
import time

# Analysis
from utils import *
from binning import bin
import numpy as np

def main():
    """
    TKS prioritization code.

    Example command line call:
        python prioritize.py toi+-2019-10-29.csv exofop_search2019-10-29_combined.csv --toi_col_dict toi_col_dict_foo.json
    """
    # Handle the command line input
    parser = argparse.ArgumentParser(description = "Prioritize TKS targets for atmospheric follow-up.")
    parser.add_argument("toi_fname", metavar="TOI List filename", type=str,
                        help="Fileneame for the toi list .csv file. Should be stored in path data/toi/")
    parser.add_argument("exofop_fname", metavar="Exofop fileneame", type=str,
                        help="Filename for the exofop .csv file. Should be stored in the path data/exofop/")
    parser.add_argument("--toi_col_dict", default='toi_col_dict.json',
                        help="The .json file with a dict to convert between standard column names and the (sometimes changing TOI column names).")
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    parser.add_argument("--output_fname", default=os.path.join("my_tois", 'my_tois_{}.txt'.format(timestr)),
                        help="Output (.txt) filename where toi list is stored.")
    args = parser.parse_args()

    toi_fname = os.path.join("data/toi", args.toi_fname)
    exo_fname = os.path.join("data/exofop", args.exofop_fname)
    planet_df = load_and_merge(toi_fname, exo_fname)
    toi_col_dict = load_toi_col_names(args.toi_col_dict)

    # Clean up the resulting df a little bit
    planet_df = planet_df.drop_duplicates(subset="Full TOI ID").sort_values("Full TOI ID")
    # Add a column for stellar mass calculated from surface gravity
    # N.B. the Exofop data contains stellar mass, but for rows that are not matched
    # with Exofop data, give them a stellar mass.
    planet_df[toi_col_dict["ms_key"]] = (10**planet_df["Surface Gravity Value"] * (planet_df[toi_col_dict["rs_key"]] * const.R_sun) / const.G) / const.M_sun

    # Remove rows from the df that don't meet these three criteria
    planet_df = planet_df[np.logical_and.reduce((planet_df[toi_col_dict["rp_key"]] > 0,
                                                 planet_df[toi_col_dict["pp_key"]] > 0,
                                                 planet_df["Source Pipeline"] == "spoc"))]

    # Add a column for the ratio of the planet's semi-major orbital axis and the radius of the host star
    # This column is needed for calculating the TSM (but don't need it if calculating equilibrium temperature via insolation flux)
    planet_df["ar_ratio"] = ar_ratio(planet_df[toi_col_dict["pp_key"]], planet_df[toi_col_dict["ms_key"]], planet_df[toi_col_dict["rs_key"]])
    planet_df = planet_df.reset_index(drop = True) # Reset the indices because some rows might've been removed

    # Estimate planet masses given radii
    planet_df[toi_col_dict["mp_key"]] = chen_kipping_louie_mass(planet_df[toi_col_dict["rp_key"]])

    # Estimate K amplitude of RV observation
    planet_df["K_amp"] = k_amp(planet_df[toi_col_dict["pp_key"]], planet_df[toi_col_dict["mp_key"]], planet_df[toi_col_dict["ms_key"]])

    # Cull the sample for systems that are observable by Keck in both dec and RV resolution (above -20 degrees dec, > 2 m/s k_amp)
    desirable_inds = np.logical_and(planet_df[toi_col_dict["dec_key"]] > -20, planet_df['K_amp'] > 2)
    planet_df = planet_df[desirable_inds]
    planet_df = planet_df.reset_index(drop = True)

    # Calculate TSM values
    planet_df["TSM"] = calculate_TSM(planet_df[toi_col_dict["rp_key"]],
                                     planet_df[toi_col_dict["rs_key"]],
                                     planet_df[toi_col_dict["Ts_key"]],
                                     planet_df[toi_col_dict["Jmag_key"]],
                                     planet_df[toi_col_dict["mp_key"]],
                                     planet_df[toi_col_dict["Fp_key"]])

    ##### Copied from Nicholas' notebook #####
    ##########################################
    rad_bins = 10**(np.linspace(0,1,6))
    fpl_bins = 10**(np.linspace(-1,4,6))
    tef_bins = np.array([2500,3900,5200,6500])
    all_bins = [rad_bins, fpl_bins, tef_bins]
    id_key   = "Full TOI ID"
    binned   = bin(toi_col_dict, planet_df, all_bins, id_key, "TSM")
    priority = 1
    my_tois  = binned[binned["priority"]==priority].reset_index(drop=True).sort_values(id_key)[id_key].values
    ##########################################
    ##########################################

    print("Priority {} targets: \n{}".format(priority, my_tois))
    np.savetxt(args.output_fname, my_tois, fmt='%.2f')
    print("Binning sucessful, output stored in {}".format(args.output_fname))

if __name__ == "__main__":
    main()
