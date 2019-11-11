# Command line input
import sys

# Files
import os
from astropy.io import ascii

# Analysis
import numpy as np
import pandas as pd
import astropy as ap
from astropy import constants as const
from scipy.stats import binned_statistic_dd
from scipy.stats import linregress
# from astroquery.mast import Catalogs

# Development
from pdb import set_trace

def load_and_merge(toi_fname, exo_fname):
    """
    Load and merge the toi and exofop data.
    """
    toi_df = pd.read_csv(toi_fname, comment='#')
    exo_df = pd.read_csv(exo_fname)

    # Merge the two, but don't drop rows in the toi_df if there's no exofop data
    return toi_df.merge(exo_df, left_on='TIC', right_on='Target', how='left').reset_index(drop=True)

def ar_ratio(P, m_star, r_star):
    """
    Calculate the ratio of the planet's semi-major orbital axis and its host star's radius
        using Kepler's 3rd Law: G M = Omega^2 a^3 (where Omega = 2 * pi / P).
    Args:
        P (ndarray): Planet orbital period (units of days)
        m_star (ndarray): Stellar mass (units of solar mass)
        r_star (ndarray): Stellar radius (units of solar radius)
    """
    P_s = P * 24 * 60 * 60 # Convert planet orbital period from days to seconds
    a = ((const.G * m_star * P_s**2) / (2 * np.pi)**2) ** (1/3)
    return a / r_star

def chen_kipping_louie_mass(radius):
    """
    Returns mass given radius based on mass-radius relation found by Chen & Kipping 2016 (https://arxiv.org/abs/1603.08614)
        and Louie et al. 2018 (https://iopscience.iop.org/article/10.1088/1538-3873/aaa87b).
    Args:
        radius (ndarray): Planet radii in units of Earth radii.
    Returns:
        ndarray of corresponding masses. Return value element is -1 if planet radius is too large.
    """
    return np.array([0.9718 * r**3.58 if r < 1.23 else 1.436 * r**1.7 if r < 14.26 else -1. for r in radius])

def k_amp(P, m_p, m_star):
    '''
    RV signal semi-amplitude (i.e. K amplitude) calculation.
        (from https://exoplanetarchive.ipac.caltech.edu/docs/poet_calculations.html)
    Assumes circular orbit (i.e. e = 0) and inclination of 90 degrees.

    Args:
        P (ndarray): Planet orbital period (units of days)
        m_p (ndarray): Planet mass (units of Earth mass)
        m_star (ndarray): Host star mass (units of Solar mass)
    Returns:
        ndarray of estimated K amplitude in RV signal.
    '''
    me_mj = const.M_earth / const.M_jup
    return 203 * (p ** (-1/3)) * m_p * me_mj / ((m_star + (9.548e-4 * planet_mass * me_mj)) ** (2/3))

def main():
    """
    TKS prioritization code.

    Example command line call:
        python prioritize.py toi+-2019-10-29.csv exofop_search2019-10-29_combined.csv
    """
    toi_fname = os.path.join("data/toi", sys.argv[1])
    exo_fname = os.path.join("data/exofop", sys.argv[2])
    planet_df = load_and_merge(toi_fname, exo_fname)

    # Clean up the resulting df a little bit
    planet_df = planet_df.drop_duplicates(subset="Full TOI ID").sort_values("Full TOI ID")
    # Add a column for stellar mass calculated from surface gravity
    # N.B. the Exofop data contains stellar mass, but for rows that are not matched
    # with Exofop data, give them a stellar mass.
    planet_df["Stellar Mass (from logg)"] = (10**planet_df["Surface Gravity Value"] * (planet_df["Star Radius Value"] * const.R_sun) / const.G) / const.M_sun

    # Remove rows from the df that don't meet these three criteria
    planet_df = planet_df[np.logical_and.reduce((planet_df["Planet Radius Value"] > 0,
                                                 planet_df["Orbital Period Value"] > 0,
                                                 planet_df["Source Pipeline"] == "spoc"))]

    # Add a column for the ratio of the planet's semi-major orbital axis and the radius of the host star
    planet_df["ar_ratio"] = ar_ratio(planet_df["Orbital Period Value"], planet_df["Stellar Mass (from logg)"], planet_df["Star Radius Value"])
    planet_df = planet_df.reset_index(drop=True) # Reset the indices because some rows might've been removed

    

if __name__ == "__main__":
    main()
