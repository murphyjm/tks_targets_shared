# File i/o
import json

# Analysis
import numpy as np
import pandas as pd
from astropy import constants as const

def load_and_merge(toi_fname, exo_fname):
    """
    Load and merge the toi and exofop data.
    """
    toi_df = pd.read_csv(toi_fname, comment='#')
    exo_df = pd.read_csv(exo_fname)
    
    # Merge the two, if there's no exofop data we have to get rid of the row
    return toi_df.merge(exo_df, left_on='TIC', right_on='Target').reset_index(drop=True)

def load_toi_col_names(toi_col_dict_fname):
    """
    Loads a dictionary with standard toi column names, since sometimes the column names in the TOI files
        change on the TESS website. If the column names change, just update the .json file. No need to
        change code in here.
    """
    toi_col_dict = {}
    with open(toi_col_dict_fname) as f:
        toi_col_dict = json.load(f)
    return toi_col_dict

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

def k_amp(P, m_planet, m_star):
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
    return 203 * (P ** (-1/3)) * m_planet * me_mj / ((m_star + (9.548e-4 * m_planet * me_mj)) ** (2/3))

def calculate_TSM(r_planet, r_star, teff_star, Jmag, m_planet, ar_ratio):
    """
    Calculate TSM, a S/N proxy for JWST atmospheric transmission spectra,
        from Kepmton et al. 2018 (https://arxiv.org/pdf/1805.03671.pdf).
    Args:
        r_planet (ndarray): Planet radii (units of Earth radius)
        r_star (ndarray): Stellar radii (units of solar radius)
        teff_star (ndarray): Stellar effective temperature (Kelvin)
        Jmag (ndarray): J-band stellar magnitude (mag)
        m_planet (ndarray): Planet mass (units of Earth mass)
        ar_ratio (ndarray): Orbital semi-major axis and stellar radius ratio.
    Returns:
        ndarray: TSM value for each entry (see Kempton et al. 2018 equation 1)
    """
    # Table 1 in Kempton et al.
    scale_factors = np.array([0.19 if r < 1.5 \
                             else 1.26 if np.logical_and(r >= 1.5, r < 2.75) \
                             else 1.28 if np.logical_and(r >= 2.75, r < 4) \
                             else 1.15 \
                             for r in r_planet])

    T_eq = teff_star * (np.sqrt(1 / ar_ratio) * (0.25**0.25)) # Equation 3 in Kempton et al. 2018

    # Equation 1 in Kempton et al. 2018
    return scale_factors * (r_planet**3) * T_eq * (10**(-Jmag/5)) / (m_planet * (r_star**2))

def calculate_TSM_natalie(r_planet, r_star, teff_star, Jmag, m_planet, insol_flux):
    """
    Calculate TSM, a S/N proxy for JWST atmospheric transmission spectra,
        from Kepmton et al. 2018 (https://arxiv.org/pdf/1805.03671.pdf).
    Args:
        r_planet (ndarray): Planet radii (units of Earth radius)
        r_star (ndarray): Stellar radii (units of solar radius)
        teff_star (ndarray): Stellar effective temperature (Kelvin)
        Jmag (ndarray): J-band stellar magnitude (mag)
        m_planet (ndarray): Planet mass (units of Earth mass)
        insol_flux (ndarray): Insolation flux (units of Earth insolation flux --> NOTE: Actually not sure the unit here)
    Returns:
        ndarray: TSM value for each entry (see Kempton et al. 2018 equation 1)
    """
    # Table 1 in Kempton et al.
    scale_factors = np.array([0.19 if r < 1.5 \
                             else 1.26 if np.logical_and(r >= 1.5, r < 2.75) \
                             else 1.28 if np.logical_and(r >= 2.75, r < 4) \
                             else 1.15 \
                             for r in r_planet])
    T_SUN = 5777 # Kelvin
    # This is Natalie's way of calculating planet equilibrium temperature based on insolation flux
    T_eq = (((insol_flux)*(1/4))**0.25)*np.sqrt(const.R_sun / const.au) * T_SUN # Need source for explaining this line

    # Equation 1 in Kempton et al. 2018
    return scale_factors * (r_planet**3) * T_eq * (10**(-Jmag/5)) / (m_planet * (r_star**2))
