###Atmospheric prioritization tools
#these tools are used to compare TOIs to each other,
#and existing planets, for their atmospheric followup value.
#There are a variety of functions in here, most of which are
#useful for this particular purpose; some are more generally
#useful

import numpy as np
import matplotlib
matplotlib.use('TkAgg') # To fix annoying "python is not installed as a framework" error
import matplotlib.pyplot as plt
import pandas as pd
import astropy as ap
from scipy.stats import binned_statistic_dd
from scipy.stats import linregress
from scipy.optimize import brentq

def basic_mr(r):
    '''
    This function applies Chen & Kipping 2017's M-R relation
    to planets with an unknown mass but known radius (aka
    basically all TOIs, KOIs, and K2OIs.
    '''
    m = np.zeros(len(r))
    for i in np.arange(len(r)):
        m[i] = (0.9718 * ((r[i])**3.58)) if r[i] < 1.23 \
        else (1.436 * ((r[i])**1.7)) if r[i] < 14.26 and r[i] > 1.23\
        else 317.8
    return m

def tess_colnames():
    '''
    this function returns, in order, the keys for the following cols:
    radius, star mass, period, stellar Teff, ars (orbital distance
    over stellar radius), Jmag, Vmag, stellar radius, declination,
    insolation flux relative to earth, planet mass, planet mass
    units (always earth for TESS data), id_key
    '''
    return ('Planet Radius Value', 'Stellar Mass', 'Orbital Period Value',
            'Effective Temperature Value', 'Ars', 'J mag', 'V mag',
            'Star Radius Value', 'TIC Declination', 'Effective Stellar Flux Value',
            'pl_masses', 'Earth', 'Full TOI ID')

#creating the colnames inside the .py file
rp_key,ms_key,pp_key,Ts_key,ars_key,Jmag_key,\
    Vmag_key,rs_key,dec_key,Fp_key,mp_key,\
    mp_units,id_key = tess_colnames()

def ars_from_t(period,star_mass,star_radius):
    '''
    This function finds the ars (ratio of orbital distance
    to stellar radius; needed for Kempton's TSM calculation).
    Period should be given in days; star mass and radius in
    terms of their value in solar units. These units seem
    to be pretty much standard across exoplanet databases
    so I have not implemented any sort of unit options
    in this function.
    '''
    G = 6.6726e-11
    Msun = 1.989e30
    Mstar = star_mass * Msun
    period_s = period * 86400
    a_m = (G*Mstar*(period_s**2) / (4 * (np.pi**2))) ** (1/3)
    star_radius_m = star_radius * 6.9551e8 #sun radius in m
    ars = a_m / star_radius_m
    #this value is correct for earth
    return ars

def get_TSM(planet_radius,star_radius,star_teff,Jmag,planet_mass,ars):
    '''
    calculates Kempton's Transmission Spectroscopy Metric for planets in the TESS
    dataset. KOI, K2OI, and known planets should already have it calculated.
    '''
    scale_factors = np.zeros(len(planet_radius))
    i = 0
    while i < len(planet_radius):
        #from Table 1 in Kempton et. al.
        scale_factors[i] = 0.19 if planet_radius[i]<1.5\
        else 1.26 if np.logical_and(planet_radius[i]>1.5,planet_radius[i]<2.75)\
        else 1.28 if np.logical_and(planet_radius[i]>2.75, planet_radius[i]<4)\
        else 1.15
        i += 1

    Teq = star_teff * (np.sqrt(1/ars)*(0.25**0.25)) #eqn 3 in Kempton et. al 2018

    #TSM is transmission spectroscopy metric
    TSM = scale_factors * (planet_radius**3) * Teq * (10**(-Jmag/5))\
            / (planet_mass * (star_radius**2))
    return TSM

def k_amp_finder(star_mass,star_radius,planet_mass,ars,mp_units='Earth'):
    '''
    finds the amplitude of stellar oscillations due to the planet's orbit
    star_mass: units of m_sun
    star_radius: units of r_sun
    planet_mass: units of m_earth or m_jup (should be able to set "mp_units = mp_units" as that
                                            variable is named higher up in the chain. Note "Earth"
                                            is capitalized, if you set this manually.
    ars: ratio of semimajor axis and stellar radius. This is used as an input instead of a alone
                        because a is often not directly available from these datasets.
    '''

    G = 6.67e-11 #SI units
    Msun = 1.98e30 #kg
    Rsun = 6.9551e8 #meters
    v_star = np.sqrt(G*star_mass*Msun/(ars*star_radius*Rsun))
    mp_factor = 5.972e24 if mp_units == 'Earth' else 1.898e27
    v_pl = v_star * ((planet_mass*mp_factor)/(star_mass*Msun))
    return v_pl

def clean_tess_data(toi_plus_list, tic_star_info, dec_cut=-20,
        k_amp_cut = 2, include_qlp=False):
    '''
    Performs cleaning on the toi+ list (available at tev.mit.edu) combined with
    the TIC star info, available at https://exofop.ipac.caltech.edu/tess/search.php.
    Recommended method is to save the TIC IDs imported from the TOI+ list as a CSV,
    then break them into two lists with <1000 entries (exofop appears to crash
    given inputs with >=1000 rows). If you downloaded this file from Github,
    the data/exofop directory should contain these data.
    '''

    #give the toi plus list a shorter name that's easier to type (c is for catalog)
    c = toi_plus_list.copy()

    #read in the TIC info, which has magnitude data that's missing from the TOI+ list
    star_info = tic_star_info

    #generate star mass from logg because that's easier to look at
    c['Stellar Mass'] = ((10**c['Surface Gravity Value']) * ((c['Star Radius Value']\
            *6.9551e10)**2) /  6.67e-8 ) / 1.99e33

    #generate mass, ars, and K_amp values; merge w/ TIC
    c[mp_key] = basic_mr(c[rp_key])
    c[ars_key] = ars_from_t(c[pp_key], c[ms_key], c[rs_key])
    c['K_amp'] = k_amp_finder(c[ms_key],c[rs_key],c[mp_key],c[ars_key])
    c['mass_flag'] = 0. # Mass flag is 1 if targets have a known mass, 0 if calculated from M-R relationship.
                        # A few TOIs are known planets *with* masses, though, so this is slightly incorrect.
                        # To fix, need to get a list of TOIs that are KPs with masses.
    catalog_2 = pd.merge(c,star_info, left_on = 'TIC', right_on = 'Target')

    #get rid of junk columns
    c2 = catalog_2.drop(columns=['TFOP SG1a','TFOP SG1b','TFOP SG2',\
                                'TFOP SG3','TFOP SG4','TFOP SG5','TFOP Master',\
                                'TOI Disposition'])
    c = c2
    c['TSM'] = get_TSM(c[rp_key],c[rs_key],c[Ts_key],c[Jmag_key],c[mp_key],c[ars_key])

    #get rid of anything with unknown radius or period values,
    #and optionally throw QLP planets back in
    if include_qlp == False:
        c = c[np.logical_and.reduce((c[rp_key]>0, c[pp_key]>0,
            c['Source Pipeline']=='spoc'))]
    elif include_qlp == True:
        c = c[np.logical_and(c[rp_key]>0, c[pp_key]>0)]
    c = c.drop_duplicates(subset=id_key).reset_index(drop=True)

    #get observational desirables. Note that there are no mag cuts made
    #I exempt a few planets from the Kamp cut because they're bright
    #enough that we've been observing them anyway!
    desirables = np.logical_and(
            c[dec_key]>dec_cut,
            np.logical_or.reduce((
                c['K_amp']>k_amp_cut, c[id_key]==554.01,
                c[id_key]==431.02)
            ))
    catalog_cleaned = c[desirables]#.drop_duplicates(subset='TOI')
    cc = catalog_cleaned.reset_index(drop=True)

    #remove known FPs or repeats. excluded_tois.txt explains
    #why these tois have been dropped
    cc = cc[np.logical_and.reduce((
        cc[id_key] != 468.01,
        cc[id_key] != 634.01,
        cc[id_key] != 635.01,
        cc[id_key] != 656.01,
        cc[id_key] != 1144.01,
        cc[id_key] != 1419.01
        ))]

    return cc

def binning_function(dataset,bins,id_key='Full TOI ID', sort_val='TSM'):
    '''
    This function bns all the TESS data by radius, Fp, and Teff.
    It's a little messy but does the job. Line by line documentation
    is below. Inputs are:
    -the data (should be combined known planet/KOI data and TESS data
    -The bins (should be 5 in radius, 5 in insol, and 3 in Teff -
        currently no customization options exist but if you want them,
        you can probably just change the "label" options below to
        add the number you want. No promises there though
    -id_key (a string that indexes where in the dataset the ID is)
    -sort_val: what you want to sort on (in our case, 'TSM').
    '''
    rad_bins = bins[0]
    fpl_bins = bins[1]
    tef_bins = bins[2]

    pre_bin = dataset.assign(
        radius_bin = pd.cut(dataset[rp_key],bins=rad_bins,labels = [1,2,3,4,5]),
        insol_bin = pd.cut(dataset[Fp_key],bins=fpl_bins,labels = [1,2,3,4,5]),
        st_Teff_bin = pd.cut(dataset[Ts_key],bins=tef_bins,labels = [1,2,3])
    )
        #pd.cut returns the bin number (or label - ints chosen here for ease)
        #of each row based on its place within a specified column.

    binned = pre_bin.dropna(subset=['radius_bin','insol_bin','st_Teff_bin']).\
            groupby(['radius_bin','insol_bin','st_Teff_bin']).apply(lambda _pre_bin:\
            _pre_bin.sort_values(by=[sort_val],ascending=False))\
            .reset_index(level = 3,drop=True)
             #this multi-line call:
                #1) drops values which are not in any of the desired bins
                #2) groups within those bins
                #3) sorts by TSM (the lambda thing is necessary because
                     #"groupby" produces a "groupby object" which can't be
                     #operated on normally)
                #4) drops all indexes which are not the bin numbers, which were
                    #just 1 to N anyway and therefore
                    #were worthless

    all_idx = binned.index.tolist()
    unique_idx = []
    for element in all_idx:
        if element not in unique_idx:
            unique_idx.append(element)

    binned['priority'] = np.zeros(len(binned))
    for idx in unique_idx:

        bin_items = len(binned.loc[idx].sort_values(sort_val,ascending=False)\
                .iloc[0:3].sort_values(Vmag_key)[id_key])
            #the number of objects in each bin

        if bin_items >= 3:
            binned.loc[binned[id_key] == binned.loc[idx].sort_values\
                    (sort_val,ascending=False).iloc[0:3]\
                    .sort_values(Vmag_key)[id_key].iloc[0],'priority'] = 1
            binned.loc[binned[id_key] == binned.loc[idx].sort_values\
                    (sort_val,ascending=False).iloc[0:3]\
                    .sort_values(Vmag_key)[id_key].iloc[1],'priority'] = 2
            binned.loc[binned[id_key] == binned.loc[idx].sort_values\
                    (sort_val,ascending=False).iloc[0:3]\
                    .sort_values(Vmag_key)[id_key].iloc[2],'priority'] = 3
            continue

        elif bin_items == 2:
            binned.loc[binned[id_key] == binned.loc[idx].sort_values\
                    (sort_val,ascending=False).iloc[0:3]\
                    .sort_values(Vmag_key)[id_key].iloc[0],'priority'] = 1
            binned.loc[binned[id_key] == binned.loc[idx].sort_values\
                    (sort_val,ascending=False).iloc[0:3]\
                    .sort_values(Vmag_key)[id_key].iloc[1],'priority'] = 2
            continue

        elif bin_items == 1:
            binned.loc[binned[id_key] == binned.loc[idx].sort_values\
                    (sort_val,ascending=False).iloc[0:3]\
                    .sort_values(Vmag_key)[id_key].iloc[0],'priority'] = 1

        #this is a HIDEOUS call but the idea is:
            #you are going into each bin sequentially (by index), sorting by TSM,
            #then sorting those top 3 by Vmag.
            #then, you are taking out the TOI value of the top entry there
            #(i.e., highest priority)
            #THEN, you are indexing that TOI in the list, .loc'ing to that row
            #and the priority column, and setting
            #THAT entry to 1. Then repeating this for the other priority values

            #all these if statements are a lot but unless I want to predefine
            #how many are in each bin (?) I think this is the fastest way to go, and as long as
            #TESS keeps its number of targets < 10^5 or something this shouldn't be unacceptably
            #long in terms of its run time

    return binned

def binning_function_X(dataset, bins, id_key='Full TOI ID', sort_val='X', num_to_rank=3):
    '''
    This function is a copy of the one above, though it doesn't include sorting
    by Vmag because, by default, it sorts by the "X" column, which is the ratio
    of the TSM and the expected total exposure time needed on HIRES to get a 5-sigma
    mass measurement.

    Arguments and return object are the same as above, save for the different
    sorting method.

    num_to_rank (optional, int): The number of targets to rank per bin.

    - Joey, 03/22/20
    '''
    rad_bins = bins[0]
    fpl_bins = bins[1]
    tef_bins = bins[2]

    pre_bin = dataset.assign(
        radius_bin = pd.cut(dataset[rp_key],bins=rad_bins,labels = [1,2,3,4,5]),
        insol_bin = pd.cut(dataset[Fp_key],bins=fpl_bins,labels = [1,2,3,4,5]),
        st_Teff_bin = pd.cut(dataset[Ts_key],bins=tef_bins,labels = [1,2,3])
    )
        #pd.cut returns the bin number (or label - ints chosen here for ease)
        #of each row based on its place within a specified column.

    binned = pre_bin.dropna(subset=['radius_bin','insol_bin','st_Teff_bin']).\
            groupby(['radius_bin','insol_bin','st_Teff_bin']).apply(lambda _pre_bin:\
            _pre_bin.sort_values(by=[sort_val],ascending=False))\
            .reset_index(level = 3,drop=True)
             #this multi-line call:
                #1) drops values which are not in any of the desired bins
                #2) groups within those bins
                #3) sorts by TSM (the lambda thing is necessary because
                     #"groupby" produces a "groupby object" which can't be
                     #operated on normally)
                #4) drops all indexes which are not the bin numbers, which were
                    #just 1 to N anyway and therefore
                    #were worthless

    all_idx = binned.index.tolist()
    unique_idx = []
    for element in all_idx:
        if element not in unique_idx:
            unique_idx.append(element)

    binned['priority'] = np.zeros(len(binned))

    for idx in unique_idx:

        bin_items = len(binned.loc[idx].sort_values(sort_val,ascending=False).iloc[0:num_to_rank][id_key])
        #the number of objects in each bin

        for i in range(1, num_to_rank+1):

            if bin_items == i and bin_items <= num_to_rank:
                for j in range(i):
                    binned.loc[binned[id_key] == binned.loc[idx].sort_values\
                            (sort_val,ascending=False).iloc[0:num_to_rank][id_key].iloc[j],'priority'] = j + 1

            elif bin_items > num_to_rank:
                for j in range(num_to_rank):
                    binned.loc[binned[id_key] == binned.loc[idx].sort_values\
                            (sort_val,ascending=False).iloc[0:num_to_rank][id_key].iloc[j],'priority'] = j + 1

        # Note from Nicholas, leftover from function above (but with part about
        # sorting by Vmag deleted):

        #this is a HIDEOUS call but the idea is:
            #you are going into each bin sequentially (by index), sorting by TSM/t_HIRES
            #then, you are taking out the TOI value of the top entry there
            #(i.e., highest priority)
            #THEN, you are indexing that TOI in the list, .loc'ing to that row
            #and the priority column, and setting
            #THAT entry to 1. Then repeating this for the other priority values

            #all these if statements are a lot but unless I want to predefine
            #how many are in each bin (?) I think this is the fastest way to go, and as long as
            #TESS keeps its number of targets < 10^5 or something this shouldn't be unacceptably
            #long in terms of its run time

    return binned

################################################################################
############# These two functions taken from the CPS utils Github ##############
################################################################################
def exposure_time(vmag, counts, iod=False, time1=110.0, vmag1=8.0, exp1=250.0):
    """Exposure Time

    Estimate exposure time based on scaling. Cannonical exposure time
    is 110s to get to 250k on 8th mag star with iodine cell in.

    Args:
        vmag (float): V-band magnitude
            250 = 250k, 10 = 10k (CKS) i.e. SNR = 45 per pixel.
        counts (float): desired number of counts.
            250 = 250k, 10 = 10k (CKS) i.e. SNR = 45 per pixel.
        iod (bool): is iodine cell in or out? If out, throughput is higher
            by 30%

    Returns:
        float: exposure time (seconds)

    """
    iodfactor = 0.7

    # flux star / flux 8th mag star
    fluxfactor = 10.0**(-0.4*(vmag-vmag1))
    time = time1 / fluxfactor
    time *= counts / exp1
    if iod==False:
        time *= iodfactor
    return time

def exposure_counts(vmag, time, **kwargs):
    """Exposure counts

    Inverse of `exposure_time.` Given a magnitude and an exposure
    time, how many counts will be collected?

    Args:
        vmag (float) : vband magnitude
        time (float) : exposure time (seconds)
        **kwargs : keyword arguments passed to exposure_time

    Returns:
        float: expected number of counts

    """
    f = lambda counts : exposure_time(vmag, counts, **kwargs) - time
    _counts = brentq(f,0,2000,)
    return _counts
################################################################################
############# Two functions above taken from the CPS utils Github ##############
################################################################################

def counts_to_sigma(counts):
    '''
    Convert exposure meter counts (in thousands) to a velocity precision (in m/s).
    This is taken approximately from https://caltech-ipac.github.io/hiresprv/performance.html,
    and the sigmas here are likely an underestimate. Once a target has RVs, then we
    should take the sigma straight from the data. These are fine for now I guess?
    - Joey, 03/21/20

    Note, right now this function only works for a standard number of exposure meter
    counts across all targets (unless this will be called in a loop in t_HIRES in some
    future version). Might be desirable to be able to set different exposure meter
    count numbers for different vmag?
    - Joey, 03/22/20

    Args:
        counts (int): An int, which represents the desired exposure meter counts (in thousands)

    Returns:
        Radial velocity measurement precision, in m/s.
    '''
    counts_to_sigma_dict = {
        30:2.5,
        60:2.,
        120:1.4,
        125:1.5,
        250:1.
    }

    # # Make it iterable, if you have just one exposure meter count for every target.
    # if type(counts) is int:
    #         counts = [counts]
    #assert all([count in counts_to_sigma_dict.keys() for count in counts]), 'For now, only use counts of 60k, 125k, and 250k.'

    assert counts in counts_to_sigma_dict.keys(), 'For now, only use counts of 30k, 60k, 120k, 125k, and 250k.'
    return counts_to_sigma_dict[counts]

def t_HIRES_plavchan(vmag, n_counts, k, SNR=5.):
    '''
    Calculate the total HIRES time (in seconds) needed to get a SNR-sigma detection of a planet.

    Some caveats:
        - This is really a SNR-sigma detection of the planet in velocity space, not mass. If there are significant
            uncertainties going from a K-space detection to a mass measurement, then I'll need to include that later.
        - This uses a *rough* estimate of the velocity precision via counts (hardcoded values are
            estimated by eye from https://caltech-ipac.github.io/hiresprv/performance.html). Once a target actually
            is observed and has RVs, we'll use the sample mean of the uncertainty on each RV.
        - Does not include time per observation for overhead. Should check if Ashley's code already includes this,
            because don't want to double count overhead time and short-change ourselves by estimating higher exposure
            times than actually necessary.
    - Joey, 03/21/20

    Let's just move ahead with this, knowing that the expected exposure time is a
        vast underestimate. It should work fine enough for planning purposes since
        the targets are all being ranked relative to eachother on the same metric.
        - Joey, 03/24/20

    Args:
        vmag (ndarray(float)): Target vmag
        n_counts (ndarray(int) or int): Desired exposure meter counts
        k (ndarray(float)): Expected K amplitude, in m/s
        SNR (optional, float): Default = 5.0. Desired detection SNR.

    Returns:
        ndarray(float): Estimated total exposure time needed on HIRES for a SNR-sigma
            mass measurement. Be wary of caveats noted above until they are addressed.
        ndarray(float): Estimated number of observations needed to get a 5-sigma mass.
    '''
    t_ob = exposure_time(vmag, n_counts) # Time in seconds for a single exposure
    sigma = counts_to_sigma(n_counts)   # Velocity precision
    N_obs = (SNR * sigma / k)**2        # Num observations needed for a 5-sigma mass

    return t_ob * N_obs

def t_HIRES_howard(vmag, n_counts, k, p_orb, SNR=6., t_span=90):
    '''
    Same as t_HIRES_plavchan but uses equation 9 from Howard & Fulton 2016
    to calculate N_obs. Notice the different default value for the SNR, which they
    suggest. Also, need to use the orbital information.

    Comparing the rankings using the two t_HIRES functions, they don't change. Let's
    just move ahead using the Plavchan one for now since it's slightly simpler.
    - Joey, 03/24/20
    '''
    t_ob = exposure_time(vmag, n_counts)
    sigma = counts_to_sigma(n_counts)
    tau = p_orb / t_span # Uses fiducial value of 90 days as the default time span
    N_obs = (SNR * sigma / k)**2 * (1 + (10**(tau-1.5))**2)

    return t_ob * N_obs

def return_known_spectra():
    '''
    returns planets under 11.2Re with known transmission
    spectra. the index of the planet in kp_w_spectra
    corresponds to the index in has_features, which
    identifies whether any absorption features have
    been positively identified. For some cases, especially
    TESS planets like pi Men c, lack of any known features
    does NOT indicate that the planet is cloudy.
    '''
    kp_w_spectra = ['WASP-107 b', 'GJ 1214 b', 'WASP-80 b',
               'GJ 3470 b',   'HAT-P-11 b','HAT-P-12 b',
               '55 Cnc e',    'HAT-P-18 b','WASP-166 b',
               'HAT-P-26 b',  'pi Men c',  'WASP-29 b',
               'HD 149026 b', 'K2-25 b',   'HD 97658 b',
               'HD 3167 b',   'GJ 436 b',  'K2-18 b',
               'HD 106315 c']

    has_features = [1,             0,           0,
                   1,              1,           0,
                   1,              0,           0,
                   1,              0,           0,
                   0,              0,           0,
                   0,              0,           1,
                   0]

    return (kp_w_spectra, has_features)

def has_obs(data, kpwks):
    '''
    using the list of known spectra (above), adds a
    dataframe column (1 or 0) describing whether a
    planet has spectra observations
    '''
    kp_w_spectra,_ = return_known_spectra()
    ho = np.zeros(len(data[id_key]))
    names = np.array(data[id_key])
    for i in np.arange(len(ho)):
        if names[i] in kp_w_spectra:
            ho[i] = 1
    return ho

def bin_plotter(binned_data, bins, rbin, use_alpha=False):
    '''
    Function for visualization of the binned data.
    Inputs are the dataframe itself and the radius bin
    (1, 2, 3, 4, or 5) of interest.
    '''
    #do a little bit of data cleaning; NOTE should
    #ultimately move this outside of the function
    kpwks, hf = return_known_spectra()
    data_copy = binned_data.copy()
    data_copy['has_spectrum'] = has_obs(data_copy, kpwks)

    #we only want data products which have high atmospheric
    #priority or already have observations
    aois = data_copy[np.logical_or(
        data_copy['priority']!=0, data_copy['has_spectrum']==1
        )]
    rbin1 = aois.loc[rbin,:,:]

    #get the bin edges
    fpl_bins = bins[1]
    tef_bins = bins[2]

    F = np.array(rbin1[Fp_key])
    Ts = np.array(rbin1[Ts_key])
    rp = np.array(rbin1[rp_key])
    P = np.array(rbin1['priority'])
    N = np.array(rbin1[id_key])
    #okay, so I think one useful method would be to try to print these and use the
    #annotate function
    #(https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-
    #with-different-text-at-each-data-point) and see how that works
    #(i.e. whether it's legible)

    def colorfinder(name, priority):

        mass_flag = data_copy[data_copy[id_key] == name]['mass_flag'].values[0]

        if name in kpwks:
            idx = kpwks.index(name)
            if hf[idx]:
                return 'red'
            elif not hf[idx]:
                return 'dimgrey'

        # If this is a planet with a known mass
        elif mass_flag == 1.:
            return 'magenta'

        # elif name in ['K2-182 b', 'K2-199 b', 'K2-199 c']:
        #     return 'lime'


        else:
            if use_alpha:
                return 'green'
            else:
                if priority == 1:
                    return 'green'
                if priority == 2:
                    return 'yellow'
                if priority == 3:
                    return 'orange'
                if priority == 4:
                    return 'pink'
                if priority == 5:
                    return 'cyan'
                else:
                    return 'black'

    def alphafinder(name, priority):
        mass_flag = data_copy[data_copy[id_key] == name]['mass_flag'].values[0]

        if name in kpwks:
            idx = kpwks.index(name)
            if hf[idx]:
                return 1.
            elif not hf[idx]:
                return 1.
        elif mass_flag == 1.:
            return 1.
        else:
            if priority in [1,2,3,4,5]:
                return 1./priority
            else:
                return 0.15

    def textcolorfinder(name):
        mass_flag = data_copy[data_copy[id_key] == name]['mass_flag'].values[0]

        if name in kpwks:
            idx = kpwks.index(name)
            if hf[idx]:
                return 'red'
            elif not hf[idx]:
                return 'dimgrey'
        elif mass_flag == 1.:
            return 'magenta'
        else:
            return 'black'

    fig, ax = plt.subplots(figsize=(14,9))
    figsize=[10,7]
    ax.grid()
    txt = np.array(rbin1[id_key])

    #doing the title stuff
    bin_edges = np.round(bins[0],1)
    title_txt = r'Planets & Planet Candidates With Radius Between ' + str(bin_edges[rbin-1]) + r' and ' + \
        str(bin_edges[rbin]) + r'$R_\oplus$'


    for i in np.arange(len(rbin1)):
        if use_alpha:
            ax.semilogx(F[i], Ts[i], '.',ms=rp[i]*5, color=colorfinder(N[i],P[i]), alpha=alphafinder(N[i], P[i]))
        else:
            ax.semilogx(F[i], Ts[i], '.',ms=rp[i]*5, color=colorfinder(N[i],P[i]))
        ax.annotate(txt[i], (F[i], Ts[i]+rp[i]*9),color=textcolorfinder(N[i]), alpha=0.7)

    ##added for TKS in person
    for f in np.arange(1,6,1):
        for t in np.arange(1,4,1):
            try:
                size = len(binned_data.loc[rbin,f,t])
                ax.annotate('   ' + str(size),(fpl_bins[f-1],tef_bins[t-1]+1100))
            except KeyError:
                continue

    ax.set_yticks(tef_bins)
    ax.set_xticks(fpl_bins)
    ax.set_ylabel('Stellar Effective Temperature (K)')
    ax.set_xlabel(r'Insolation Flux')
    ax.set_title(title_txt)

    plt.show()

    return fig, ax
