###Atmospheric prioritization tools
#these tools are used to compare TOIs to each other, 
#and existing planets, for their atmospheric followup value. 
#There are a variety of functions in here, most of which are
#useful for this particular purpose; some are more generally
#useful

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy as ap
from scipy.stats import binned_statistic_dd
from scipy.stats import linregress

def basic_mr(r):
    '''
    This function applies Chen & Kipping 2017's M-R relation
    to planets with an unknown mass but known radius (aka
    basically all TOIs, KOIs, and K2OIs. 
    '''
    m = np.zeros(len(r))
    for i in np.arange(len(r)):
        m[i] = (0.9718 * ((r[i])**3.58)) if r[i] < 1.23 \
        else (1.436 * ((r[i])**1.7)) if r[i] < 14.26 \
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

def clean_tess_data(toi_plus_list, tic_star_info,include_qlp=False):
    '''
    Performs cleaning on the toi+ list (available at tev.mit.edu) combined with 
    the TIC star info, available at https://exofop.ipac.caltech.edu/tess/search.php.
    Recommended method is to save the TIC IDs imported from the TOI+ list as a CSV, 
    then break them into two lists with <1000 entries (exofop appears to crash
    given inputs with >=1000 rows). If you downloaded this file from Github, 
    the data/exofop directory should contain these data.
    '''
    
    c = toi_plus_list
    
    #read in the TIC info, which has magnitude data that's missing from the TOI+ list
    star_info = pd.read_csv('data/exofop/TIC_star_info_jan_15_all.csv',delimiter=',',header=10)

    #generate star mass from logg because that's easier to look at
    c['Stellar Mass'] = ((10**c['Surface Gravity Value']) * ((c['Star Radius Value']\
            *6.9551e10)**2) /  6.67e-8 ) / 1.99e33 

    #generate mass, ars, and K_amp values; merge w/ TIC
    c[mp_key] = basic_mr(c[rp_key])
    c[ars_key] = ars_from_t(c[pp_key], c[ms_key], c[rs_key])
    c['K_amp'] = k_amp_finder(c[ms_key],c[rs_key],c[mp_key],c[ars_key])
    catalog_2 = pd.merge(c,star_info, left_on = 'TIC', right_on = 'Target')

    #get rid of junk columns
    c2 = catalog_2.drop(columns=['TFOP SG1a','TFOP SG1b','TFOP SG2',\
                                'TFOP SG3','TFOP SG4','TFOP SG5','TFOP Master',\
                                'TOI Disposition'])
    c = c2.sort_values('Full TOI ID')
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
    desirables = np.logical_and(c[dec_key]>-20, c['K_amp']>2)
    catalog_cleaned = c[desirables]#.drop_duplicates(subset='TOI')
    cc = catalog_cleaned.reset_index(drop=True)
    return cc

def binning_function(dataset,bins,id_key='Full TOI ID',sort_val='TSM'):
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

def bin_plotter(binned_data, bins, rbin):
    '''
    Function for visualization of the binned data. 
    Inputs are the dataframe itself and the radius bin
    (1, 2, 3, 4, or 5) of interest.
    '''

    aois = binned_data[binned_data['priority']!=0]
    rbin1 = aois.loc[rbin,:,:]

    #get the bin edges
    fpl_bins = bins[1]
    tef_bins = bins[2]

    F = np.array(rbin1[Fp_key])
    Ts = np.array(rbin1[Ts_key])
    rp = np.array(rbin1[rp_key])
    P = np.array(rbin1['priority'])
    #okay, so I think one useful method would be to try to print these and use the
    #annotate function
    #(https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-
    #with-different-text-at-each-data-point) and see how that works
    #(i.e. whether it's legible)
    
    def colorfinder(priority):
        if priority == 1:
            return 'green'
        if priority == 2:
            return 'yellow'
        if priority == 3:
            return 'orange'
    
    fig, ax = plt.subplots(figsize=(14,9))
    figsize=[10,7]
    ax.grid()
    txt = np.array(rbin1[id_key])
    
    #doing the title stuff
    bin_edges = np.round(10**(np.linspace(0,1,6)),1)
    title_txt = r'Planets & Planet Candidates With Radius Between ' + str(bin_edges[rbin-1]) + r' and ' + \
        str(bin_edges[rbin]) + r'$R_\oplus$'


    for i in np.arange(len(rbin1)):
        ax.semilogx(F[i], Ts[i], '.',ms=rp[i]*5,color=colorfinder(P[i]))
        ax.annotate(txt[i], (F[i], Ts[i]+rp[i]*9))

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
