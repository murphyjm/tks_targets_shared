'''
This file is under construction!

- Joey, 2020-07-31
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

id_key = 'Full TOI ID'

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

def bin_plotter_new(binned_data, bins, rbin, use_alpha=False, show_fig=True, save_fig=False, save_fname=''):
    '''
    TODO
    This function needs updating


    Function for visualization of the binned data.
    Inputs are the dataframe itself and the radius bin
    (1, 2, 3, 4, or 5) of interest.
    '''
    #do a little bit of data cleaning; NOTE should
    #ultimately move this outside of the function
    kpwks, hf = return_known_spectra()
    data_copy = binned_data.copy()
    data_copy['has_spectrum'] = has_obs(data_copy, kpwks)

    #we I have a try...except block in my code and When an exception is throw. I really just want to continue with the code because in that case, everything is still able to run just fine. The problem is if you leave the except: block empty or with a #do nothing, it gives you a syntax error. I can't use continue because its not in a loop. Is there a keyword i can use that tells the code to just keep going?I have a try...except block in my code and When an exception is throw. I really just want to continue with the code because in that case, everything is still able to run just fine. The problem is if you leave the except: block empty or with a #do nothing, it gives you a syntax error. I can't use continue because its not in a loop. Is there a keyword i can use that tells the code to just keep going?I have a try...except block in my code and When an exception is throw. I really just want to continue with the code because in that case, everything is still able to run just fine. The problem is if you leave the except: block empty or with a #do nothing, it gives you a syntax error. I can't use continue because its not in a loop. Is there a keyword i can use that tells the code to just keep going?only want data products which have high atmospheric
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

    if save_fig:
        fig.savefig(f'{save_fname}')

    if show_fig:
        plt.show()

    return fig, ax
