import numpy as np
import pandas as pd

def bin(toi_col_dict, dataset, bins, id_key, sort_val):
    """
    Binning function copied from Nicholas' notebook.
    """
    rp_key = toi_col_dict["rp_key"]
    Fp_key = toi_col_dict["Fp_key"]
    Ts_key = toi_col_dict["Ts_key"]
    Vmag_key = toi_col_dict["Vmag_key"]

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

    binned = pre_bin.dropna(subset=['radius_bin','insol_bin','st_Teff_bin']).groupby(['radius_bin',\
                                    'insol_bin','st_Teff_bin']).apply(lambda _pre_bin:\
                                    _pre_bin.sort_values(by=[sort_val],ascending=False)).reset_index(level = 3,drop=True)
                    #this multi-line call:
                        #1) drops values which are not in any of the desired bins
                        #2) groups within those bins
                        #3) sorts by TSM (the lambda thing is necessary because "groupby" produces a "groupby object"
                                #which can't be operated on normally)
                        #4) drops all indexes which are not the bin numbers, which were just 1 to N anyway and therefore
                                #were worthless

    all_idx = binned.index.to_list()
    unique_idx = []
    for element in all_idx:
        if element not in unique_idx:
            unique_idx.append(element)

    binned['priority'] = np.zeros(len(binned))
    for idx in unique_idx:

        bin_items = len(binned.loc[idx].sort_values(sort_val,ascending=False).iloc[0:3].sort_values(Vmag_key)[id_key])
            #the number of objects in each bin

        if bin_items >= 3:
            binned.loc[binned[id_key] == binned.loc[idx].sort_values(sort_val,ascending=False).iloc[0:3]\
                       .sort_values(Vmag_key)[id_key].iloc[0],'priority'] = 1
            binned.loc[binned[id_key] == binned.loc[idx].sort_values(sort_val,ascending=False).iloc[0:3]\
                       .sort_values(Vmag_key)[id_key].iloc[1],'priority'] = 2
            binned.loc[binned[id_key] == binned.loc[idx].sort_values(sort_val,ascending=False).iloc[0:3]\
                       .sort_values(Vmag_key)[id_key].iloc[2],'priority'] = 3
            continue

        elif bin_items == 2:
            binned.loc[binned[id_key] == binned.loc[idx].sort_values(sort_val,ascending=False).iloc[0:3]\
                       .sort_values(Vmag_key)[id_key].iloc[0],'priority'] = 1
            binned.loc[binned[id_key] == binned.loc[idx].sort_values(sort_val,ascending=False).iloc[0:3]\
                       .sort_values(Vmag_key)[id_key].iloc[1],'priority'] = 2
            continue

        elif bin_items == 1:
            binned.loc[binned[id_key] == binned.loc[idx].sort_values(sort_val,ascending=False).iloc[0:3]\
                       .sort_values(Vmag_key)[id_key].iloc[0],'priority'] = 1

        #this is a HIDEOUS call but the idea is:
            #you are going into each bin sequentially (by index), sorting by TSM, then sorting those top 3 by Vmag.
            #then, you are taking out the TOI value of the top entry there (i.e., highest priority)
            #THEN, you are indexing that TOI in the list, .loc'ing to that row and the priority column, and setting
            #THAT entry to 1. Then repeating this for the other priority values

            #all these if statements are a lot but unless I want to predefine
            #how many are in each bin (?) I think this is the fastest way to go, and as long as
            #TESS keeps its number of targets < 10^5 or something this shouldn't be unacceptably
            #long in terms of its run time
    return binned
