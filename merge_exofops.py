import os
import pandas as pd
import argparse

"""
Example usage:
    python merge_exofops.py exofop_search_2019-11-14_1of2.csv exofop_search_2019-11-14_2of2.csv
"""
data_folder = "data/exofop"

parser = argparse.ArgumentParser(description = "Merge two lists of TIC IDs.")
parser.add_argument("fname_1", type=str, help="Filename 1")
parser.add_argument("fname_2", type=str, help="Filename 2")
args = parser.parse_args()

exofop_1_fname = os.path.join(data_folder, args.fname_1)
exofop_2_fname = os.path.join(data_folder, args.fname_2)

exo_df1 = pd.read_csv(exofop_1_fname, comment='#')
exo_df2 = pd.read_csv(exofop_2_fname, comment='#')

exo_df  = pd.concat((exo_df1, exo_df2))

output_fname = os.path.join(data_folder, args.fname_1[:-8] + 'combined.csv')
exo_df.to_csv(output_fname)
print("Output stored in {}".format(output_fname))
