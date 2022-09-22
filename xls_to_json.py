"""
author:     cjsisco
date:       2022-09-16
objective: 
    read props from .xlsx file and output to .json file
inputs:
    .xls file containing thermo properties in P-T tables at various GORs.
        - each GOR is in a different sheet
        - last sheet contains raw deviation data.
return:
    .json file (thermo props)
"""

# %% import pkgs
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import json
from pathlib import Path
from config import ROOT_DIR, DATA_DIR

# %% get file input/output path and set dataframe

# specify well and string (must match lookup table .xlsx file name)
#well = "sample"
#string = "2"
#fluid = f"{well}{string}"
fluid = "S14-SAFT-MILA2020"

# construct input/output file names
file_In = f"lookup_{fluid}.xlsx"
file_Out = f"{fluid}.json"

# root directory
#CWD = Path( __file__ ).parent.absolute()
#CWD = Path.cwd()
data_In = DATA_DIR / "xls"
data_Out = DATA_DIR / "json"

# create excel object and create dataFrame
file_Path = data_In / file_In
xls = pd.ExcelFile(file_Path)
df_master = pd.read_excel(xls, sheet_name="master")
df_lookup = pd.read_excel(xls, sheet_name=None, header=None)

# get columns headings from df_master
df_cols = df_master.columns.values.tolist()

#%% close excel file to allow saving
#xls.close()

#%% extract master properties
# write each col of df_master to a list
df_str = [df_master[col].tolist() for col in df_cols]

# extract CSV into lists
df_list = [col[0].split(", ") for col in df_str]

# convert coordinate vals from string to float
idx_coord = df_cols.index("coord_label")    #idx of coord_label
str_coord = df_list[idx_coord]
nCoord = len(str_coord)
nX = [None] * nCoord
for i,col in enumerate(str_coord):
    idx = df_cols.index(col)
    df_list[idx] = [float(item) for item in df_list[idx]]
    nX[i] = len(df_list[idx])
    
#%% define number of steps used to generate the lookup tables (make this dynamic!!)
# count num props
idx_prop = df_cols.index("prop_label")  #idx of prop_label
nProp = len(df_list[idx_prop])

# preallocate `prop_table`
# 4D block of data containing all properties
prop_shape = tuple([nProp] + nX)
prop_table = np.zeros(prop_shape, dtype=float)

#%% loop through sheets ('TPF' only) and generate slice corresponding to each sheet
import itertools
i = 0
for (sht_name, sht_df) in df_lookup.items():
    if "TPF" in sht_name:
        df = pd.read_excel(xls, sheet_name=sht_name, header=None)
        prop_TP = df.to_numpy()
        #prop_TP = sht_df.to_numpy()
        for iX1, iX2 in itertools.product(range(nX[1]), range(nX[2])):
            # split each string and convert to a list of floats
            prop_str = prop_TP[iX1][iX2]
            prop_val = prop_str.split(", ")
            prop_val = [float(item) for item in prop_val]
            
            # populate prop_table
            prop_table[:,i,iX1,iX2] = prop_val

        # next index     
        i += 1

# put prop_table in df_list
idx_prop = df_cols.index("prop_table")  #idx of prop_table
df_list[idx_prop] = prop_table.tolist()

# %% save information in a dictionary then to .json (for later use during data preparation)
prop_dict = {
        'fileName': file_In,
        'fluid': fluid,
        }
for i in range(len(df_cols)):
    prop_dict[df_cols[i]] = df_list[i]

# create json object from dict
prop_json = json.dumps(prop_dict)

# write data to json file
file_Path = data_Out / file_Out
with open(file_Path, "w") as f:
    f.write(prop_json)

# %% close xls
xls.close

# %%
