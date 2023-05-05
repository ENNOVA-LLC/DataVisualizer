import numpy as np
import pandas as pd
import json
import itertools
from pathlib import Path
from config import ROOT_DIR, DATA_DIR

def xlsx_to_dict(xlsx_file: str) -> tuple:
    """
    returns 4D data block in dictionary format from input xls file
    
    arguments:
        xlsx_file [str]: name of file (including file extension)
    returns:
        fluid [str]: fluid name
        prop_dict [dict]: dictionary
    """
    
    # remove file prefixes/suffixes
    fluid = xlsx_file.name.replace('.xlsx', '')
    fluid = fluid.replace('lookup_', '')
    
    # create excel object and create dataFrame
    xls = pd.ExcelFile(xlsx_file)
    df_master = pd.read_excel(xls, sheet_name="master")
    df_lookup = pd.read_excel(xls, sheet_name=None, header=None)

    # get columns headings from df_master
    df_cols = df_master.columns.values.tolist()

    # extract master properties
    # write each col of df_master to a list
    df_str = [df_master[col].tolist() for col in df_cols]

    # remove empty spaces
    df_str_no_space = [str(col[0]).replace(' ', '') if isinstance(col[0], (int, float)) else col[0].replace(' ', '') for col in df_str]

    # extract CSV into lists
    df_list = [col.split(",") for col in df_str_no_space]

    # convert coordinate vals from string to float
    idx_coord = df_cols.index("coord_label")    #idx of coord_label
    str_coord = df_list[idx_coord]
    nCoord = len(str_coord)
    nX = [None] * nCoord
    for i,col in enumerate(str_coord):
        idx = df_cols.index(col)
        df_list[idx] = [float(item) for item in df_list[idx]]
        nX[i] = len(df_list[idx])

    # count num props
    idx_prop = df_cols.index("prop_label")  #idx of prop_label
    nProp = len(df_list[idx_prop])

    # preallocate `prop_table` (4D data block containing all properties as a function of the coordinates)
    prop_shape = tuple([nProp] + nX)
    prop_table = np.zeros(prop_shape, dtype=float)

    # loop through sheets ('TPF' only) and generate slice corresponding to each sheet
    i = 0
    for (sht_name, sht_df) in df_lookup.items():
        if "TPF" in sht_name:
            df = pd.read_excel(xls, sheet_name=sht_name, header=None)
            prop_TP = df.to_numpy()
            #prop_TP = sht_df.to_numpy()
            
            # each iteration is a slice of coord[1], coord[2] at current coord[0]
            for iX1, iX2 in itertools.product(range(nX[1]), range(nX[2])):
                # split each string and convert to a list of floats
                prop_str = prop_TP[iX1][iX2]
                prop_str_no_space = prop_str.replace(' ', '')
                prop_val = prop_str_no_space.split(",")
                prop_val = [float(item) for item in prop_val]
                
                # populate prop_table
                prop_table[:,i,iX1,iX2] = prop_val

            # next coord[0] slice     
            i += 1

    # close xls file
    xls.close
    
    # put prop_table in df_list
    idx_prop = df_cols.index("prop_table")  #idx of prop_table
    df_list[idx_prop] = prop_table.tolist()
    prop_dict = {'fluid': fluid}

    for i in range(len(df_cols)):
        prop_dict[df_cols[i]] = df_list[i]

    # create json object from dict
    return fluid, prop_dict


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'data'
xlsx = DATA_DIR / 'xls' / 'lookup_S14-SAFT-MILA2020.xlsx'

fluid, dictionary = xlsx_to_dict(xlsx)

with open(DATA_DIR / 'json' / f'{fluid}.json', 'w') as f:
    json.dump(dictionary, f)