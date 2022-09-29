#%% import pkgs
#from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import json
import streamlit as st
import openpyxl
import subprocess
#from config import ROOT_DIR, DATA_DIR
#from pathlib import Path

#%% functions
@st.cache
def xlsx_to_json(uploaded_file: str) -> tuple:
    """
    returns 4D data block in json format from input xls file
    
    arguments:
        uploaded_file [str]: name of file (including file extension)
    return:
        fluid [str]: fluid name
        dict [json]: json file
    """
    
    # remove file prefixes/suffixes
    fluid = uploaded_file.name.replace('.xlsx', '')
    fluid = fluid.replace('lookup_', '')
    
    # create excel object and create dataFrame
    xls = pd.ExcelFile(uploaded_file)
    df_master = pd.read_excel(xls, sheet_name="master")
    df_lookup = pd.read_excel(xls, sheet_name=None, header=None)

    # get columns headings from df_master
    df_cols = df_master.columns.values.tolist()

    # extract master properties
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

    # count num props
    idx_prop = df_cols.index("prop_label")  #idx of prop_label
    nProp = len(df_list[idx_prop])

    # preallocate `prop_table` (4D data block containing all properties as a function of the coordinates)
    prop_shape = tuple([nProp] + nX)
    prop_table = np.zeros(prop_shape, dtype=float)

    # loop through sheets ('TPF' only) and generate slice corresponding to each sheet
    import itertools
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
                prop_val = prop_str.split(", ")
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
    return fluid, json.dumps(prop_dict)

def clear_cache():
    # Delete all the items in Session state
    # for key in st.session_state.keys():
    #     del st.session_state[key]
    subprocess.run('streamlit cache clear')

@st.cache
def get_data_from_json(prop_json) -> tuple:
    """
    extract data from json file to np.array
    data: fluid properties (Ceq, dens, visco, etc) = f(coord: GOR,P,T)
    
    arguments:
        prop_json: `.json` file
    returns:
        coord_label [np.array[str]]: coordinate labels (ex: GOR, P, T)
        coord_unit [np.array[str]]: unit corresponding to coordinates (ex: scf/stb, psig, F) 
        coord_range [np.array[float]]: value ranges corresponding to coordinates
        prop_label [np.array[str]]: property labels (ex: dens_L1, visco_L2)
        prop_unit [np.array[str]]: units corresponding to property strings (ex: g/cc, cP)        
        prop_table [np.array[float]]: 4D block (nProp x nGOR x nP x nT) of data containing all properties: Props = f(GOR, P, T) 
    """
    # convert json to df
    df = pd.read_json(prop_json, orient='index')

    # extract properties
    coord_label = np.array(df.at['coord_label', 0])
    coord_unit = np.array(df.at['coord_unit', 0])
    prop_label = np.array(df.at['prop_label', 0])
    prop_unit = np.array(df.at['prop_unit', 0])
    coord_1 = np.array(df.at[coord_label[0], 0])
    coord_2 = np.array(df.at[coord_label[1], 0])
    coord_2 = np.round(coord_2, 2)
    coord_3 = np.array(df.at[coord_label[2], 0])
    coord_3 = np.round(coord_3, 2)
    coord_range = (coord_1, coord_2, coord_3)
    prop_table =  np.asarray(df.at['prop_table', 0], dtype=float)  # 4D block of data containing all properties (nProps, nGOR, nP, nT)
    
    return coord_label, coord_unit, coord_range, prop_label, prop_unit, prop_table

#--/ functions