#%% import pkgs
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from scipy import interpolate as ip
import json
import streamlit as st
import itertools
import openpyxl
from io import BytesIO


#%% functions
@st.experimental_memo(show_spinner=False)
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

@st.experimental_memo(show_spinner=False)
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
    prop_table =  np.asarray(df.at['prop_table', 0], dtype=np.float64)  # 4D block of data containing all properties (nProps, nGOR, nP, nT)
    prop_table =  preproc(prop_table, coord_range, prop_label)       # interpolates failed calculations

    return coord_label, coord_unit, coord_range, prop_label, prop_unit, prop_table

@st.experimental_memo
def convert_df(df: pd.DataFrame, to_type: str):
    """
    DataFrame converter: DF -> csv/excel/json
    """
    if to_type == 'csv':
        return df.to_csv().encode('utf-8') 
    elif to_type == 'json':
        return df.to_json(orient='index').encode('utf-8')   
    elif to_type == 'xlsx':
        output = BytesIO()
        writer = pd.ExcelWriter(output)
        df.to_excel(writer)
        writer.save()
        return output.getvalue()

def f_founder(arr):
    c1 = nd.uniform_filter(arr, size=3)
    c2 = nd.uniform_filter(arr*arr, size=3)
    var = c2 - c1*c1
    var = np.where(var < 1e-7, 1e-7, var)
    std = np.sqrt(var)
    mu = c1
    z = np.abs((arr - mu)/std)      # z-score with abs
    z = np.where(z == np.inf, 0, z)
    return np.where(z >= 1)

@st.experimental_memo(show_spinner=False)
def preproc(ptable, crange, plabel):
    x = range(len(plabel))
    y = range(len(crange[0]))
    err = []
    for X1 in y:
        arr = ptable[0, X1]     # get the failures of the first prop only, all props fail at the same places
        err.append(f_founder(arr))
    err = tuple(err)
    for X0, X1 in itertools.product(x, y):
        arr = ptable[X0, X1]
        e = err[X1]
        arr[e] = np.nan
        arr = np.ma.masked_invalid(arr)
        xx, yy = np.meshgrid(crange[2], crange[1])
        # get only the valid values
        x1 = xx[~arr.mask]
        y1 = yy[~arr.mask]
        newarr = arr[~arr.mask]
        newarr = ip.griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic')
        # https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
        ptable[X0, X1] = newarr
    return ptable

#--/ functions