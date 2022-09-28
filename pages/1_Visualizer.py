#%% import pkgs
# from config import DATA_DIR, ROOT_DIR
# from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from io import BytesIO
import plotly_express as px
import streamlit as st
import fileConverter as fc


#%% streamlit app properties
# Page configuration (necessary for streamlit to work, must be at the beginning of program)
st.set_page_config(page_title='Visualizer', page_icon=':bar_chart:', layout='wide')


#%% -- functions
def get_idx(X: np.array, a: str) -> str:
    """
    Gets index of coordinate in its corresponding range of possible values
    arguments:
        X: coordinate range of values
        a: specific coordinate value
    returns:
        index position of a
    """
    return np.where(X == a)[0][0]

# DataFrame converter: DF -> csv/excel/json
def convert_df(df: pd.DataFrame, to_type: str):
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

def interruptor(df: pd.DataFrame, fig, xaxis, yaxis, type_series):
    tab4.plotly_chart(fig)
    with tab5:
        st.dataframe(df.copy())
        to_type = st.radio('Select data format:', ('csv', 'xlsx', 'json'))
        nombre = f'{xaxis}_vs_{yaxis}_{type_series}.{to_type}'
        st.download_button(label="Download data", data=convert_df(df, to_type), file_name=nombre)

def isocurve(type_series: str) -> tuple[np.array]:
    """
    returns:
        nCoord_array: array of values of the multiple isocurves
    """
    idx = get_idx(const_coord, type_series)
    minimo=coord_range[idx][0]
    maximo=coord_range[idx][-1]

    # add/delete buttons
    cont2 = tab3.container()
    col1, col2 = tab3.columns(2)
    if 'count' not in st.session_state:
        st.session_state.count = 0

    new_input = col1.button("Add input box")
    delete_input = col2.button("Delete input box")

    if new_input:
        st.session_state.count += 1
    elif delete_input and st.session_state.count > 0:
        st.session_state.count -= 1

    with cont2:
        nCoord_array = [st.number_input(f'Enter a {coord[idx]}:', min_value=minimo, max_value=maximo, key=i) 
                        for i in range(st.session_state.count)]

    return np.array(nCoord_array)

def coordinates(axis: str, type_series: str) -> tuple[str]:
    """
    Gets value of single coordinate (number input at bottom of sidebar)

    returns:
        coord_value: value of coordinate 
        titulo: plot title
    """
    type_idx = get_idx(const_coord, type_series)
    axis_idx = get_idx(coord, axis)
    idx = np.delete(np.array([0, 1, 2]), [axis_idx, type_idx])[0]
    minimo=coord_range[idx][0]
    maximo=coord_range[idx][-1]
    coord_value = tab3.number_input(f'Enter a {coord[idx]}:', min_value=minimo, max_value=maximo)

    a = (f'{coord_label[0]}: {str(coord_value)} {coord_unit[0]}', 
        f'{coord_label[1]}: {str(coord_value)} {coord_unit[1]}', 
        f'{coord_label[2]}: {str(coord_value)} {coord_unit[2]}')
    titulo = a[idx]
    return coord_value, titulo

def matrix_for_df_creator(i, j, coord_value, nCoord_array, axis, type_series) -> list:
    if axis == coord[0] and type_series == f'constant {coord_label[1]}':
        nXY = [coord_range[0][j], nCoord_array[i], coord_value]
    elif axis == coord[0] and type_series == f'constant {coord_label[2]}':
        nXY = [coord_range[0][j], coord_value, nCoord_array[i]]
    elif axis == coord[1] and type_series == f'constant {coord_label[0]}':
        nXY = [nCoord_array[i], coord_range[1][j], coord_value]
    elif axis == coord[1] and type_series == f'constant {coord_label[2]}':
        nXY = [coord_value, coord_range[1][j], nCoord_array[i]]
    elif axis == coord[2] and type_series == f'constant {coord_label[0]}':
        nXY = [nCoord_array[i], coord_value, coord_range[2][j]]
    elif axis == coord[2] and type_series == f'constant {coord_label[1]}':
        nXY = [coord_value, nCoord_array[i], coord_range[2][j]]
    return nXY

def df_creator(axis, type_series, iso, nCoord_array, coord_value, nprop, rango_coord, df) -> pd.DataFrame:
    """
    Creates DF that is used to make the plot and table, reads data from 4D array and interpolates
    """
    for i in range(len(nCoord_array)):
        f = np.empty_like(rango_coord)
        for j in range(len(rango_coord)):
            nXY = matrix_for_df_creator(i, j, coord_value, nCoord_array, axis, type_series)
            pt = np.array([nprop] + nXY)
            f[j] = interp(pt)
        df[iso[i]] = f
    return df

def xarray_creator(xaxis, yaxis, nprop, xidx, yidx, z_value):
    Data = np.empty((len(coord_range[xidx]), len(coord_range[yidx])), dtype='float64')
    for i in range(len(coord_range[xidx])):
        for j in range(len(coord_range[yidx])):
            if xaxis == coord[0] and yaxis == coord[1]:
                nXY = [coord_range[0][i], coord_range[1][j], z_value]
            elif xaxis == coord[0] and yaxis == coord[2]:
                nXY = [coord_range[0][i], z_value, coord_range[2][j]]
            elif xaxis == coord[1] and yaxis == coord[0]:
                nXY = [coord_range[0][j], coord_range[1][i], z_value]
            elif xaxis == coord[1] and yaxis == coord[2]:
                nXY = [z_value, coord_range[1][i], coord_range[2][j]]
            elif xaxis == coord[2] and yaxis == coord[0]:
                nXY = [coord_range[0][j], z_value, coord_range[2][i]]
            elif xaxis == coord[2] and yaxis == coord[1]:
                nXY = [z_value, coord_range[1][j], coord_range[2][i]]
            pt = np.array([nprop] + nXY)
            Data[i, j] = interp(pt)
    Data = xr.DataArray(Data, dims=(xaxis, yaxis), coords={xaxis: coord_range[xidx], yaxis: coord_range[yidx]})
    return Data

def fig_creator(xaxis, yaxis, axis1, axis2, titulo, nCoord_array_str, df, idx2):
    fig = px.scatter(title=titulo)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis, legend_title=f'{coord[idx2]}:')
    for i in range(len(nCoord_array_str)):
        if xaxis in coord:
            fig.add_scatter(x=df[axis1], y=df[axis2[i]], name=nCoord_array_str[i], mode='markers')
        elif yaxis in coord:
            fig.add_scatter(x=df[axis2[i]], y=df[axis1], name=nCoord_array_str[i], mode='markers')
        else:
            fig.add_scatter(x=df[axis1[i]], y=df[axis2[i]], name=nCoord_array_str[i], mode='markers')
    return fig

def coord_on_one_axis(xaxis: str, yaxis: str, type_series: str):
    """
    This function is called when the xaxis or the yaxis is a coordinate (GOR, T, P)
    returns:
        plot, table, file
    """
    if xaxis in coord:
        axis = xaxis
        nprop = get_idx(prop, yaxis)
    elif yaxis in coord:
        axis = yaxis
        nprop = get_idx(prop, xaxis)
    
    nCoord_array = isocurve(type_series)
    coord_value, titulo = coordinates(axis, type_series)

    idx = get_idx(coord, axis)
    idx2 = get_idx(const_coord, type_series)

    nCoord_array_str = nCoord_array.astype('str')
    iso1 = [f'{nCoord_array_str[i]} {coord_unit[idx2]}' for i in range(len(nCoord_array))]
    iso1 = np.array(iso1)

    df = pd.DataFrame()
    df[axis] = coord_range[idx]
    df = df_creator(axis, type_series, iso1, nCoord_array, coord_value, nprop, coord_range[idx], df)

    fig = fig_creator(xaxis, yaxis, axis, iso1, titulo, nCoord_array_str, df, idx2)     
    interruptor(df, fig, xaxis, yaxis, type_series)

def coord_on_both_axes(xaxis, yaxis, Fixed):
    nprop = get_idx(prop, Fixed)
    xidx = get_idx(coord, xaxis)
    yidx = get_idx(coord, yaxis)

    zidx = np.delete(np.array([0, 1, 2]), [xidx, yidx])[0]
    minimo=coord_range[zidx][0]
    maximo=coord_range[zidx][-1]
    z_value = tab3.number_input(f'Choose a {coord[zidx]}:', min_value=minimo, max_value=maximo)

    Data = xarray_creator(xaxis, yaxis, nprop, xidx, yidx, z_value)
    fig = px.imshow(Data.T, labels={'color': Fixed})
    fig.update_layout(title=f'{coord[zidx]}: {z_value}')
    df = Data.to_pandas()

    interruptor(df, fig, xaxis, yaxis, Fixed)

def prop_on_both_axes(xaxis: str, yaxis: str, type_series: str, Not_fixed: str):
    """
    This function is called when both the xaxis and the yaxis are a property
    returns:
        plot, table, file
    """
    nCoord_array = isocurve(type_series)
    coord_value, titulo = coordinates(Not_fixed, type_series)

    nprop1 = get_idx(prop, xaxis)
    nprop2 = get_idx(prop, yaxis)

    idx = get_idx(coord, Not_fixed)
    idx2 = get_idx(const_coord, type_series)

    nCoord_array_str = nCoord_array.astype('str')
    iso1 = [f'{xaxis} {nCoord_array_str[i]} {coord_unit[idx2]}' for i in range(len(nCoord_array))]
    iso1 = np.array(iso1)
    iso2 = [f'{yaxis} {nCoord_array_str[i]} {coord_unit[idx2]}' for i in range(len(nCoord_array))]
    iso2 = np.array(iso2)

    df = pd.DataFrame()
    df[Not_fixed] = coord_range[idx]
    df = df_creator(Not_fixed, type_series, iso1, nCoord_array, coord_value, nprop1, coord_range[idx], df)
    df = df_creator(Not_fixed, type_series, iso2, nCoord_array, coord_value, nprop2, coord_range[idx], df)
                    
    fig = fig_creator(xaxis, yaxis, iso1, iso2, titulo, nCoord_array_str, df, idx2)    
    interruptor(df, fig, xaxis, yaxis, type_series)

# --/ functions


#%%
# containers to structure page
maincont = st.container()
cont1 = st.container()
with cont1:
    tab1, tab2, tab3 = st.sidebar.tabs(["File uploader", "Choose axes", "Series type"])

# set file path 
# ruta = DATA_DIR / 'output' / 'S14-SAFT-MILA2020.json'

# File uploader
uploaded_file = tab1.file_uploader('Upload fluid file:', type = ('xlsx', 'json'), on_change=fc.clear_cache())

if uploaded_file is not None:
    if uploaded_file.type == 'application/json':
        fluid = uploaded_file.name.replace('.json', '')
        fluid = fluid.replace('lookup_', '')
        prop_json = uploaded_file
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        fluid, prop_json = fc.xlsx_to_json(uploaded_file)
  
    # properties from json
    coord_label, coord_unit, coord_range, prop_label, prop_unit, prop_table = fc.get_data_from_json(prop_json)

    # some useful arrays
    const_coord = np.array([f'constant {coord_label[0]}', f'constant {coord_label[1]}', f'constant {coord_label[2]}'])
    coord = [f"{coord_label[i]} [{coord_unit[i]}]" for i in range(len(coord_label))]    
    prop = [f"{prop_label[i]} [{prop_unit[i]}]" for i in range(len(prop_label))]        
    variables = np.array(coord + prop)  #coordinate and property labels + units
    coord = np.array(coord)             #coordinate labels + units
    prop = np.array(prop)               #property labels + units
    const_coord_dict = {coord[0]: (f'constant {coord_label[1]}', f'constant {coord_label[2]}'), 
                        coord[1]: (f'constant {coord_label[0]}', f'constant {coord_label[2]}'), 
                        coord[2]: (f'constant {coord_label[0]}', f'constant {coord_label[1]}')}

    # needed for interpolation
    prop_range = np.linspace(0, len(prop_label) - 1, len(prop_label))
    interp = RegularGridInterpolator((prop_range, coord_range[0], coord_range[1], coord_range[2]), prop_table)

    # container for page
    maincont.header(fluid)
    with maincont:
        tab4, tab5 = st.tabs(("Plot", "Table"))

    # sidebar filters
    tab2.header('Choose axes:')
    xaxis = tab2.selectbox('Select x-axis:', variables)
    yaxis = tab2.selectbox('Select y-axis:', variables) 

    # choose what to display on main page from user selection
    if xaxis in coord and yaxis in coord and xaxis == yaxis:
        st.warning("Choose another combination of axes")
    elif xaxis in coord and yaxis in coord:
        Fixed = tab3.selectbox('Choose a property for the color plot', prop)
        coord_on_both_axes(xaxis, yaxis, Fixed)
    elif xaxis in coord and yaxis in prop:   
        type_series = tab3.radio('Choose the type of the series:', const_coord_dict[xaxis])
        coord_on_one_axis(xaxis, yaxis, type_series)
    elif xaxis in prop and yaxis in coord:
        type_series = tab3.radio('Choose the type of the series:', const_coord_dict[yaxis])
        coord_on_one_axis(xaxis, yaxis, type_series) 
    elif xaxis in prop and yaxis in prop:
        Not_fixed = tab2.selectbox('Which parameter should vary?', coord)
        type_series = tab3.radio('Choose the type of the series:', const_coord_dict[Not_fixed])
        prop_on_both_axes(xaxis, yaxis, type_series, Not_fixed) 

    st.sidebar.markdown("#")    #empty space at the bottom of sidebar
else:
    st.warning('Please upload a fluid file to begin')
    st.session_state.count = 0
