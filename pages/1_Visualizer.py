#%% import pkgs
# from config import DATA_DIR, ROOT_DIR
# from pathlib import Path
import numpy as np
import pandas as pd
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

def series_options(axis: str) -> st.radio:
    """
    Dropdown menu of the types of series available
    """
    const_coord_dict = {coord[0]: (f'constant {coord_label[1]}', f'constant {coord_label[2]}'), 
                    coord[1]: (f'constant {coord_label[0]}', f'constant {coord_label[2]}'), 
                    coord[2]: (f'constant {coord_label[0]}', f'constant {coord_label[1]}')}
    return tab3.radio('Choose the type of the series:', const_coord_dict[axis])

# DataFrame converter: DF -> csv/excel/json
# cjsisco note: should be able to output in different formats
def convert_df(df: pd.DataFrame, to_type: str):
    if to_type == 'csv':
        return df.to_csv().encode('utf-8') 
    elif to_type == 'json':
        return df.to_json(orient='index').encode('utf-8')   
    elif to_type == 'xlsx':
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False)
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
        iso: array of strings of the selected isocurves
        nCoord_array: array of indices of the multiple isocurves
    """
    idx = get_idx(const_coord, type_series)
    iso = tab3.multiselect(f'Choose from the available {coord[idx]}:', coord_range[idx])
    nCoord_array = [get_idx(coord_range[idx], iso[i]) for i in range(len(iso))]

    iso = np.array(iso)
    iso = iso.astype('str')
    return iso, np.array(nCoord_array)

def coordinates(axis: str, type_series: str) -> tuple[str]:
    """
    Gets value of single coordinate (number input at bottom of sidebar)

    returns:
        coord_value: value of coordinate 
        titulo: plot title
    """
    type_idx = get_idx(const_coord, type_series)
    axis_idx = get_idx(coord, axis)
    idx = np.delete(np.array([0, 1, 2]), [axis_idx, type_idx])
    minimo=coord_range[idx][0][0]
    maximo=coord_range[idx][0][-1]
    coord_value = tab3.number_input(f'Enter a {coord[idx][0]}:', min_value=minimo, max_value=maximo)

    a = (f'{coord_label[0]}: {str(coord_value)} {coord_unit[0]}', 
        f'{coord_label[1]}: {str(coord_value)} {coord_unit[1]}', 
        f'{coord_label[2]}: {str(coord_value)} {coord_unit[2]}')
    titulo = a[idx[0]]
    return coord_value, titulo

def matrix_for_df_creator(i, j, coord_value, nCoord_array, axis, type_series) -> list:
    k = int(nCoord_array[i])
    if axis == coord[0] and type_series == f'constant {coord_label[1]}':
        nXY = [coord_range[0][j], coord_range[1][k], coord_value]
    elif axis == coord[0] and type_series == f'constant {coord_label[2]}':
        nXY = [coord_range[0][j], coord_value, coord_range[2][k]]
    elif axis == coord[1] and type_series == f'constant {coord_label[0]}':
        nXY = [coord_range[0][k], coord_range[1][j], coord_value]
    elif axis == coord[1] and type_series == f'constant {coord_label[2]}':
        nXY = [coord_value, coord_range[1][j], coord_range[2][k]]
    elif axis == coord[2] and type_series == f'constant {coord_label[0]}':
        nXY = [coord_range[0][k], coord_value, coord_range[2][j]]
    elif axis == coord[2] and type_series == f'constant {coord_label[1]}':
        nXY = [coord_value, coord_range[1][k], coord_range[2][j]]
    return nXY

def df_creator(axis, type_series, iso, nCoord_array, coord_value, nprop, rango_coord, df) -> pd.DataFrame:
    """
    Creates DF that is used to make the plot and table, reads data from 4D array and interpolates
    """
    interp = RegularGridInterpolator((prop_range, coord_range[0], coord_range[1], coord_range[2]), prop_table)
    for i in range(len(nCoord_array)):
        f = np.empty_like(rango_coord)
        for j in range(len(rango_coord)):
            nXY = matrix_for_df_creator(i, j, coord_value, nCoord_array, axis, type_series)
            pt = np.array([nprop] + nXY)
            f[j] = interp(pt)
        df[iso[i]] = f
    return df

def coord_on_axis(xaxis: str, yaxis: str, type_series: str):
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
    
    iso, nCoord_array = isocurve(type_series)
    coord_value, titulo = coordinates(axis, type_series)

    idx = get_idx(coord, axis)
    idx2 = get_idx(const_coord, type_series)

    iso1 = np.empty_like(iso)
    for i in range(len(nCoord_array)):
        iso1[i] = f'{iso[i]} {coord_unit[idx2]}'

    df = pd.DataFrame()
    df[axis] = coord_range[idx]
    df = df_creator(axis, type_series, iso1, nCoord_array, coord_value, nprop, coord_range[idx], df)

    if xaxis in coord:
        fig = px.scatter(df, x=axis, y=iso1, title=titulo, labels={'value': yaxis, 'variable': f'{coord_label[idx2]}:'})
    elif yaxis in coord:
        fig = px.scatter(df, x=iso1, y=axis, title=titulo, labels={'value': xaxis, 'variable': f'{coord_label[idx2]}:'})

    interruptor(df, fig, xaxis, yaxis, type_series)

def prop_vs_prop(xaxis: str, yaxis: str, type_series: str, Not_fixed: str):
    """
    This function is called when both the xaxis and the yaxis are a property
    returns:
        plot, table, file
    """
    iso, nCoord_array = isocurve(type_series)
    coord_value, titulo = coordinates(Not_fixed, type_series)

    nprop1 = get_idx(prop, xaxis)
    nprop2 = get_idx(prop, yaxis)

    idx = get_idx(coord, Not_fixed)
    idx2 = get_idx(const_coord, type_series)

    iso1 = np.empty_like(iso)
    iso2 = np.empty_like(iso)
    for i in range(len(nCoord_array)):
        iso1[i] = f'{xaxis} {iso[i]} {coord_unit[idx2]}'
        iso2[i] = f'{yaxis} {iso[i]} {coord_unit[idx2]}'

    df = pd.DataFrame()
    df[Not_fixed] = coord_range[idx]
    df = df_creator(Not_fixed, type_series, iso1, nCoord_array, coord_value, nprop1, coord_range[idx], df)
    df = df_creator(Not_fixed, type_series, iso2, nCoord_array, coord_value, nprop2, coord_range[idx], df)
                    
    fig = px.scatter(title=titulo)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis, legend_title=f'{coord[idx2]}:')
    for i in range(len(nCoord_array)):
        fig.add_scatter(x=df[iso1[i]], y=df[iso2[i]], name=iso[i], mode='markers')

    interruptor(df, fig, xaxis, yaxis, type_series)

# --/ functions


#%%
# containers to structure page
maincont = st.container()
cont1 = st.container()
cont2 = st.container()
cont3 = st.container()
tab1, tab2, tab3 = st.sidebar.tabs(["File uploader", "Choose axes", "Series type"])

# set file path 
# ruta = DATA_DIR / 'output' / 'S14-SAFT-MILA2020.json'

# File uploader
uploaded_file = tab1.file_uploader('Upload fluid file:', type = ('xlsx', 'json'))

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
    variables = np.array(coord + prop)
    coord = np.array(coord)     #coordinate labels + units
    prop = np.array(prop)       #property labels + units
    prop_range = np.linspace(0, len(prop_label) - 1, len(prop_label))

    maincont.header(fluid)
    with maincont:
        tab4, tab5 = st.tabs(["Plot", "Table"])

    # Sidebar filters
    with cont1:
        tab2.header('Choose axes:')
        xaxis = tab2.selectbox('Select x-axis:', variables)
    
    # choose what to display on main page from user selection
    if xaxis in coord:
        yaxis = tab2.selectbox('Select y-axis:', prop)    
        type_series = series_options(xaxis)
        coord_on_axis(xaxis, yaxis, type_series)
    else:           #xaxis == Property
        yaxis = tab2.selectbox('Select y-axis:', variables)

        if yaxis in coord: 
            type_series = series_options(yaxis)
            coord_on_axis(xaxis, yaxis, type_series) 
        else:       #yaxis == Property 
            Not_fixed = tab2.selectbox('Which parameter should vary?', coord)
            type_series = series_options(Not_fixed)
            prop_vs_prop(xaxis, yaxis, type_series, Not_fixed) 

    st.sidebar.markdown("#")
else:
    st.warning('Please upload a fluid file to begin')
# %%
