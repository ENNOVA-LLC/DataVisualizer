#%% import pkgs
# from config import DATA_DIR, ROOT_DIR
# from pathlib import Path
import numpy as np
import pandas as pd
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
    return st.sidebar.radio('Choose the type of the series:', const_coord_dict[axis])

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
    with cont4:
        switch = st.radio('Switch between plot and table:', ('Plot', 'Table'), horizontal=True)
        if switch == 'Plot':
            st.plotly_chart(fig)
        else:
            st.dataframe(df)
        to_type = st.radio('Select the data format:', ('csv', 'xlsx', 'json'))
        nombre = f'{xaxis}_vs_{yaxis}_{type_series}.{to_type}'
        st.download_button(label="Download data", data=convert_df(df, to_type), file_name=nombre)

def isocurve(type_series: str) -> tuple[np.array]:
    """
    returns:
        iso: array of strings of the selected isocurves
        nCoord_array: array of indices of the multiple isocurves
    """
    idx = get_idx(const_coord, type_series)
    iso = st.sidebar.multiselect(f'Choose from the available {coord[idx]}:', coord_range[idx])
    nCoord_array = [get_idx(coord_range[idx], iso[i]) for i in range(len(iso))]

    iso = np.array(iso)
    iso = iso.astype('str')
    return iso, np.array(nCoord_array)

def coordinates(axis: str, type_series: str) -> tuple[str]:
    """
    Gets index of coordinate selected by slider

    returns:
        nCoord_idx: index of coordinate 
        titulo: plot title
    """
    type_idx = get_idx(const_coord, type_series)
    axis_idx = get_idx(coord, axis)
    idx = np.delete(np.array([0, 1, 2]), [axis_idx, type_idx])
    coord_value = st.sidebar.select_slider(f'Select a {coord[idx][0]}:', coord_range[idx][0])
    nCoord_idx = get_idx(coord_range[idx][0], coord_value)

    a = (f'{coord_label[0]}: {str(coord_value)} {coord_unit[0]}', 
        f'{coord_label[1]}: {str(coord_value)} {coord_unit[1]}', 
        f'{coord_label[2]}: {str(coord_value)} {coord_unit[2]}')
    titulo = a[idx[0]]
    return nCoord_idx, titulo

def matrix_for_df_creator(i, j, nCoord_idx, nCoord_array, axis, type_series) -> list:
    A = pd.DataFrame(index=coord, columns=const_coord)
    A.at[coord[0], f'constant {coord_label[2]}'] = [j, nCoord_idx, int(nCoord_array[i])]
    A.at[coord[0], f'constant {coord_label[1]}'] = [j, int(nCoord_array[i]), nCoord_idx]
    A.at[coord[2], f'constant {coord_label[0]}'] = [int(nCoord_array[i]), nCoord_idx, j]
    A.at[coord[2], f'constant {coord_label[1]}'] = [nCoord_idx, int(nCoord_array[i]), j]
    A.at[coord[1], f'constant {coord_label[0]}'] = [int(nCoord_array[i]), j, nCoord_idx]
    A.at[coord[1], f'constant {coord_label[2]}'] = [nCoord_idx, j, int(nCoord_array[i])]
    return A.at[axis, type_series]

def df_creator(axis, type_series, iso, nCoord_array, nCoord_idx, nprop, rango_coord, df) -> pd.DataFrame:
    """
    Creates DF that is used to make the plot and table, reads data from 4D array
    """
    for i in range(len(nCoord_array)):
        f = np.empty_like(rango_coord)
        for j in range(len(rango_coord)):
            nXY = matrix_for_df_creator(i, j, nCoord_idx, nCoord_array, axis, type_series)
            idx = tuple([nprop] + nXY)
            f[j] = prop_table[idx]
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

    with cont3:
        iso, nCoord_array = isocurve(type_series)
        nCoord_idx, titulo = coordinates(axis, type_series)

    idx = get_idx(variables, axis)
    rango_coord = coord_range[idx]
    idx2 = get_idx(const_coord, type_series)

    iso1 = np.empty_like(iso)
    for i in range(len(nCoord_array)):
        iso1[i] = f'{iso[i]} {coord_unit[idx2]}'

    df = pd.DataFrame()
    df[axis] = rango_coord
    df = df_creator(axis, type_series, iso1, nCoord_array, nCoord_idx, nprop, rango_coord, df)

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
    with cont3:
        iso, nCoord_array = isocurve(type_series)
        nCoord_idx, titulo = coordinates(Not_fixed, type_series)

    nprop1 = get_idx(prop, xaxis)
    nprop2 = get_idx(prop, yaxis)

    idx = get_idx(variables, Not_fixed)
    rango_coord = coord_range[idx]
    idx2 = get_idx(const_coord, type_series)

    iso1 = np.empty_like(iso)
    iso2 = np.empty_like(iso)
    for i in range(len(nCoord_array)):
        iso1[i] = f'{xaxis} {iso[i]} {coord_unit[idx2]}'
        iso2[i] = f'{yaxis} {iso[i]} {coord_unit[idx2]}'

    df = pd.DataFrame()
    df[Not_fixed] = rango_coord
    df = df_creator(Not_fixed, type_series, iso1, nCoord_array, nCoord_idx, nprop1, rango_coord, df)
    df = df_creator(Not_fixed, type_series, iso2, nCoord_array, nCoord_idx, nprop2, rango_coord, df)
                    
    fig = px.scatter(title=titulo)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis, legend_title=f'{coord[idx2]}:')
    for i in range(len(nCoord_array)):
        fig.add_scatter(x=df[iso1[i]], y=df[iso2[i]], name=iso[i], mode='markers')

    interruptor(df, fig, xaxis, yaxis, type_series)

# --/ functions


#%%
# containers to separate content and allow dynamic selection
maincont = st.container()
cont1 = st.container()
cont2 = st.container()
cont3 = st.container()
cont4 = st.container()

# set file path 
# ruta = DATA_DIR / 'output' / 'S14-SAFT-MILA2020.json'

# File uploader
uploaded_file = st.sidebar.file_uploader('Upload fluid file:', type = ('xlsx', 'json'))

if uploaded_file is not None:
    with maincont:
        if uploaded_file.type == 'application/json':
            # st.success("json file successfully uploaded")
            fluid = uploaded_file.name.replace('.json', '')
            fluid = fluid.replace('lookup_', '')
            prop_json = uploaded_file
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # st.success("xlsx file successfully uploaded")
            fluid, prop_json = fc.xlsx_to_json(uploaded_file)
        
        # properties from json
        coord_label, coord_unit, coord_range, prop_label, prop_unit, prop_table = fc.get_data_from_json(prop_json)
        st.write(prop_table[16, 3, 0, 0])

        # some useful arrays
        const_coord = np.array([f'constant {coord_label[0]}', f'constant {coord_label[1]}', f'constant {coord_label[2]}'])
        coord = [f"{coord_label[i]} [{coord_unit[i]}]" for i in range(len(coord_label))]    
        prop = [f"{prop_label[i]} [{prop_unit[i]}]" for i in range(len(prop_label))]        
        variables = np.array(coord + prop)
        coord = np.array(coord)     #coordinate labels + units
        prop = np.array(prop)       #property labels + units

        st.write(f"# {fluid}")

        # Sidebar filters
        st.sidebar.header('Choose axes:')
        xaxis = st.sidebar.selectbox('Select x-axis:', variables)
        
        # choose what to display on main page from user selection
        if xaxis in coord:
            with cont1:
                yaxis = st.sidebar.selectbox('Select y-axis:', prop)
                
            with cont2:    
                type_series = series_options(xaxis)

            coord_on_axis(xaxis, yaxis, type_series)
        else:           #xaxis == Property
            with cont1:
                yaxis = st.sidebar.selectbox('Select y-axis:', variables)

            if yaxis in coord: 
                with cont2:
                    type_series = series_options(yaxis)

                coord_on_axis(xaxis, yaxis, type_series) 
            else:       #yaxis == Property 
                with cont1:
                    Not_fixed = st.sidebar.selectbox('Which parameter should vary?', coord)

                with cont2:
                    type_series = series_options(Not_fixed)
                    
                prop_vs_prop(xaxis, yaxis, type_series, Not_fixed) 

        st.sidebar.markdown("#")
else:
    with maincont:
        st.warning('Please upload a fluid file to begin')