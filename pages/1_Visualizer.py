#%% import pkgs
import itertools
import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate as ip
import plotly_express as px
import streamlit as st
import fileConverter as fc
import kaleido

# streamlit app properties
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

def clear_cache():
    # Delete all keys in Session state
    for key in st.session_state.keys():
        del st.session_state[key]
    # Clear values from *all* memoized functions:
    st.experimental_memo.clear()

def reset_axes():
    """
    Resets range of axes
    """
    if 'axes' not in st.session_state:
        st.session_state.axes = 0

    st.session_state.axes += 1

def interruptor(df: pd.DataFrame, fig, xaxis, yaxis, type_series):
    tab5.plotly_chart(fig)
    with tab6:
        st.dataframe(df.copy())
        to_type = st.radio('Select data format:', ('csv', 'xlsx', 'json'))
        nombre = f'{xaxis}_vs_{yaxis}_{type_series}.{to_type}'
        st.download_button(label="Download data", data=fc.convert_df(df, to_type), file_name=nombre)

def prop_definer(prop_label, prop_table, prop, variables, new_prop):
    def D(array, axis):
        """
        Function to compute the gradient of a property
        """
        if axis == 0:
            dx = crange[0][1] - crange[0][0]   # dInjAmt/dGOR
        elif axis == 1:
            dx = crange[1][1] - crange[1][0]   # dP
        elif axis == 2:
            dx = crange[2][1] - crange[2][0]   # dT
        return np.gradient(array, dx, axis=axis)

    prop_list = remove_operators(new_prop)
    res = np.array([p for p in prop_list if p in prop_label])
    if new_prop != "":
        prop_str = new_prop
        if res.size != 0:
            for r in res:
                x = get_idx(prop_label, r)
                prop_str = prop_str.replace(r, f"prop_table[{x}, :, :, :]")

            try:
                y = eval(prop_str)
                y = np.array([y])
                prop_table = np.concatenate((prop_table, y), axis=0)
            except Exception:
                tab1.error("Check your input")
            else:
                prop_label = np.append(prop_label, new_prop)
                prop = np.append(prop, new_prop + " [-]")
                variables = np.append(variables, new_prop + " [-]")
                tab1.success("Property successfully added!")
        else:
            tab1.error("Check your input")
    return prop_label, prop_table, prop, variables

def remove_operators(new_prop):
    mod = new_prop.replace("/", " ")
    mod = mod.replace("+", " ")
    mod = mod.replace("-", " ")
    mod = mod.replace("*", " ")
    mod = mod.replace("(", " ")
    mod = mod.replace(")", " ")
    mod = mod.replace(",", " ")
    return mod.split()

def isocurve(type_series: str) -> tuple[np.array]:
    """
    returns:
        nCoord_array: array of values of the multiple isocurves
    """
    idx = get_idx(const_coord, type_series)
    minimo=crange[idx][0]
    maximo=crange[idx][-1]

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
        st.write(f'Enter a {coord[idx]}:')
        nCoord_array = [st.number_input(label="Enter value", min_value=minimo, max_value=maximo, key=i, label_visibility="collapsed") 
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
    minimo=crange[idx][0]
    maximo=crange[idx][-1]
    coord_value = tab3.number_input(f'Enter a {coord[idx]}:', min_value=minimo, max_value=maximo)

    titles = [f'{coord[i]}: {str(coord_value)}' for i in range(3)]
    titulo = titles[idx]
    return coord_value, titulo

def df_creator(axis, type_series, iso, nCoord_array, coord_value, nprop, rango_coord, df) -> pd.DataFrame:
    """
    Creates DF that is used to make the plot and table, reads data from 4D array and interpolates
    """
    for i, nCoord in enumerate(nCoord_array):
        f = np.empty_like(rango_coord)
        for j in range(len(rango_coord)):
            pt = df_point(j, coord_value, nCoord, axis, type_series, nprop)
            f[j] = interp(pt)
        df[iso[i]] = f
    return df

def df_point(j, coord_value, nCoord, axis, type_series, nprop) -> np.array:
    """
    Generates point for df_creator()
    """
    inner = [nprop]
    inner.extend(crange[k][j] if axis == coord[k] else nCoord if type_series == f'constant {coord_label[k]}' 
                else coord_value for k in range(3))
    return np.array(inner, dtype=np.float64)

def xarray_creator(xaxis, yaxis, nprop, xidx, yidx, z_value) -> xr.DataArray:
    """
    Creates xarray for heatmap plot
    """
    pt = xarray_point(xaxis, yaxis, z_value, nprop, xidx, yidx)
    Data = interp(pt)
    return xr.DataArray(Data, dims=(xaxis, yaxis), coords={xaxis: crange[xidx], yaxis: crange[yidx]})

def xarray_point(xaxis, yaxis, z_value, nprop, xidx, yidx) -> list:
    """
    Generates points for xarray_creator()
    """
    x = range(len(crange[xidx]))
    y = range(len(crange[yidx]))

    matrix = []
    for i in x:
        row = []
        for j in y:
            inner = [nprop]
            inner.extend(crange[k][i] if xaxis == coord[k] else crange[k][j] if yaxis == coord[k] 
                        else z_value for k in range(3))
            row.append(np.array(inner, dtype=np.float64))

        matrix.append(row)

    return matrix

def fig_creator(xaxis, yaxis, axis1, axis2, titulo, nCoord_array_str, df, idx2) -> px.scatter:
    fig = px.scatter(title=titulo)
    if np.size(nCoord_array_str) != 0:
        if xaxis in coord:
            for i, nCoord_str in enumerate(nCoord_array_str):
                fig.add_scatter(x=df[axis1], y=df[axis2[i]], name=nCoord_str, mode='markers',
                                hovertemplate=f'{xaxis}'+': %{x} <br>'+f'{yaxis}'+': %{y}')
        elif yaxis in coord:
            for i, nCoord_str in enumerate(nCoord_array_str):
                fig.add_scatter(x=df[axis2[i]], y=df[axis1], name=nCoord_str, mode='markers',
                                hovertemplate=f'{xaxis}'+': %{x} <br>'+f'{yaxis}'+': %{y}')
        else:
            for i, nCoord_str in enumerate(nCoord_array_str):
                fig.add_scatter(x=df[axis1[i]], y=df[axis2[i]], name=nCoord_str, mode='markers',
                                hovertemplate=f'{xaxis}'+': %{x} <br>'+f'{yaxis}'+': %{y}')
        fig = change_range(fig)

    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis, legend_title=f'{coord[idx2]}:', showlegend=True)
    return fig

def change_range(fig):
    # number input widgets are populated with default values 
    full_fig = fig.full_figure_for_development(warn=False)    # needed to access default range of axes (kaleido package)
    if tab4.button("Reset axes"):
        reset_axes()

    x_min = tab4.number_input("Min value of x-axis", value=full_fig.layout.xaxis.range[0], key=f"xmin_{st.session_state.axes}")
    x_max = tab4.number_input("Max value of x-axis", value=full_fig.layout.xaxis.range[1], key=f"xmax_{st.session_state.axes}")
    y_min = tab4.number_input("Min value of y-axis", value=full_fig.layout.yaxis.range[0], key=f"ymin_{st.session_state.axes}")
    y_max = tab4.number_input("Max value of y-axis", value=full_fig.layout.yaxis.range[1], key=f"ymax_{st.session_state.axes}")

    fig.update_yaxes(range=[y_min, y_max])
    fig.update_xaxes(range=[x_min, x_max])
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
    iso1 = [f'{nCoord_str} {coord_unit[idx2]}' for nCoord_str in nCoord_array_str]
    iso1 = np.array(iso1)

    df = pd.DataFrame()
    df[axis] = crange[idx]
    df = df_creator(axis, type_series, iso1, nCoord_array, coord_value, nprop, crange[idx], df)

    fig = fig_creator(xaxis, yaxis, axis, iso1, titulo, nCoord_array_str, df, idx2)
    interruptor(df, fig, xaxis, yaxis, type_series)

def coord_on_both_axes(xaxis, yaxis, Fixed):
    """
    This function is called whenever both axes are a coordinate, it returns a heatmap plot
    """
    nprop = get_idx(prop, Fixed)
    xidx = get_idx(coord, xaxis)
    yidx = get_idx(coord, yaxis)

    zidx = np.delete(np.array([0, 1, 2]), [xidx, yidx])[0]
    minimo=crange[zidx][0]
    maximo=crange[zidx][-1]
    z_value = tab3.number_input(f'Choose a {coord[zidx]}:', min_value=minimo, max_value=maximo)

    Data = xarray_creator(xaxis, yaxis, nprop, xidx, yidx, z_value)
    fig = px.imshow(Data.T, labels={'color': Fixed}, color_continuous_scale='Jet', origin='lower')
    fig.update_layout(title=f'{coord[zidx]}: {z_value}')
    if tab5.checkbox("Smooth plot"):
        fig.update_traces(zsmooth="best")
        
    fig = change_range(fig)
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
    iso1 = [f'{xaxis} {nCoord_str} {coord_unit[idx2]}' for nCoord_str in nCoord_array_str]
    iso1 = np.array(iso1)
    iso2 = [f'{yaxis} {nCoord_str} {coord_unit[idx2]}' for nCoord_str in nCoord_array_str]
    iso2 = np.array(iso2)

    df = pd.DataFrame()
    df[Not_fixed] = crange[idx]
    df = df_creator(Not_fixed, type_series, iso1, nCoord_array, coord_value, nprop1, crange[idx], df)
    df = df_creator(Not_fixed, type_series, iso2, nCoord_array, coord_value, nprop2, crange[idx], df)
                    
    fig = fig_creator(xaxis, yaxis, iso1, iso2, titulo, nCoord_array_str, df, idx2)    
    interruptor(df, fig, xaxis, yaxis, type_series)

# --/ functions


#%%
# containers to structure page
maincont = st.container()
cont1 = st.container()
with cont1:
    tab1, tab2, tab3, tab4 = st.sidebar.tabs(("File uploader", "Choose axes", "Series type", "Custom range"))

# set file path 
# ruta = DATA_DIR / 'output' / 'S14-SAFT-MILA2020.json'

# File uploader
# the on_change parameter expects a function object (without the parenthesis)
uploaded_file = tab1.file_uploader('Upload fluid file:', type=('xlsx', 'json'), on_change=clear_cache)

if uploaded_file is not None:
    if uploaded_file.type == 'application/json':
        fluid = uploaded_file.name.replace('.json', '')
        fluid = fluid.replace('lookup_', '')
        prop_json = uploaded_file
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        with st.spinner(text="In progress..."):
            fluid, prop_json = fc.xlsx_to_json(uploaded_file)
  
    # properties from json
    with st.spinner(text="Almost done..."):
        coord_label, coord_unit, crange, prop_label, prop_unit, prop_table = fc.get_data_from_json(prop_json)

    # some useful arrays
    const_coord = np.array([f'constant {coord_label[0]}', f'constant {coord_label[1]}', f'constant {coord_label[2]}'])
    coord = [f"{label} [{coord_unit[i]}]" for i, label in enumerate(coord_label)]    
    prop = [f"{label} [{prop_unit[i]}]" for i, label in enumerate(prop_label)]        
    variables = np.array(coord + prop)  #coordinate and property labels + units
    coord = np.array(coord)             #coordinate labels + units
    prop = np.array(prop)               #property labels + units
    const_coord_dict = {coord[0]: (const_coord[1], const_coord[2]), 
                        coord[1]: (const_coord[0], const_coord[2]), 
                        coord[2]: (const_coord[0], const_coord[1])}

    # define a new property by performing elementary operations on the base properties
    # update: we can now calculate the gradient, e.g., D(Ceq, 1) is the derivative of Ceq wrt Pressure
    new_prop = tab1.text_input("Define a new property:", placeholder="e.g. FRI_L1 / MW_L1")
    prop_label, prop_table, prop, variables = prop_definer(prop_label, prop_table, prop, variables, new_prop)

    # needed for interpolation
    prop_range = np.linspace(0, len(prop_label) - 1, len(prop_label))
    interp = ip.RegularGridInterpolator((prop_range, crange[0], crange[1], crange[2]), prop_table)

    # container for page
    maincont.header(fluid)
    with maincont:
        tab5, tab6 = st.tabs(("Plot", "Table"))

    # sidebar filters
    xaxis = tab2.selectbox('Select x-axis:', variables, on_change=reset_axes)
    yaxis = tab2.selectbox('Select y-axis:', variables, on_change=reset_axes) 

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

    st.sidebar.markdown("#")    # empty space at the bottom of sidebar
else:
    st.warning('Please upload a fluid file to begin')
