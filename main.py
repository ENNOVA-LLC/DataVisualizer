from distutils.command.upload import upload
import streamlit as st
import numpy as np
import pandas as pd
import json
from config import DATA_DIR
from fileConverter import xlsx_to_json, get_data_from_json

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to ENNOVA's app! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Engineered Innovation (ENNOVA) is an independent company that provides consulting, 
    laboratory, computer modeling and chemical services for energy, petrochemical and 
    environmental industries.
     
    **👈 Select a demo from the sidebar** to see some examples
    of what ENNOVA's app can do!
    ### Want to learn more?
    - Check out [ennova.us](https://ennova.us/)
    - To get in touch [click here](https://ennova.us/contact/)
    """
    )

# File uploader
uploaded_file = st.sidebar.file_uploader('Upload fluid file:', type = ('xlsx', 'json'))

if uploaded_file is not None:
    if uploaded_file.type == 'application/json':
        st.success("json file successfully uploaded")
        fluid = uploaded_file.name.replace('.json', '')
        prop_json = uploaded_file
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        st.success("xlsx file successfully uploaded")
        fluid = uploaded_file.name.replace('.xlsx', '')
        prop_json = xlsx_to_json(uploaded_file)
    
    # properties from json
    coord_label, coord_unit, coord_range, prop_label, prop_unit, Property = get_data_from_json(prop_json)
else:
    st.warning('Please upload a fluid file to begin')