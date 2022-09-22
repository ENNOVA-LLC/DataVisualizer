import streamlit as st
from config import DATA_DIR

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
