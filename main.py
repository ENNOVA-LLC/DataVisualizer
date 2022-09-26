import streamlit as st
import numpy as np

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

container = st.sidebar.container()
col1, col2 = st.sidebar.columns(2)

if 'count' not in st.session_state:
	st.session_state.count = 0

with col1:
    new_input = st.button("Add input box")

with col2:
    delete_input = st.button("Delete input box")

if new_input:
	st.session_state.count += 1
elif delete_input and st.session_state.count > 0:
    st.session_state.count -= 1

with container:
    x = [st.number_input('Enter value', min_value=1.0, max_value=5.0, key=i) for i in range(st.session_state.count)]
x = np.array(x)
# st.write(x)