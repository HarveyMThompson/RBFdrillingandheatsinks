# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:00:05 2024

@author: harveythompson
"""

import streamlit as st

drilling_page = st.Page("st_plotly_rbf_drilling.py", title="Single objective optimisation: drilling")
heatsink_page = st.Page("st_plotly_rbf_heatsink.py",title="Double objective optimisation: heat sink")

pg = st.navigation([drilling_page, heatsink_page])
pg.run()
