# program to calculate rbf response surface given DoE points and 
# calculating the single and multi-objective optimisation from these responses
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from rbffunctions import *

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

rbfmodel = 1     # Gaussian weights
n = 20
ndv = 2
x1min_actual = 0.05
x1max_actual = 0.5
x2min_actual = 0.0
x2max_actual = 90.0
max_pd = 33.56574
min_pd = 26.7848
nbetas = 101

##########################################################
# reading data into relevant files
##########################################################
def read_data():
    
    st.session_state.Xr = np.loadtxt("drillingdata/Xr.txt")
    st.session_state.Yr = np.loadtxt("drillingdata/Yr.txt")
    st.session_state.Zr_rbf = np.loadtxt("drillingdata/Zr_rbf.txt")
    st.session_state.x_scaled = np.loadtxt("drillingdata/x_scaled.txt")
    st.session_state.x_actual = np.loadtxt("drillingdata/x.txt")
    st.session_state.pressure_drop = np.loadtxt("drillingdata/pressure_drop.txt")
    st.session_state.beta_min = np.loadtxt("drillingdata/beta_min.txt")
    st.session_state.beta_array = np.loadtxt("drillingdata/beta_array.txt")
    st.session_state.RMSE_array = np.loadtxt("drillingdata/RMSE_array.txt")
    lam1 =  np.loadtxt("drillingdata/lam.txt")
    st.session_state.lam = np.transpose(np.asmatrix(lam1))
    st.session_state.xopt = np.loadtxt("drillingdata/xopt.txt")
    st.session_state.yopt = np.loadtxt("drillingdata/yopt.txt")
    st.session_state.optval = np.loadtxt("drillingdata/optval.txt")
    
    st.session_state.x = st.session_state.x_actual[:,0]
    st.session_state.y = st.session_state.x_actual[:,1]
    st.session_state.z = st.session_state.pressure_drop
    dfscatter = pd.DataFrame(
        {'x': st.session_state.x,
         'y': st.session_state.y,
         'z': st.session_state.z,
    })
    st.session_state.dfscatter = dfscatter
    st.session_state.title_text = 'RBF approximation with beta='+str(st.session_state.beta_min)+' and n='+str(n)
    
    # process the data into a form that can be used with the go.Mesh3d function
    xmesh =[]
    ymesh = []
    zmesh = []
    for i in range(41):
        for j in range(41):
            xmesh.append(st.session_state.Xr[i,j])
            ymesh.append(st.session_state.Yr[i,j])
            zmesh.append(st.session_state.Zr_rbf[i,j])

    st.session_state.xmesh = xmesh
    st.session_state.ymesh = ymesh
    st.session_state.zmesh = zmesh
    
    st.session_state.df = pd.DataFrame(
        {'beta': st.session_state.beta_array,
         'RMSE': st.session_state.RMSE_array,
        })

# end read_data

def beta_plot():

    figbeta = px.line(st.session_state.df, x='beta', y='RMSE', markers=True, title='Plot of RMSE vs Beta for Leave One Out Cross Validation')
    st.plotly_chart(figbeta)

# plot out the RBF surrogate model of pressure drop
def pd_plot():
 
    x = st.session_state.xmesh
    y = st.session_state.ymesh
    z = st.session_state.zmesh
    trace2 = go.Mesh3d(x=x,
                       y=y,
                       z=z,
                       opacity=0.5,
                       color='rgba(244,22,100,0.6)')

    xscatter = st.session_state.x_actual[:,0]
    yscatter = st.session_state.x_actual[:,1]
    zscatter = st.session_state.pressure_drop
    trace3 = go.Scatter3d(x=xscatter, y=yscatter, z=zscatter, mode='markers')
    data2 = [trace2, trace3]
    layout = go.Layout(title=st.session_state.title_text,
                       title_font=dict(size=20,
                                       color='blue',
                                       family='Arial'),
                       title_x=0.25,
                       title_y=0.85)

    fig2 = go.Figure(data=data2, layout=layout)
    fig2.update_scenes(xaxis=dict(title="corner radius (mm)",nticks=6, range=[0.0,0.5]),
                       yaxis=dict(title="angle (degrees)",nticks=10, range=[0.0,90.0]), 
                       zaxis=dict(title="Pressure Drop (Pa)"))

    st.plotly_chart(fig2)    

# end pd_plot    
 
# calculate pressure drop at specified point
def pd_f(x): 
    return rbf_point(n,st.session_state.x_scaled,ndv,x,st.session_state.lam,rbfmodel,st.session_state.beta_min)

# calculate surrogate models of thermal resistance and pressure drop
def calc_surrogates(x1,x2):

    xp = np.zeros(ndv)
    x1_scaled = (x1-x1min_actual)/(x1max_actual-x1min_actual)
    x2_scaled = (x2-x2min_actual)/(x2max_actual-x2min_actual)
    xp[0] = x1_scaled
    xp[1] = x2_scaled
    pd_val = pd_f(xp)   
    return pd_val


st.title("Drilling Optimisation")
st.write("This application enables you to explore the pressure drop of a twist drill")
st.markdown("---")

checking_password = 0
if (checking_password != 0):
    if 'pwdcheck' not in st.session_state:
        st.session_state['pwdcheck'] = 0
        password_guess = st.text_input('What is the password?')
        if password_guess != st.secrets["password"]:
            st.stop()

# read in data for calculations only at the beginning of session
if 'Xr' not in st.session_state:
    read_data()

tab1, tab2, tab3 = st.tabs(["Surrogate model", "Pressure drop plot", "Leave one out cross validation"])
with tab1:
    
    # Create the input sliders
    row1 = st.columns([1,1])

    default_value = 0.2
    x1 = row1[0].slider("Corner radius (mm)",x1min_actual,x1max_actual,0.2)
    x2 = row1[1].slider("Orientation angle (degrees)",x2min_actual,x2max_actual,45.0)
    pd_val = calc_surrogates(x1,x2)
    st.write(f'Pressure drop: {pd_val:.1f} Pa')

with tab2:
    pd_plot()

with tab3:
    beta_plot()

