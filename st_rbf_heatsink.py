# program to calculate rbf response surface given DoE points and responses
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import streamlit as st

import numpy as np

from rbffunctions import *
# note can get rid of all warning flags by using import rbffunctions and
# making an explicit call to the function using 'rbffunctions.etc'

rbfmodel = 1     # Gaussian weights - can automate this
n = 30    # number of DoE points - can automate this
ndv = 2    #
x1min_actual = 0.4
x1max_actual = 1.0
x2min_actual = 0.4
x2max_actual = 1.0

###################################
# reading data into relevant files
###################################
def read_data():
    
    # Objective 1 - thermal resistance
    st.session_state.Xr_tr = np.loadtxt("heatsinkdata/Xr_tr.txt")
    st.session_state.Yr_tr = np.loadtxt("heatsinkdata/Yr_tr.txt")
    st.session_state.Zr_tr = np.loadtxt("heatsinkdata/Zr_tr.txt")
    st.session_state.x_scaled_tr = np.loadtxt("heatsinkdata/x_scaled_tr.txt")
    lam1 = np.loadtxt("heatsinkdata/lam_tr.txt")
    st.session_state.lam_tr = np.transpose(np.asmatrix(lam1))    
    st.session_state.xopt_tr = np.loadtxt("heatsinkdata/xopt_tr.txt")
    st.session_state.yopt_tr = np.loadtxt("heatsinkdata/yopt_tr.txt")
    st.session_state.optval_tr = np.loadtxt("heatsinkdata/optval_tr.txt")
    st.session_state.beta_tr = np.loadtxt("heatsinkdata/beta_tr.txt")
    st.session_state.obj_tr = np.loadtxt("heatsinkdata/obj_tr.txt")

    # Objective 2 - pressure difference
    st.session_state.Xr_pd = np.loadtxt("heatsinkdata/Xr_pd.txt")
    st.session_state.Yr_pd = np.loadtxt("heatsinkdata/Yr_pd.txt")
    st.session_state.Zr_pd = np.loadtxt("heatsinkdata/Zr_pd.txt")
    st.session_state.x_scaled_pd = np.loadtxt("heatsinkdata/x_scaled_pd.txt")
    lam2 = np.loadtxt("heatsinkdata/lam_pd.txt")
    st.session_state.lam_pd = np.transpose(np.asmatrix(lam2))
    st.session_state.xopt_pd = np.loadtxt("heatsinkdata/xopt_pd.txt")
    st.session_state.yopt_pd = np.loadtxt("heatsinkdata/yopt_pd.txt")
    st.session_state.optval_pd = np.loadtxt("heatsinkdata/optval_pd.txt")
    st.session_state.beta_pd = np.loadtxt("heatsinkdata/beta_pd.txt")
    st.session_state.obj_pd = np.loadtxt("heatsinkdata/obj_pd.txt")

    # Pareto points
    st.session_state.trpareto = np.loadtxt("heatsinkdata/trpareto.txt")
    st.session_state.pdpareto = np.loadtxt("heatsinkdata/pdpareto.txt")

    # create the actual points
    x1min_actual = 0.4
    x1max_actual = 1.0
    x2min_actual = 0.4
    x2max_actual = 1.0
    x_actual = np.loadtxt("heatsinkdata/x_scaled_tr.txt")
    for i in range(0,n):
        x_actual[i][0] = x1min_actual + st.session_state.x_scaled_tr[i][0]*(x1max_actual-x1min_actual)
        x_actual[i][1] = x2min_actual + st.session_state.x_scaled_tr[i][1]*(x2max_actual-x2min_actual)
    st.session_state.x_actual = x_actual

    # scale the thermal resistance arrays
    st.session_state.min_tr = 0.1618792
    st.session_state.max_tr = 0.2472688
    min_tr = st.session_state.min_tr
    max_tr = st.session_state.max_tr
    st.session_state.Xr = x1min_actual + (x1max_actual-x1min_actual)*st.session_state.Xr_tr
    st.session_state.Yr = x2min_actual + (x2max_actual-x2min_actual)*st.session_state.Yr_tr
    st.session_state.Zr = min_tr + (max_tr-min_tr)*st.session_state.Zr_tr
    st.session_state.objactual_tr = min_tr + (max_tr-min_tr)*st.session_state.obj_tr
    st.session_state.xoptactual_tr = x1min_actual + (x1max_actual-x1min_actual)*st.session_state.xopt_tr
    st.session_state.yoptactual_tr = x1min_actual + (x1max_actual-x1min_actual)*st.session_state.yopt_tr
    st.session_state.optactual_tr = min_tr + (max_tr-min_tr)*st.session_state.optval_tr

    # scale the pressure drop arrays
    st.session_state.min_pd = 12553.08
    st.session_state.max_pd = 38679.57
    min_pd = st.session_state.min_pd
    max_pd = st.session_state.max_pd
    st.session_state.Zp = min_pd + (max_pd-min_pd)*st.session_state.Zr_pd
    st.session_state.objactual_pd = min_pd + (max_pd-min_pd)*st.session_state.obj_pd
    st.session_state.xoptactual_pd = x1min_actual + (x1max_actual-x1min_actual)*st.session_state.xopt_pd
    st.session_state.yoptactual_pd = x1min_actual + (x1max_actual-x1min_actual)*st.session_state.yopt_pd
    st.session_state.optactual_pd = min_pd + (max_pd-min_pd)*st.session_state.optval_pd
    
# calculate dimensionless value of thermal resistance at specified point
def tr_f(x):
    return rbf_point(n,st.session_state.x_scaled_tr,ndv,x,st.session_state.lam_tr,rbfmodel,st.session_state.beta_tr)

# calculate dimensionless value of pressure difference at specified point
def pd_f(x): 
    return rbf_point(n,st.session_state.x_scaled_pd,ndv,x,st.session_state.lam_pd,rbfmodel,st.session_state.beta_pd)

# plot thermal resistance RBF surrogate model
def tr_plot():
    
    beta_tr = st.session_state.beta_tr
    Xr = st.session_state.Xr
    Yr = st.session_state.Yr
    Zr = st.session_state.Zr
    x_actual = st.session_state.x_actual
    objactual_tr = st.session_state.objactual_tr
    xoptactual_tr = st.session_state.xoptactual_tr
    yoptactual_tr = st.session_state.yoptactual_tr
    optactual_tr = st.session_state.optactual_tr
    
    # Plot out RBF approximation
    plt.figure()
    strbeta = f'{beta_tr:.2f}'
    plt.suptitle('RBF approximation of thermal resistance with beta='+strbeta+' and n='+str(n),fontsize=10)
    #ax = fig.gca(projection='3d')
    ax=plt.axes(projection='3d')
    surf = ax.plot_surface(Xr, Yr, Zr, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    # ax.set_zlim(-0.1, 1.1)
    ax.set_xlim(x1min_actual, x1max_actual)
    ax.set_ylim(x2min_actual, x2max_actual)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    # plot out the scatter points
    ax.scatter(x_actual[:,0],x_actual[:,1],objactual_tr, c='r', marker='o',s=4)
    ax.scatter(xoptactual_tr,yoptactual_tr,optactual_tr, c='k', marker='o',s=16)   # plot optimum point
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Thermal Resistance')
    plt.savefig('tf_rbfheatsink.jpg')
    st.pyplot(plt)

# plot pressure drop RBF surrogate model
def pd_plot():
    
    beta_pd = st.session_state.beta_pd
    Xr = st.session_state.Xr
    Yr = st.session_state.Yr
    Zp = st.session_state.Zp
    x_actual = st.session_state.x_actual
    objactual_pd = st.session_state.objactual_pd
    xoptactual_pd = st.session_state.xoptactual_pd
    yoptactual_pd = st.session_state.yoptactual_pd
    optactual_pd = st.session_state.optactual_pd
    
    # Plot out RBF approximation
    plt.figure()
    plt.suptitle('RBF approximation of pressure drop with beta='+str(beta_pd)+' and n='+str(n),fontsize=10)
    #ax = fig.gca(projection='3d')
    ax=plt.axes(projection='3d')
    surf = ax.plot_surface(Xr, Yr, Zp, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    # ax.set_zlim(-0.1, 1.1)
    ax.set_xlim(x1min_actual, x1max_actual)
    ax.set_ylim(x2min_actual, x2max_actual)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    # plot out the scatter points
    ax.scatter(x_actual[:,0],x_actual[:,1],objactual_pd, c='r', marker='o',s=4)
    ax.scatter(xoptactual_pd,yoptactual_pd,optactual_pd, c='k', marker='o',s=16)   # plot optimum point
    ax.set_ylabel('x2')
    ax.set_xlabel('x1')
    ax.set_zlabel('Pressure Drop')
    plt.savefig('pd_rbfheatsink.jpg')
    st.pyplot(plt)

# Plot Pareto Front
def pareto_plot():

    trpareto = st.session_state.trpareto
    pdpareto = st.session_state.pdpareto
    
    # Plot out pareto set
    plt.figure()
    # plt.ion()
    plt.xlabel('thermal resistance')
    plt.ylabel('pressure drop, Pa')
    plt.title('Pareto front for thermal resistance vs pressure drop',fontsize=10)
    plt.plot(trpareto,pdpareto)
    # plt.show()
    plt.savefig('pareto_rbfheatsink.jpg')
    st.pyplot(plt)

# calculate surrogate models of thermal resistance and pressure drop
def calc_surrogates(x1,x2):

    xp = np.zeros(ndv)
    x1_scaled = (x1-x1min_actual)/(x1max_actual-x1min_actual)
    x2_scaled = (x2-x2min_actual)/(x2max_actual-x2min_actual)
    xp[0] = x1_scaled
    xp[1] = x2_scaled
    min_tr = st.session_state.min_tr
    max_tr = st.session_state.max_tr
    min_pd = st.session_state.min_pd
    max_pd = st.session_state.max_pd
    tr_val = min_tr + (max_tr - min_tr)*tr_f(xp)
    pd_val = min_pd + (max_pd - min_pd)*pd_f(xp)   
    return tr_val,pd_val


# Add heading and introductory text
# st.set_page_config(layout='wide')

st.title("Heat Sink Optimisation")
st.write("This application enables you to explore the thermal resistance and pressure drop of a heat sink")
st.markdown("---")

# read in data for calculations only at the beginning of session
if 'Xr_tr' not in st.session_state:
    read_data()

tab1, tab2, tab3, tab4 = st.tabs(["Surrogate model", "Thermal Resistance plot", "Pressure Drop plot", "Pareto Front"])
with tab1:
    
    # Create the input sliders
    row1 = st.columns([1,1])

    default_value = 0.6
    st.session_state.x1 = default_value
    x1 = row1[0].slider("Design variable x1",0.4,1.0,default_value)
    st.session_state.x1 = x1

    st.session_state.x2 = default_value
    x2 = row1[1].slider("Design variable x2",0.4,1.0,default_value)
    st.session_state.x2 = x2

    # Calculate surrogate models
    tr_val,pd_val = calc_surrogates(x1,x2)
    col1, col2 = st.columns(2)
    with col1:
        st.write(f'Thermal resistance: {tr_val:.3f}')
    with col2:
        st.write(f'Pressure drop: {pd_val:.1f} Pa')

with tab2:
    tr_plot()

with tab3:
    pd_plot()

with tab4:
    pareto_plot()