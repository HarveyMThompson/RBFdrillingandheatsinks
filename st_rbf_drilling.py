# program to calculate rbf response surface given DoE points and 
# calculating the single and multi-objective optimisation from these responses
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from rbffunctions import *

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
    st.session_state.x = np.loadtxt("drillingdata/x.txt")
    st.session_state.pressure_drop = np.loadtxt("drillingdata/pressure_drop.txt")
    st.session_state.beta_min = np.loadtxt("drillingdata/beta_min.txt")
    st.session_state.beta_array = np.loadtxt("drillingdata/beta_array.txt")
    st.session_state.RMSE_array = np.loadtxt("drillingdata/RMSE_array.txt")
    lam1 =  np.loadtxt("drillingdata/lam.txt")
    st.session_state.lam = np.transpose(np.asmatrix(lam1))
    st.session_state.xopt = np.loadtxt("drillingdata/xopt.txt")
    st.session_state.yopt = np.loadtxt("drillingdata/yopt.txt")
    st.session_state.optval = np.loadtxt("drillingdata/optval.txt")

def beta_plot():
    RMSE_array = st.session_state.RMSE_array
    beta_array = st.session_state.beta_array
    plt.figure()
    plt.xlabel('beta')
    plt.ylabel('RMSE')
    plt.title('Plot out RMSE vs Beta for Leave One Out Cross Validation')
    plt.plot(beta_array,RMSE_array)
    st.pyplot(plt)

# plot out the RBF surrogate model of pressure drop
def pd_plot():
    
    # Plot out RBF approximation
    plt.figure()
    beta = st.session_state.beta_min
    Xr = st.session_state.Xr
    Yr = st.session_state.Yr
    Zr_rbf = st.session_state.Zr_rbf
    x = st.session_state.x
    pressure_drop = st.session_state.pressure_drop
    xopt = st.session_state.xopt
    yopt = st.session_state.yopt
    optval = st.session_state.optval
    plt.suptitle('RBF approximation with beta='+str(beta)+' and n='+str(n),fontsize=10)
    #ax = fig.gca(projection='3d')
    ax=plt.axes(projection='3d')
    surf = ax.plot_surface(Xr, Yr, Zr_rbf, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
    ax.set_zlim(25, 35)
    ax.set_xlim(0, x1max_actual)
    ax.set_ylim(0, x2max_actual)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    # plot out the scatter points
    ax.scatter(x[:,0],x[:,1],pressure_drop, c='r', marker='o',s=4)
    ax.scatter(xopt,yopt,optval, c='k', marker='o',s=16)   # plot optimum point
    ax.set_xlabel('corner radius (mm)')
    ax.set_ylabel('orientation angle (degrees)')
    ax.set_zlabel('Pressure Drop')
    plt.savefig('rbf.jpg')
    st.pyplot(plt)

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

