# program to calculate rbf response surface given DoE points and responses
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

import numpy as np

from rbffunctions import *
# note can get rid of all warning flags by using import rbffunctions and
# making an explicit call to the function using 'rbffunctions.etc'

# reading data into a file
xydata = open('heatsink.txt','r').readlines()
x1 = []
x2 = []
thermal_resistance = []
pressure_drop = []

for line in xydata:
    x1.append(float(line.split()[0]))
    x2.append(float(line.split()[1]))
    thermal_resistance.append(float(line.split()[2]))
    pressure_drop.append(float(line.split()[3]))
    
print ('read finished')

x = (np.vstack([x1,x2])).transpose()

# Calculating number of DoE points for matrix x
n=int(x.shape[0])

# Calculating number of dvs
ndv=int(x.shape[1])
	
print ('n = {0:d} ndv = {1:d}'.format(n,ndv))

# scale x1 and x2 design variables to lie between 0 and 1
x1_scaled = []
x2_scaled = []

for i in range(0,n):
    x1_scaled.append((x1[i]-min(x1))/(max(x1)-min(x1)))
    x2_scaled.append((x2[i]-min(x2))/(max(x2)-min(x2)))

# vector of scaled DoE points
x_scaled = (np.vstack([x1_scaled,x2_scaled])).transpose()

# Now scale the objectives to lie between 0 and 1: objective 1 =
# scaled thermal resistance and objective 2 = scaled pressure drop
minobj1 = min(thermal_resistance)
denom1 = (max(thermal_resistance)-min(thermal_resistance))
minobj2 = min(pressure_drop)
denom2 = (max(pressure_drop)-min(pressure_drop))
obj1 = []
obj2 = []
for i in range(0,len(thermal_resistance)):
    obj1.append((thermal_resistance[i]-minobj1)/denom1)
    obj2.append((pressure_drop[i]-minobj2)/denom2)

obj1_full = []   # full data set of scaled thermal resistances
obj2_full = []   # full data set of scaled pressure drops
obj1_full = obj1
obj2_full = obj2

# (1) Calibration/single-objective optimisation for thermal resistance
# -----------------------------------------------------------------
# calculate rbf weights lambda based on training data only rbf settings
rbfmodel = 1     # Gaussian weights
nbetas = 101
beta_array = np.linspace(0.5,10,nbetas)
RMSE_array = np.linspace(0,1,nbetas)
beta_min = 0.0
RMSE_min = 1e8
ntrain = n-1
ntest = 1

# calculate RMSE for each beta in the specified range
for i in range(0,nbetas):
    
    beta = beta_array[i]
    # create leave one out data sets
    RMSE = 0.0
    for j in range(0,n):
        
        [obj1_train,x_train,obj1_test,x_test] = leaveoneout(j,x_scaled,obj1_full,n,ndv)
        lam = rbfweights(ntrain,x_train,ndv,obj1_train,rbfmodel,beta)

        # evaluate rbf approximation based on training data at the left out point
        obj1_rbf_test = rbf_testeval(ntrain,x_train,ndv,x_test,ntest,lam,rbfmodel,beta)

        # calculate the Root Mean Square Error between rbf approximations and actual data
        for k in range(0,ntest):
            RMSE = RMSE + (obj1_rbf_test[k]-obj1_test[k])**2

    RMSE = np.sqrt(RMSE/n)
    RMSE_array[i] = RMSE
        
    if (RMSE < RMSE_min):
        beta_min = beta
        RMSE_min = RMSE
    
# find beta which minimises RMSE
# RMSE_min = np.min(RMSE_array)
print("Minimum RMSE = {0:6.3f} at beta = {1:6.3f}".format(RMSE_min[0,0],beta_min))
beta = beta_min
obj1_beta = beta
beta_tr = []
beta_tr.append(beta)

plot1 = plt.figure(1)
plt.xlabel('beta')
plt.ylabel('RMSE')
plt.title('Plot out RMSE vs Beta for Leave One Out Cross Validation of Objective 1')
plt.plot(beta_array,RMSE_array)
plt.show()

# Find the uncontrained global minimum using scipy optimisation routines
# rbf_point is an amended version of the rbf function which calculates
# rbf value given a single design point
# use testing flag to check that rbf_point is working correctly
obj1_lam = rbfweights(n,x_scaled,ndv,obj1_full,rbfmodel,obj1_beta)

testing = 1
if (testing):
    xp = np.zeros((1,ndv))
    xp[0][0] = 0.5
    xp[0][1] = 0.5
    # evaluate rbf approximation using all data points
    yp1 = rbf(n,x_scaled,ndv,xp,1,obj1_lam,rbfmodel,obj1_beta)
    xp = np.zeros(ndv)
    xp[0] = 0.5
    xp[1] = 0.5
    yp2=rbf_point(n,x_scaled,ndv,xp,obj1_lam,rbfmodel,obj1_beta)

from scipy import optimize

def obj1_f(x): # function to be optimised
    return rbf_point(n,x_scaled,ndv,x,obj1_lam,rbfmodel,obj1_beta)

bnds = ((0,1),(0,1))     # set optimisation bounds
optmethod = "Nelder-Mead"   # set optimisation method

# optimise with scipy functions
if (optmethod=="Nelder-Mead"):
    res = optimize.minimize(obj1_f, [0.5,0.5], bounds=bnds, method="Nelder-Mead")
elif (optmethod=="Powell"):
        res = optimize.minimize(obj1_f, [0.5,0.5], bounds=bnds, method="Powell")

print("res success {0:5d}".format(res.success))
print("optimal scaled points {0:10.5e} {1:10.5e}".format(res.x[0], res.x[1]))
print("optimal value {0:10.5e}".format(res.fun))
xopt = res.x[0]
yopt = res.x[1]
optval = res.fun
xopt = []
yopt = []
optval = []
xopt.append(res.x[0])
yopt.append(res.x[1])
optval.append(res.fun)

# create output points for displaying the rbf surface
na = 41
xa = np.zeros((na*na,ndv))
ya_exact = np.zeros((na*na))

deltax = 1.0/(na-1)
deltay = deltax
countpos = 0
for i in range(0,na):
    xval = i*deltax
    for j in range(0,na):
        yval = j*deltay
        xa[countpos][0] = xval
        xa[countpos][1] = yval
        countpos = countpos + 1

# evaluate rbf approximation using all data points
ya = rbf(n,x_scaled,ndv,xa,na,obj1_lam,rbfmodel,obj1_beta)
np.savetxt('obj1_rbfvalues.txt',ya,fmt='%.7e')

# now plot out results
# create equivalent 2d arrays from the response data
ipos = -1
Xr = np.zeros((41,41))
Yr = np.zeros((41,41))
Xr_scaled = np.zeros((41,41))
Yr_scaled = np.zeros((41,41))
Zr_rbf = np.zeros((41,41))

# RBF surface array
for i in range(0,41):
    for j in range(0,41):
        ipos = ipos + 1
        Xr_scaled[i][j]=xa[ipos][0]
        Xr[i][j]=Xr_scaled[i][j]        
        Yr_scaled[i][j]=xa[ipos][1]
        Yr[i][j]=Yr_scaled[i][j]                
        Zr_rbf[i][j]=ya[ipos]

# Plot out RBF approximation
fig = plt.figure(2)
fig.suptitle('RBF approximation of objective 1 with beta='+str(obj1_beta)+' and n='+str(n),fontsize=10)
#ax = fig.gca(projection='3d')
ax=plt.axes(projection='3d')
surf = ax.plot_surface(Xr, Yr, Zr_rbf, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim(-0.1, 1.1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)

# plot out the scatter points
ax.scatter(x_scaled[:,0],x_scaled[:,1],obj1, c='r', marker='o',s=4)
ax.scatter(xopt,yopt,optval, c='k', marker='o',s=16)   # plot optimum point
ax.set_ylabel('x2')
ax.set_zlabel('Thermal Resistance')
fig.savefig('tf_rbfheatsink.jpg')

# save data for objective 1
np.savetxt("data/Xr_tr.txt",Xr)
np.savetxt("data/Yr_tr.txt",Yr)
np.savetxt("data/Zr_tr.txt",Zr_rbf)
np.savetxt("data/x_scaled_tr.txt",x_scaled)
np.savetxt("data/Zr_tr.txt",Zr_rbf)
np.savetxt("data/lam_tr.txt",obj1_lam)
np.savetxt("data/xopt_tr.txt",xopt)
np.savetxt("data/yopt_tr.txt",yopt)
np.savetxt("data/optval_tr.txt",optval)
np.savetxt("data/beta_tr.txt",beta_tr)
np.savetxt("data/obj_tr.txt",obj1)

# (2) Calibration/single-objective optimisation for second objective
# -----------------------------------------------------------------
# calculate rbf weights lambda based on training data only rbf settings
rbfmodel = 1     # Gaussian weights
nbetas = 101
beta_array = np.linspace(0.5,10,nbetas)
RMSE_array = np.linspace(0,1,nbetas)
beta_min = 0.0
RMSE_min = 1e8
ntrain = n-1
ntest = 1

# calculate RMSE for each beta in the specified range
for i in range(0,nbetas):
    
    beta = beta_array[i]
    # create leave one out data sets
    RMSE = 0.0
    for j in range(0,n):
        
        [obj2_train,x_train,obj2_test,x_test] = leaveoneout(j,x_scaled,obj2_full,n,ndv)
        lam = rbfweights(ntrain,x_train,ndv,obj2_train,rbfmodel,beta)

        # evaluate rbf approximation based on training data at the left out point
        obj2_rbf_test = rbf_testeval(ntrain,x_train,ndv,x_test,ntest,lam,rbfmodel,beta)

        # calculate the Root Mean Square Error between rbf approximations and actual data
        for k in range(0,ntest):
            RMSE = RMSE + (obj2_rbf_test[k]-obj2_test[k])**2

    RMSE = np.sqrt(RMSE/n)
    RMSE_array[i] = RMSE
        
    if (RMSE < RMSE_min):
        beta_min = beta
        RMSE_min = RMSE
    
# find beta which minimises RMSE
# RMSE_min = np.min(RMSE_array)
print("Minimum RMSE = {0:6.3f} at beta = {1:6.3f}".format(RMSE_min[0,0],beta_min))
beta = beta_min
obj2_beta = beta
beta_pd = []
beta_pd.append(beta)
plot3 = plt.figure(3)
plt.ion()
plt.xlabel('beta')
plt.ylabel('RMSE')
plt.title('Plot out RMSE vs Beta for objective 2: Leave One Out Cross Validation')
plt.plot(beta_array,RMSE_array)
plt.show()

# Find the uncontrained global minimum using scipy optimisation routines
# rbf_point is an amended version of the rbf function which calculates
# rbf value given a single design point
# use testing flag to check that rbf_point is working correctly
obj2_lam = rbfweights(n,x_scaled,ndv,obj2_full,rbfmodel,obj2_beta)

testing = 1
if (testing):
    xp = np.zeros((1,ndv))
    xp[0][0] = 0.5
    xp[0][1] = 0.5
    # evaluate rbf approximation using all data points
    yp1 = rbf(n,x_scaled,ndv,xp,1,obj2_lam,rbfmodel,obj2_beta)
    xp = np.zeros(ndv)
    xp[0] = 0.5
    xp[1] = 0.5
    yp2=rbf_point(n,x_scaled,ndv,xp,obj2_lam,rbfmodel,obj2_beta)

def obj2_f(x): # function to be optimised
    return rbf_point(n,x_scaled,ndv,x,obj2_lam,rbfmodel,obj2_beta)

# use the functions defined above with the new 'lam' array in rbfpoint
bnds = ((0,1),(0,1))     # set optimisation bounds
optmethod = "Nelder-Mead"   # set optimisation method

# optimise with scipy functions
if (optmethod=="Nelder-Mead"):
    res = optimize.minimize(obj2_f, [0.5,0.5], bounds=bnds, method="Nelder-Mead")
elif (optmethod=="Powell"):
        res = optimize.minimize(obj2_f, [0.5,0.5], bounds=bnds, method="Powell")

print("res success {0:5d}".format(res.success))
print("optimal scaled points {0:10.5e} {1:10.5e}".format(res.x[0], res.x[1]))
print("optimal value {0:10.5e}".format(res.fun))
xopt = res.x[0]
yopt = res.x[1]
optval = res.fun
xopt = []
yopt = []
optval = []
xopt.append(res.x[0])
yopt.append(res.x[1])
optval.append(res.fun)

# create output points for displaying the rbf surface
na = 41
xa = np.zeros((na*na,ndv))

deltax = 1.0/(na-1)
deltay = deltax
countpos = 0
for i in range(0,na):
    xval = i*deltax
    for j in range(0,na):
        yval = j*deltay
        xa[countpos][0] = xval
        xa[countpos][1] = yval
        countpos = countpos + 1

# evaluate rbf approximation using all data points
ya = rbf(n,x_scaled,ndv,xa,na,obj2_lam,rbfmodel,obj2_beta)
np.savetxt('obj2_rbfvalues.txt',ya,fmt='%.7e')

# now plot out results
# create equivalent 2d arrays from the response data
ipos = -1
Xr = np.zeros((41,41))
Yr = np.zeros((41,41))
Xr_scaled = np.zeros((41,41))
Yr_scaled = np.zeros((41,41))
Zr_rbf = np.zeros((41,41))

# RBF surface array
for i in range(0,41):
    for j in range(0,41):
        ipos = ipos + 1
        Xr_scaled[i][j]=xa[ipos][0]
        Xr[i][j]=Xr_scaled[i][j]        
        Yr_scaled[i][j]=xa[ipos][1]
        Yr[i][j]=Yr_scaled[i][j]                
        Zr_rbf[i][j]=ya[ipos]

# Plot out RBF approximation
fig = plt.figure(4)
fig.suptitle('RBF approximation of objective 2 with beta='+str(obj2_beta)+' and n='+str(n),fontsize=10)
#ax = fig.gca(projection='3d')
ax2=plt.axes(projection='3d')
surf2 = ax2.plot_surface(Xr, Yr, Zr_rbf, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax2.set_zlim(-0.1, 1.1)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf2, shrink=0.5, aspect=5)

# plot out the scatter points
ax2.scatter(x_scaled[:,0],x_scaled[:,1],obj2, c='r', marker='o',s=4)
ax2.scatter(xopt,yopt,optval, c='k', marker='o',s=16)   # plot optimum point
ax2.set_ylabel('x2')
ax2.set_zlabel('Objective 2')
fig.savefig('obj2_rbfheatsink.jpg')

# save data for objective 1
np.savetxt("data/Xr_pd.txt",Xr)
np.savetxt("data/Yr_pd.txt",Yr)
np.savetxt("data/Zr_pd.txt",Zr_rbf)
np.savetxt("data/x_scaled_pd.txt",x_scaled)
np.savetxt("data/Zr_pd.txt",Zr_rbf)
np.savetxt("data/lam_pd.txt",obj2_lam)
np.savetxt("data/xopt_pd.txt",xopt)
np.savetxt("data/yopt_pd.txt",yopt)
np.savetxt("data/optval_pd.txt",optval)
np.savetxt("data/beta_pd.txt",beta_pd)
np.savetxt("data/obj_pd.txt",obj2)

# Now create Pareto front for the multi-objective problem minimising 
# objectives 1 and 2: thermal resistance and pressure drop

# create the weighted objective function
def weighted_f2(x): # function to be optimised
    return (omega*obj1_f(x)+(1-omega)*obj2_f(x))

# create the pareto set 
optmethod="Nelder-Mead"
bnds = ((0,1),(0,1))     # set optimisation bounds
x0 = [0.5,0.5]    # initial point for optimisation search
xpareto = []
obj1pareto = []
obj2pareto = []

OmegaSet = np.linspace(0,1,100)
for omega in OmegaSet:
    # find minimal solution for current omega
    if (optmethod=="Nelder-Mead"):
        res = optimize.minimize(weighted_f2, x0, bounds=bnds, method="Nelder-Mead")
        x0 = res.x    # update initial position of search
    elif (optmethod=="Powell"):
        res = optimize.minimize(weighted_f2, x0, bounds=bnds, method="Powell")
        x0 = res.x    # update initial position of search
    print("res success {0:5d} for omega= {1:4.2e}".format(res.success,omega))
    print("optimal scaled points {0:10.5e} {1:10.5e}".format(res.x[0], res.x[1]))
    print("optimal value {0:10.5e}".format(res.fun))
    xpareto.append(res.x)
    obj1pareto.append(obj1_f(res.x))
    obj2pareto.append(obj2_f(res.x))
    
# Plot out Pareto Front
fig = plt.figure(5)
#fig.suptitle('Pareto front of scaled objectives - obj1 vs obj2',fontsize=10)
plt.ion()
plt.xlabel('obj1')
plt.ylabel('obj2')
plt.title('Pareto front for scaled objectives- obj1 vs obj2',fontsize=10)
plt.plot(obj1pareto,obj2pareto)
plt.show()

# create physical values of objectives
trpareto = []; trpareto = obj1pareto
mintf= min(thermal_resistance); maxtf = max(thermal_resistance)
pdpareto = []; pdpareto = obj2pareto
minpd= min(pressure_drop); maxpd = max(pressure_drop)
for i in range(0,len(trpareto)):
    trpareto[i] = mintf + (maxtf-mintf)*obj1pareto[i]
    pdpareto[i] = minpd + (maxpd-minpd)*obj2pareto[i]

fig = plt.figure(6)
plt.ion()
plt.xlabel('thermal resistance')
plt.ylabel('pressure drop, Pa')
plt.title('Pareto front for thermal resistance vs pressure drop',fontsize=10)
plt.plot(trpareto,pdpareto)
plt.show()

# save pareto data points
np.savetxt("data/trpareto.txt",trpareto)
np.savetxt("data/pdpareto.txt",pdpareto)