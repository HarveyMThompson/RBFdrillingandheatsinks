# program to calculate rbf response surface given DoE points and 
# calculating the single and multi-objective optimisation from these responses
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

import numpy as np

from rbffunctions import *

# reading data into a file
xydata = open('drillDoE.txt','r').readlines()
x1 = []
x2 = []
pressure_drop = []

for line in xydata:
    x1.append(float(line.split()[0]))
    x2.append(float(line.split()[1]))
    pressure_drop.append(float(line.split()[2]))

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
 
pd_full = []   # full data set of pressure drops
# calculate value of Pressure Drop from DoE points
for i in range(0,n):
    pd_full.append(pressure_drop[i])

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
        
        [pd_train,x_train,pd_test,x_test] = leaveoneout(j,x_scaled,pd_full,n,ndv)
        lam = rbfweights(ntrain,x_train,ndv,pd_train,rbfmodel,beta)

        # evaluate rbf approximation based on training data at the left out point
        pd_rbf_test = rbf_testeval(ntrain,x_train,ndv,x_test,ntest,lam,rbfmodel,beta)

        # calculate the Root Mean Square Error between rbf approximations and actual data
        for k in range(0,ntest):
            RMSE = RMSE + (pd_rbf_test[k]-pd_test[k])**2

    RMSE = np.sqrt(RMSE/n)
    RMSE_array[i] = RMSE
        
    if (RMSE < RMSE_min):
        beta_min = beta
        RMSE_min = RMSE
    
# find beta which minimises RMSE
# RMSE_min = np.min(RMSE_array)
print("Minimum RMSE = {0:6.3f} at beta = {1:6.3f}".format(RMSE_min[0,0],beta_min))
beta = beta_min
plt.ion()
plt.xlabel('beta')
plt.ylabel('RMSE')
plt.title('Plot out RMSE vs Beta for Leave One Out Cross Validation')
plt.plot(beta_array,RMSE_array)
plt.show()
betaval = []
betaval.append(beta_min)

# Find the uncontrained global minimum using scipy optimisation routines
# rbf_point is an amended version of the rbf function which calculates
# rbf value given a single design point
# use testing flag to check that rbf_point is working correctly
lam = rbfweights(n,x_scaled,ndv,pd_full,rbfmodel,beta)

testing = 0
if (testing):
    xp = np.zeros((m,ndv))
    xp[0][0] = 0.5
    xp[0][1] = 0.5
    # evaluate rbf approximation using all data points
    yp1 = rbf(n,x_scaled,ndv,xp,m,lam,rbfmodel,beta)
    xp = np.zeros(ndv)
    xp[0] = 0.5
    xp[1] = 0.5
    yp2=rbf_point(n,x_scaled,ndv,xp,lam,rbfmodel,beta)

from scipy import optimize

def f(x): # function to be optimised
    return rbf_point(n,x_scaled,ndv,x,lam,rbfmodel,beta)

bnds = ((0,1),(0,1))     # set optimisation bounds
optmethod = "Nelder-Mead"   # set optimisation method

# optimise with scipy functions
if (optmethod=="Nelder-Mead"):
    res = optimize.minimize(f, [0.5,0.5], bounds=bnds, method="Nelder-Mead")
elif (optmethod=="Powell"):
        res = optimize.minimize(f, [0.5,0.5], bounds=bnds, method="Powell")

print("res success {0:5d}".format(res.success))
print("optimal scaled points {0:10.5e} {1:10.5e}".format(res.x[0], res.x[1]))
print("optimal value {0:10.5e}".format(res.fun))
xopt = min(x1)+res.x[0]*(max(x1)-min(x1))
yopt = min(x2)+res.x[1]*(max(x2)-min(x2))
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
ya = rbf(n,x_scaled,ndv,xa,na,lam,rbfmodel,beta)
np.savetxt('rbfvalues.txt',ya,fmt='%.7e')

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
        Xr[i][j]=min(x1)+Xr_scaled[i][j]*(max(x1)-min(x1))        
        Yr_scaled[i][j]=xa[ipos][1]
        Yr[i][j]=min(x2)+Yr_scaled[i][j]*(max(x2)-min(x2))                
        Zr_rbf[i][j]=ya[ipos]

# Plot out RBF approximation
fig = plt.figure()
fig.suptitle('RBF approximation with beta='+str(beta)+' and n='+str(n),fontsize=10)
#ax = fig.gca(projection='3d')
ax=plt.axes(projection='3d')

surf = ax.plot_surface(Xr, Yr, Zr_rbf, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
ax.set_zlim(25, 35)
ax.set_xlim(0, max(x1))
ax.set_ylim(0, max(x2))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)

# plot out the scatter points
ax.scatter(x[:,0],x[:,1],pressure_drop, c='r', marker='o',s=4)
ax.scatter(xopt,yopt,optval, c='k', marker='o',s=16)   # plot optimum point
ax.set_xlabel('corner radius (mm)')
ax.set_ylabel('orientation angle (degrees)')
ax.set_zlabel('Pressure Drop')
fig.savefig('rbf.jpg')

# save data for use in streamlit application
np.savetxt("data/Xr.txt",Xr)
np.savetxt("data/Yr.txt",Yr)
np.savetxt("data/Zr_rbf.txt",Zr_rbf)
np.savetxt("data/x_scaled.txt",x_scaled)
np.savetxt("data/x.txt",x)
np.savetxt("data/pressure_drop.txt",pressure_drop)
np.savetxt("data/beta_min.txt",betaval)
np.savetxt("data/beta_array.txt",beta_array)
np.savetxt("data/RMSE_array.txt",RMSE_array)
np.savetxt("data/lam.txt",lam)
np.savetxt("data/xopt.txt",xopt)
np.savetxt("data/yopt.txt",yopt)
np.savetxt("data/optval.txt",optval)

