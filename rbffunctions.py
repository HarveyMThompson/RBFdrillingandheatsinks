
# function to calculate weights for rbfs
def rbfweights(n,x,ndv,f,rbfmodel,beta):
    import numpy as np
    import sys    
    
    phi = np.zeros(shape=(n,n))
    # phi = np.matrix(phi1)
    
    # calculate the gram matrix
    for i in range(0,n):
        for j in range(0,n):
            r = 0.0
            for k in range (0,ndv):
                r = r + (x[i][k]-x[j][k])**2
            r = np.sqrt(r)
            if (rbfmodel == 1):
                phi[i][j] = np.exp(-beta*(r**2))
            else:
                if (rbfmodel == 2):
                    phi[i][j] = np.sqrt(beta**2 + r**2)
                else:
                    if (rbfmodel == 3):
                        phi[i][j] = 1/np.sqrt(beta**2 + r**2)
                    else:
                        print ("invalid rbfmodel value")
                        sys.exit(0)
                        
    
    # compute the weights. need to transform f into an array and transpose it
    farray = np.asarray(f)
    fm = np.asmatrix(farray)    
    fmt = fm.transpose()
    lam = np.linalg.solve(phi, fmt)
    return lam
    print ("Finished computing rbf weights in rbfweights")

# function to calculate rbf approximation at specified plan design points xa    
def rbf(n,x,ndv,xa,na,lam,rbfmodel,beta):
    
    import numpy as np
    import sys    
    
    phi = np.zeros(shape=(na*na,n))
    # phi = np.zeros(n,n,float)
    
    # calculate the gram matrix
    for i in range(0,na*na):
        for j in range(0,n):
            r = 0.0
            for k in range (0,ndv):
                r = r + (xa[i][k]-x[j][k])**2
            r = np.sqrt(r)
            if (rbfmodel == 1):
                phi[i][j] = np.exp(-beta*(r**2))
            else:
                if (rbfmodel == 2):
                    phi[i][j] = np.sqrt(beta**2 + r**2)
                else:
                    if (rbfmodel == 3):
                        phi[i][j] = 1/np.sqrt(beta**2 + r**2)
                    else:
                        print ("invalid rbfmodel value")
                        sys.exit(0)
                        
    
    # compute the rbf response
    # ya = phi*lam
    # phim = np.matrix(phi)
    # lamm = np.matrix(lam)
    ya = np.asmatrix(phi)*lam
    return ya
    
    print ("Finished computing rbf approximations")  

# define function
def yval(x,y):
    yval = x**2 + y**2
    return yval

# define six hump camel function
def sixhumpcamel(x1,x2):
    X1 = 4*x1-2;
    X2 = 2*x2-1;
    fval = (4 - 2.1*(X1**2) + (X1**4)/3)*X1**2 + X1*X2 + (4*(X2**2) - 4)*(X2**2)
    return fval

# generate test dataset from ntest random numbers between 1 and n
def generate_test(ntest,n):
    import random

    testindex = []
    testflag = []
    for i in range(n):
        testflag.append(0)
        
    num = 0
    for i in range(100):
        if (num < ntest):
            k = random.randint(0,n-1)
            if (testflag[k] == 0):
                testindex.append(k)
                testflag[k] = 1
                num = num + 1
        else:
            break
            
            
    return testindex, testflag

# function to calculate rbf approximation at specified ntest test points x_test   
def rbf_testeval(ntrain,x_train,ndv,x_test,ntest,lam,rbfmodel,beta):
    
    import numpy as np
    import sys    
    
    phi = np.zeros(shape=(ntest,ntrain))
    # phi = np.zeros(n,n,float)
    
    # calculate the gram matrix
    for i in range(0,ntest):
        for j in range(0,ntrain):
            r = 0.0
            for k in range (0,ndv):
                r = r + (x_test[i][k]-x_train[j][k])**2
            r = np.sqrt(r)
            if (rbfmodel == 1):
                phi[i][j] = np.exp(-beta*(r**2))
            else:
                if (rbfmodel == 2):
                    phi[i][j] = np.sqrt(beta**2 + r**2)
                else:
                    if (rbfmodel == 3):
                        phi[i][j] = 1/np.sqrt(beta**2 + r**2)
                    else:
                        print ("invalid rbfmodel value")
                        sys.exit(0)
                        
    
    y_test = np.asmatrix(phi)*lam
    return y_test
    
    print ("Finished computing rbf approximations at test data points")  

# function to calculate training and testing data for leave one out validation
def leaveoneout(i,x,f,n,ndv):
    
    import numpy as np
    
    # create leave one out training and test data sets with test data on index i
    f_train = []
    f_test = []
    x1_train = []
    x1_test = []
    x2_train = []
    x2_test = []
    # create leave one out training and test data sets
    for j in range(0,n):
        if (j < i):
            f_train.append(f[j])
            x1_train.append(x[j][0])
            x2_train.append(x[j][1])        
        else:
            if (j == i):
                f_test.append(f[i])
                x1_test.append(x[i][0])
                x2_test.append(x[i][1])
            else:
                f_train.append(f[j])
                x1_train.append(x[j][0])
                x2_train.append(x[j][1]) 
    
    # put training and test data into arrays
    x_test = (np.vstack([x1_test,x2_test])).transpose()
    x_train = (np.vstack([x1_train,x2_train])).transpose()

    return f_train, x_train, f_test, x_test

# function to calculate a rpf surrogate function value at a specific point xp
def rbf_point(n,x,ndv,xp,lam,rbfmodel,beta):
    
    import numpy as np
    import sys    
    
    phi = np.zeros(shape=(1,n))
    # phi = np.zeros(n,n,float)
    
    # calculate the gram matrix
    for j in range(0,n):
        r = 0.0
        for k in range (0,ndv):
            r = r + (xp[k]-x[j][k])**2
        r = np.sqrt(r)
        if (rbfmodel == 1):
            phi[0][j] = np.exp(-beta*(r**2))
        else:
            if (rbfmodel == 2):
                phi[0][j] = np.sqrt(beta**2 + r**2)
            else:
                if (rbfmodel == 3):
                    phi[0][j] = 1/np.sqrt(beta**2 + r**2)
                else:
                    print ("invalid rbfmodel value")
                    sys.exit(0)
                        
    
    # compute the rbf response
    # ya = phi*lam
    # phim = np.matrix(phi)
    # lamm = np.matrix(lam)
    yp = (np.asmatrix(phi)*lam).item()
    return yp
    
print ("Finished computing rbf approximations")  