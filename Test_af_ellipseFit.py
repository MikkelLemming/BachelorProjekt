import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy import optimize as opt
import math as m
from numpy.linalg import eig, inv, svd
from mpl_toolkits.mplot3d import Axes3D
import random as r

data_S62 = np.genfromtxt("PosData_S62.txt", unpack=True, skip_header=1)
data_S62 = [data_S62[0], data_S62[1]*10**-3, data_S62[3]*10**-3, data_S62[2]*10**-3, data_S62[4]*10**-3]
x_S62 = data_S62[1]
y_S62 = data_S62[2]

data_S2 = np.genfromtxt("PosData_S2.txt", unpack=True, skip_header=1)
x_S2 = data_S2[1]
y_S2 = data_S2[2]

data_S38 = np.genfromtxt("PosData_S38.txt", unpack=True, skip_header=1)
x_S38 = data_S38[1]
y_S38 = data_S38[2]

data_S55 = np.genfromtxt("PosData_S55.txt", unpack=True, skip_header=1)
x_S55 = data_S55[1]
y_S55 = data_S55[2]


data_S62_new = np.genfromtxt("PosData_S62_new.txt", unpack=True, skip_header=1)
data_S62_new = [data_S62_new[0], data_S62_new[1]*10**-3, data_S62_new[3]*10**-3, data_S62_new[2]*10**-3, data_S62_new[4]*10**-3]
x_S62_new = data_S62_new[1]
y_S62_new = data_S62_new[2]





def r_theta_data(data, x_F = 0, y_F = 0):
    r = []
    theta = []
    
    for i in range(len(data[1])):
        r.append(np.sqrt((data[1][i] - x_F)**2+(data[2][i] - y_F)**2))
        if data[1][i] - x_F == 0:
            theta.append(1/2*np.pi)
        elif data[1][i] - x_F > 0:
            theta.append(np.arctan((data[2][i] - y_F)/(data[1][i] - x_F)))
        elif data[1][i] - x_F < 0:
            theta.append(np.pi + np.arctan((data[2][i] - y_F)/(data[1][i] - x_F)))
        else: 
            theta.append('NA')
            
    return r, theta








def my_f_ellipse(a, e, phi, xf, yf, theta):
    r = a*(1-e**2)/(1-e*np.cos(theta - phi))
    x = r*np.cos(theta) + xf
    y = r*np.sin(theta) + yf
    return r, [x,y]

def residual2(a, e, phi, xf, yf, theta, r):
    return (r - my_f_ellipse(a,e,phi,xf,yf,theta)[0])**2
    

def my_ultimat_fit(data):
    
    def sqr_sum_f_ellipse(parameters):
        a, e, phi, xf, yf = parameters
        r, theta = r_theta_data(data, x_F = xf, y_F=yf)
        
        residuals = []
        
        for i in range(len(r)):
            R = residual2(a,e,phi,xf,yf,theta[i],r[i])
#            residuals.append(R/(data[3][i]+data[4][i]))
            residuals.append(R)
 
            
        return sum(residuals)
    
    r, theta = r_theta_data(data)
    furthest_point = r.index(max(r))
    fp = furthest_point
        
    phi_guess = theta[fp]
    a_max = max(r)
    a_min = 0.5*max(r)
        
    center_area_x = (min(0,min(data[1]))*0.5, max(0,max(data[1]))*0.5)
    center_area_y = (min(0,min(data[2]))*0.5, max(0,max(data[2]))*0.5)

    guess_init = [0.1, 0.8, 0.5*np.pi, 0, 0]
#    guess_init = [a_min+0.01, 0.95, phi_guess, 0, 0]
#    guess_init = [0.1, 0.8, 0.5*np.pi, 0, 0]
#    guess_init = [0.125, 0.88, 0.5*np.pi, -0.0151, 0.0686]
#    guess_init = ellipse_fit_2(data)
    
    bnds = ((0,0.5), (0.5,0.99), (-np.pi,np.pi), (-0.4,0.2), (-0.2,0.2))
#    bnds = ((a_min,a_max), (0.7,0.99), (phi_guess-0.125*np.pi,phi_guess+0.125*np.pi), center_area_x, center_area_y)
#    bnds = ((a_min+0.01,a_min+0.02), (0.95,0.99), (phi_guess-0.00001, phi_guess+0.00001), (-0.01,0.01), (-0.01,0.01))
    
    result = opt.minimize(sqr_sum_f_ellipse, guess_init, method='L-BFGS-B', bounds=bnds)    
    
    X1 = []
    Y1 = []
        
    a, e, phi, xf, yf = result.x
        
    for i in np.linspace(0, 2*np.pi, 720):
        theta = i 
        r = my_f_ellipse(a,e,phi,xf,yf,theta)[0]
        x = r*np.cos(i) + xf
        y = r*np.sin(i) + yf
            
        X1.append(x)
        Y1.append(y)
    
    plt.figure()
    plt.plot(xf, yf, 'x')
    plt.plot(xf+2*a*e*np.cos(phi), yf+2*a*e*np.sin(phi), 'x')
    plt.plot(X1, Y1, color='orange')
    plt.plot(data[1], data[2], '*')
    plt.plot(0,0,'o', color='black')
    
    return result.x, X1, Y1


#dimmer, x,y = my_ultimat_fit(data_S2)


#print(dimmer)
#my_ultimat_fit(data_S2)
#my_ultimat_fit(data_S38)
#my_ultimat_fit(data_S55)
#my_ultimat_fit(data_S62)
#print(my_ultimat_fit(data_S55))
#plt.xlim([-0.4, 0.2])
#plt.ylim([-0.2,0.2])

#print(my_ultimat_fit(data_S2))
#plt.xlim([-0.4, 0.2])
#plt.ylim([-0.2,0.2])

#plt.xlim([0.2, -0.4])
#plt.ylim([-0.2,0.2])




def improve_data(data, error_limit=0, multiplyer=0.8):
    if error_limit==0:
        error_limit = (max(data[3])+max(data[4]))*multiplyer
        
    index_list = []
    for i in range(len(data[3])):
        if (data[3][i] + data[4][i]) < error_limit:
            index_list.append(i)
    
    a = []
    b = []
    c = []
    d = []
    e = []

    for i in index_list:
        a.append(data[0][i])
        b.append(data[1][i])
        c.append(data[2][i])
        d.append(data[3][i])
        e.append(data[4][i])
    
    data_improved = [a,b,c,d,e]
    return data_improved
    

#print(my_ultimat_fit(improve_data(data_S2)))
#print(my_ultimat_fit(data_S2))
#plt.xlim([0.125, -0.3])
#plt.ylim([-0.2,0.2])
#plt.title("S2")

#print(my_ultimat_fit(improve_data(data_S38)))
#print(my_ultimat_fit(data_S38))
#plt.xlim([0.125, -0.3])
#plt.ylim([-0.2,0.2])
#plt.title("S38")

#print(my_ultimat_fit(improve_data(data_S55)))
#print(my_ultimat_fit(data_S55))
#plt.xlim([0.125, -0.3])
#plt.ylim([-0.2,0.2])
#plt.title("S55")

#print(my_ultimat_fit(improve_data(data_S62)))
#print(my_ultimat_fit(data_S62))
#plt.xlim([0.125, -0.3])
#plt.ylim([-0.2,0.2])
#plt.title("S62")



def make_ellipse(a,e,phi,xf,yf):
    X1 = []
    Y1 = []
        
    for i in np.linspace(0, 2*np.pi, 720):
        theta = i 
        r = my_f_ellipse(a,e,phi,xf,yf,theta)[0]
        x = r*np.cos(i) + xf
        y = r*np.sin(i) + yf
            
        X1.append(x)
        Y1.append(y)

    return X1, Y1



fit = my_ultimat_fit(improve_data(data_S2))[0]
a = fit[0]
e = fit[1]
phi = fit[2]
xf = fit[3]
yf = fit[4]
b = a*np.sqrt(1-e**2)
yc = yf + a*e*np.sin(phi)
xc = xf + a*e*np.cos(phi)

yc_test = yf + a*e
inc = np.arccos(np.sqrt( (a**2-yc_test**2)/b**2 ))

X = my_ultimat_fit(improve_data(data_S2))[1]
Y = my_ultimat_fit(improve_data(data_S2))[2]


plt.figure()
my_ultimat_fit(improve_data(data_S2))
plt.plot(xc,yc, '*')
plt.plot([xf,xf + 2*a*e*np.cos(phi)], [yf,yf+ 2*a*e*np.sin(phi)], '*')
plt.plot(0,0,'o', color='black')
plt.xlim([-0.4, 0.2])
plt.ylim([-0.2,0.2])



angle = m.atan2((yf),(-xf))

#X1,Y1 = make_ellipse(a,e,0.5*np.pi,xf,yf)

X1 = []
Y1 = []


for i in range(len(X)):
    X1.append(X[i]*np.cos(angle) - Y[i]*np.sin(angle) - xf)
    Y1.append(X[i]*np.sin(angle) + Y[i]*np.cos(angle) - yf)

x_test = []
y_test = []
for i in np.linspace(-0.1,0.1,10):
    x_test.append(i)
    y_test.append(i*np.tan(angle+phi))

#xf_new = xf*np.cos(angle) - yf*np.sin(angle)  
#yf_new = xf*np.sin(angle) + yf*np.cos(angle) 

plt.figure()
plt.plot(X1,Y1)
plt.plot(x_test, y_test)
#plt.plot(X,Y)
#plt.plot([xf_new,xf_new + 2*a*e*np.cos(angle)], [yf_new, yf_new + 2*a*e*np.sin(angle)], '*')
plt.plot(0,0,'o', color='black')
plt.axis('square')
plt.xlim([-0.02, 0.02])
plt.ylim([-0.02,0.02])
#plt.xlim([-0.2, 0.2])
#plt.ylim([-0.2,0.2])



dist_max = 0
ij_list = []
for i in range(len(X1)):
    for j in range(len(X1)):
        dist = np.sqrt( (X1[i] - X1[j])**2 + (Y1[i] - Y1[j])**2 )
        if dist > dist_max:
            dist_max = dist
            ij_list = [i,j]

print("The greatest dist:", dist_max)
plt.plot([X1[ij_list[0]], X1[ij_list[1]]], [Y1[ij_list[0]], Y1[ij_list[1]]], 'x')







my_ultimat_fit(improve_data(data_S62))
plt.plot(improve_data(data_S62)[1], improve_data(data_S62)[2], '*')
plt.plot(0,0,'o', color='black')
plt.xlim([-0.2,0.2])
plt.ylim([-0.2,0.2])

S62_new_X = []
S62_new_Y = []

for i in range(len(data_S62[0])):
    if i*2 > len(data_S62[0]):
        break
    S62_new_X.append(data_S62[1][i])
    S62_new_Y.append(data_S62[2][i])
    
new_data = [[],S62_new_X, S62_new_Y]

my_ultimat_fit(new_data)
plt.plot(new_data[1], new_data[2], '*')
plt.plot(0,0,'o', color='black')
plt.xlim([-0.2,0.2])
plt.ylim([-0.2,0.2])























meh, X, Y = my_ultimat_fit(improve_data(data_S2))
a,e,phi,xf,yf = meh
fx = [xf,xf+np.cos(phi)*a*e*2]
fy = [yf,yf+np.sin(phi)*a*e*2]

Ax = []
Ay = []
FY = []
FX = []
for i in np.linspace(0,2*np.pi, 10):
    Bx = []
    By = []
    fx_new = []
    fy_new = []
    for j in range(len(X)):        
        x = np.cos(i)*X[j] - np.sin(i)*Y[j]
        y = np.sin(i)*X[j] + np.cos(i)*Y[j]
        Bx.append(x)
        By.append(y)
    for p in range(len(fx)):
        x = np.cos(i)*fx[p] - np.sin(i)*fy[p]
        y = np.sin(i)*fx[p] + np.cos(i)*fy[p]
        fx_new.append(x)
        fy_new.append(y)
    FX.append(fx_new)
    FY.append(fy_new)
    Ax.append(Bx)
    Ay.append(By)

plt.figure()
for i in range(len(Ax)):
#    plt.plot(Ax[i], Ay[i])
    plt.plot(FX[i], FY[i])
plt.plot(0,0,'o',c='black')
plt.xlim([-0.05,0.05])
plt.ylim([-0.05,0.05])

phi_real = np.radians(12)
X_real = []
Y_real = []
for i in range(len(X)):
    x = np.cos(phi_real)*X[i] - np.sin(phi_real)*Y[i]
    y = np.sin(phi_real)*X[i] + np.cos(phi_real)*Y[i]
    X_real.append(x)
    Y_real.append(y)

fx = [xf,xf+np.cos(phi)*a*e*2]
fy = [yf,yf+np.sin(phi)*a*e*2]

fx_new = []
fy_new = []

for i in range(len(fx)):
    x = np.cos(phi_real)*fx[i] - np.sin(phi_real)*fy[i]
    y = np.sin(phi_real)*fx[i] + np.cos(phi_real)*fy[i]
    fx_new.append(x)
    fy_new.append(y)


plt.figure()
plt.plot(X_real, Y_real)
plt.plot(X, Y)
plt.plot(0,0,'o', c='black')
plt.plot(fx,fy)
plt.plot(fx_new, fy_new)
plt.xlim([-0.1,0.2])
plt.ylim([-0.1,0.2])




    
    
    
    
    



# -----------------------------------------------------------------------------
ri, xi, yi = my_ultimat_fit(improve_data(data_S2))
r, x, y = my_ultimat_fit(data_S2)

plt.figure()
plt.title("LOOK AT THIS ONE")
plt.plot(x_S2, y_S2,'*')
plt.ylim([-0.05,0.2])
plt.xlim([-0.15,0.1])
plt.plot(x,y)
plt.plot(xi,yi)
plt.plot(0,0,'o',c='black')
plt.plot(r[-2],r[-1],'*', color = 'orange')
plt.plot(r[-2]+2*r[0]*r[1]*np.cos(r[2]),r[-1]+2*r[0]*r[1]*np.sin(r[2]),'*', color = 'orange')
plt.plot(ri[-2],ri[-1],'*', color = 'green')
plt.plot(ri[-2]+2*ri[0]*ri[1]*np.cos(ri[2]),ri[-1]+2*ri[0]*ri[1]*np.sin(ri[2]),'*', color = 'green')


    







# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def f(t, a, e, omega, OMEGA, inc):
    A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) #*a
    C = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) #*a
    B = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) #*a
    D = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc)) #*a
    
    r = my_f_ellipse(a,e,np.radians(0),0,0,t)[0]
    x = r*np.cos(t)
    y = r*np.sin(t)    
    
    X = A*x+B*y
    Y = C*x+D*y
    return X, Y




def make_test_ellipse(n,a=0,e=0,omega=0,OMEGA=0,inc=0):
    t = r.sample(range(0,2*np.pi),n)    
    if a == 0: 
        a = r.uniform(0.1,0.2)
        e = r.uniform(0.3,0.99)
        omega = r.uniform(0,2*np.pi)
        OMEGA = r.uniform(0,2*np.pi)
        inc = r.uniform(0,2*np.pi)
    
    x,y = f(t,a,e,omega,OMEGA,inc)
    return [x,y], [a,e,omega,OMEGA,inc]
    
# Now to testing this stuff




















