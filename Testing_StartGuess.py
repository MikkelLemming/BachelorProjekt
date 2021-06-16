import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy import optimize as opt
import math as m
from numpy.linalg import eig, inv, svd
from mpl_toolkits.mplot3d import Axes3D


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








def my_f_ellipse(a, e, phi, xf, yf, theta):
    r = a*(1-e**2)/(1-e*np.cos(theta - phi))
    x = r*np.cos(theta) + xf
    y = r*np.sin(theta) + yf
    return r, [x,y]

def f(t, a, e, omega, OMEGA, inc):
    A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) #*a
    B = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) #*a
    F = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) #*a
    G = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc)) #*a
    
    r = my_f_ellipse(a,e,np.radians(0),0,0,t)[0]
    x = r*np.cos(t)
    y = r*np.sin(t)    
    
    X = A*x+F*y
    Y = B*x+G*y
    return X, Y

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









def the_mega_function(data,n,name):
    t = np.linspace(0,2*np.pi, 720)
    xdata = data[1]
    ydata = data[2]
    avg_dist_list = []
    a_list = []
    e_list = []
    omega_list = []
    OMEGA_list = []
    inc_list = []
    
    nummer = 0
    
    for a in np.linspace(0.02,0.2,n):
        for e in np.linspace(0.5,0.99,n):
            for omega in np.linspace(0,2*np.pi,n):
                for OMEGA in np.linspace(0,2*np.pi,n):
                    for inc in np.linspace(0,2*np.pi,n):
                        a_list.append(a)
                        e_list.append(e)
                        omega_list.append(omega)
                        OMEGA_list.append(OMEGA)
                        inc_list.append(inc)
                        
                        nummer = nummer + 1
                        print("We are now at:", nummer, "out of:", n**5, " ", round(nummer/n**5 * 100,2),"percent done, now on:", name)
                        X,Y = f(t, a, e, omega, OMEGA, inc)
                        
                        dist_all_points = []
                        for i in range(len(xdata)):
                            min_dist_to_point = 10
                            for j in range(len(X)):
                                d = np.sqrt( (xdata[i] - X[j])**2 + (ydata[i] - Y[j])**2 )#/(np.sqrt(data[3][i]**2+data[4][i]**2))
                                if d < min_dist_to_point:
                                    min_dist_to_point = d
                            dist_all_points.append(min_dist_to_point**2)
#                            dist_all_points.append(min_dist_to_point)
                        avg_dist = sum(dist_all_points)/len(dist_all_points)
                        avg_dist_list.append(avg_dist)
    
    index = avg_dist_list.index(min(avg_dist_list))
    return a_list[index], e_list[index], omega_list[index], OMEGA_list[index], inc_list[index], min(avg_dist_list)


n = 20

S2_a,S2_e,S2_omega,S2_OMEGA,S2_inc,S2_avg = the_mega_function(data_S2,n,"S2")

#open("OrbitValues_Method5", "w").close()
open("OrbitValues_Method5", "a")
file = open("OrbitValues_Method5", "a")
file.write("Values for S2:")
file.write("\nS2 a: ")
file.write(str(S2_a))
file.write("\nS2 e: ")
file.write(str(S2_e))
file.write("\nS2 omega: ")
file.write(str(S2_omega))
file.write("\nS2 OMEGA: ")
file.write(str(S2_OMEGA))
file.write("\nS2 inc: ")
file.write(str(S2_inc))
file.write("\nS2 avg: ")
file.write(str(S2_avg))
file.write("\n\n")


S2_ai,S2_ei,S2_omegai,S2_OMEGAi,S2_inci,S2_avgi = the_mega_function(improve_data(data_S2),n,"S2i")

file.write("Values for S2 improved:")
file.write("\nS2 improved a: ")
file.write(str(S2_ai))
file.write("\nS2 improved e: ")
file.write(str(S2_ei))
file.write("\nS2 improved omega: ")
file.write(str(S2_omegai))
file.write("\nS2 improved OMEGA: ")
file.write(str(S2_OMEGAi))
file.write("\nS2 improved inc: ")
file.write(str(S2_inci))
file.write("\nS2 improved avg: ")
file.write(str(S2_avgi))
file.write("\n\n")


S62_a,S62_e,S62_omega,S62_OMEGA,S62_inc,S62_avg = the_mega_function(data_S62,n,"S62")
# --- S62
file.write("Values for S62:")
file.write("\nS62 a: ")
file.write(str(S62_a))
file.write("\nS62 e: ")
file.write(str(S62_e))
file.write("\nS62 omega: ")
file.write(str(S62_omega))
file.write("\nS62 OMEGA: ")
file.write(str(S62_OMEGA))
file.write("\nS62 inc: ")
file.write(str(S62_inc))
file.write("\nS62 avg: ")
file.write(str(S62_avg))
file.write("\n\n")


S55_a,S55_e,S55_omega,S55_OMEGA,S55_inc,S55_avg = the_mega_function(data_S55,n,"S55")
# --- S55
file.write("Values for S55:")
file.write("\nS55 a: ")
file.write(str(S55_a))
file.write("\nS55 e: ")
file.write(str(S55_e))
file.write("\nS55 omega: ")
file.write(str(S55_omega))
file.write("\nS55 OMEGA: ")
file.write(str(S55_OMEGA))
file.write("\nS55 inc: ")
file.write(str(S55_inc))
file.write("\nS55 avg: ")
file.write(str(S55_avg))
file.write("\n\n")




S38_a,S38_e,S38_omega,S38_OMEGA,S38_inc,S38_avg = the_mega_function(data_S38,n,"S38")

# -- S38
file.write("Values for S38:")
file.write("\nS38 a: ")
file.write(str(S38_a))
file.write("\nS38 e: ")
file.write(str(S38_e))
file.write("\nS38 omega: ")
file.write(str(S38_omega))
file.write("\nS38 OMEGA: ")
file.write(str(S38_OMEGA))
file.write("\nS38 inc: ")
file.write(str(S38_inc))
file.write("\nS38 avg: ")
file.write(str(S38_avg))
file.write("\n\n")


file.close()

# Lets hope its done before the 22.






t = np.linspace(0,2*np.pi,720)

#X, Y = f(t, S2_a, S2_e, S2_omega, S2_OMEGA, S2_inc)

#X2, Y2 = f(t, 0.121, 0.8337, 2.173, 4.1288, 3.9402)

#X3, Y3 = f(t,0.1254, 0.885, 2.06, 3.977, 2.33)

#plt.plot(data_S2[1], data_S2[2], '*')
#plt.plot(X,Y,'o')
#plt.plot(X2,Y2,'o',c='green')
#plt.plot(X3,Y3,'o', c='red')
#plt.plot(0,0,'o',c='black')

#dist_2 = []
#for j in range(len(data_S2[1])):
#    dist_min = 10**2
#    for i in range(len(X2)):
#        dist = np.sqrt((data_S2[1][j] - X2[i])**2 + (data_S2[2][j] - Y2[i])**2)
#        if dist < dist_min:
#            dist_min = dist
#    dist_2.append(dist_min**2)


#dist_3 = []
#for j in range(len(data_S2[1])):
#    dist_min = 10**2
#    for i in range(len(X3)):
#        dist = (data_S2[1][j] - X3[i])**2 + (data_S2[2][j] - Y3[i])**2
#        if dist < dist_min:
#            dist_min = dist
#    dist_3.append(dist_min)




def fit_method4(data, a0, e0, omega0, OMEGA0, inc0):
    
    def residual_4(parameters):
        a,e,omega,OMEGA,inc = parameters
        dist = []
        t = np.linspace(0,2*np.pi,720)
        X, Y = f(t,a,e,omega,OMEGA,inc)
        
        for j in range(len(data[1])):
            dist_min = 10**2
            for i in range(len(X)):
                distance = ((data[1][j] - X[i])**2 + (data[2][j] - Y[i])**2)#/(data[3][j]**2 + data[4][j]**2)
                if distance < dist_min:
                    dist_min = distance
            dist.append(dist_min)
                    
        return sum(dist)/len(dist)
    
    guess_init = [a0,e0,omega0,OMEGA0,inc0]
    
    bnds = ((0.05, 0.1), (0.3,0.99), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi))
    
    result = opt.minimize(residual_4, guess_init, method='L-BFGS-B', bounds=bnds)  
    
    return result.x, result.fun


def plot_fit_method4(data,a0, e0, omega0, OMEGA0, inc0):
    t = np.linspace(0,2*np.pi,720)
    X1, Y1 = f(t,a0,e0,omega0,OMEGA0,inc0)
    
    fit = fit_method4(data,a0,e0,omega0,OMEGA0,inc0)
    print(a0,e0,omega0,OMEGA0,inc0)
    a,e,omega,OMEGA,inc = fit
    X2, Y2 = f(t,a,e,omega,OMEGA,inc)

    plt.plot(data[1], data[2],'*')
    plt.plot(X1,Y1,'o',c='green')
    plt.plot(0,0,'o')
    plt.plot(X2,Y2,'o',c='orange')
    return fit









