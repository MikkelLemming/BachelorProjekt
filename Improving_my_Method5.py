import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy import optimize as opt
import math as m
from numpy.linalg import eig, inv, svd
from mpl_toolkits.mplot3d import Axes3D


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def my_f_ellipse(a, e, phi, xf, yf, theta):
    r = a*(1-e**2)/(1-e*np.cos(theta - phi))
    x = r*np.cos(theta) + xf
    y = r*np.sin(theta) + yf
    return r, [x,y]

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
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





# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
def the_mega_function(data,n,name):
    xdata = data[1]
    ydata = data[2]
    r, theta = r_theta_data(data)

    avg_dist_list = []
    a_list = []
    e_list = []
    omega_list = []
    OMEGA_list = []
    inc_list = []
    
    nummer = 0
    
    for a in np.linspace(0.02,0.2,n):
        for e in np.linspace(0.01,0.99,n):
            for omega in np.linspace(0,2*np.pi,n):
                for OMEGA in np.linspace(0,2*np.pi,n):
                    for inc in np.linspace(0,2*np.pi,n):
                        a_list.append(a)
                        e_list.append(e)
                        omega_list.append(omega)
                        OMEGA_list.append(OMEGA)
                        inc_list.append(inc)
                        
                        A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) 
                        C = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) 
                        B = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) 
                        D = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc)) 

                        nummer = nummer + 1
                        print("We are now at:", nummer, "out of:", n**5, " ", round(nummer/n**5 * 100,2),"percent done, now on:", name)
                        
                        dist_all_points = []
                        for i in range(len(theta)):
                            X0 = xdata[i]
                            Y0 = ydata[i]
                            y = (Y0 - X0*C/A)/(D - B*C/A)
                            ratio = A*y/(X0-B*y)
                            t = np.arctan(ratio)%(2*np.pi)
                            
                            X,Y = f(t, a, e, omega, OMEGA, inc)
                            
                            
                            dist = np.sqrt( (X0 - X)**2 + (Y0 - Y)**2 )
                            dist_all_points.append(dist**2)

                        
                        avg_dist = sum(dist_all_points)/len(dist_all_points)
                        avg_dist_list.append(avg_dist)

    index = avg_dist_list.index(min(avg_dist_list))
    return a_list[index], e_list[index], omega_list[index], OMEGA_list[index], inc_list[index], min(avg_dist_list)
    



# ------------------------------------------------------------------------------------------
# ------------------------------------------------ TESTING THE STUFF -----------------------
# ------------------------------------------------------------------------------------------
n = 20

#S2_a,S2_e,S2_omega,S2_OMEGA,S2_inc,S2_avg = the_mega_function(data_S2,n,"S2")
#S2_a2,S2_e2,S2_omega2,S2_OMEGA2,S2_inc2,S2_avg2 = the_mega_function(data_S2,10,"S2")



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
    
#    bnds = ((a0-0.18/9,a0+0.18/9), (e0-0.49/9, e0+0.49/9), (omega0-2*np.pi/9, omega0+2*np.pi/9), (OMEGA0-2*np.pi/9, OMEGA0+2*np.pi/9), (inc0-2*np.pi/9, inc0+2*np.pi/9))
    
    result = opt.minimize(residual_4, guess_init, method='L-BFGS-B')#, bounds=bnds)  
    
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

#plot_fit_method4(data_S2, S2_a, S2_e, S2_omega, S2_OMEGA, S2_inc)
#plt.figure()
#plot_fit_method4(data_S2, S2_a2, S2_e2, S2_omega2, S2_OMEGA2, S2_inc2)



# ------------------------------------------------------------------------------------------
# ------------------------------------------------ TESTING THE STUFF -----------------------
# ------------------------------------------------------------------------------------------
def the_mega_function2(data, a0, e0, omega0, OMEGA0, inc0, n, lap):
    xdata = data[1]
    ydata = data[2]
    r, theta = r_theta_data(data)

    avg_dist_list = []
    a_list = []
    e_list = []
    omega_list = []
    OMEGA_list = []
    inc_list = []
    
    nummer = 0
    
    for a in np.linspace(a0-a0/(n-1),a0+a0/(n-1),n):
        for e in np.linspace(e0-e0/(n-1),e0+e0/(n-1),n):
            for omega in np.linspace(omega0-omega0/(n-1),omega0+omega0/(n-1),n):
                for OMEGA in np.linspace(OMEGA0-OMEGA0/(n-1),OMEGA0+OMEGA0/(n-1),n):
                    for inc in np.linspace(inc0-inc0/(n-1),inc0+inc0/(n-1),n):
                        a_list.append(a)
                        e_list.append(e)
                        omega_list.append(omega)
                        OMEGA_list.append(OMEGA)
                        inc_list.append(inc)
                        
                        A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) 
                        C = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) 
                        B = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) 
                        D = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc)) 
                        
                        nummer = nummer + 1
                        print("We are now at:", nummer, "out of:", n**5, " ", round(nummer/n**5 * 100,2),"percent done, now on lap:", lap)
                        
                        dist_all_points = []
                        for i in range(len(theta)):
                            X0 = xdata[i]
                            Y0 = ydata[i]
                            y = (Y0/C - X0/A)/(D/C - B/A)
                            ratio = A*y/(X0-B*y)
                            t = np.arctan(ratio)
                            
                            X,Y = f(t, a, e, omega, OMEGA, inc)
                            
                            dist = np.sqrt( (X0 - X)**2 + (Y0 - Y)**2 )
                            dist_all_points.append(dist**2)
                        
                        avg_dist = sum(dist_all_points)/len(dist_all_points)
                        avg_dist_list.append(avg_dist)
    
    index = avg_dist_list.index(min(avg_dist_list))
    return a_list[index], e_list[index], omega_list[index], OMEGA_list[index], inc_list[index], min(avg_dist_list)


#S2_a_new,S2_e_new,S2_omega_new,S2_OMEGA_new,S2_inc_new,S2_avg_new = S2_a, S2_e, S2_omega, S2_OMEGA, S2_inc, S2_avg

#open("OrbitValues_Method5", "w").close()
#open("OrbitValues_Method5", "a")
#file = open("OrbitValues_Method5", "a")
#file.write("Values for lap:")
#file.write(str(0))
#file.write("\na: ")
#file.write(str(S2_a_new))
#file.write("\ne: ")
#file.write(str(S2_e_new))
#file.write("\nomega: ")
#file.write(str(S2_omega_new))
#file.write("\nOMEGA: ")
#file.write(str(S2_OMEGA_new))
#file.write("\ninc: ")
#file.write(str(S2_inc_new))
#file.write("\navg: ")
#file.write(str(S2_avg_new))
#file.write("\n\n")


#n = 5
#lap = 0
#max_labs = 15
#for i in range(max_labs):
#    lap = lap + 1
#    S2_a_new,S2_e_new,S2_omega_new,S2_OMEGA_new,S2_inc_new,S2_avg_new = the_mega_function2(data_S2,S2_a_new,S2_e_new,S2_omega_new,S2_OMEGA_new,S2_inc_new,n,lap)
#    file.write("Values for lap:")
#    file.write(str(lap))
#    file.write("\na: ")
#    file.write(str(S2_a_new))
#    file.write("\ne: ")
#    file.write(str(S2_e_new))
#    file.write("\nomega: ")
#    file.write(str(S2_omega_new))
#    file.write("\nOMEGA: ")
#    file.write(str(S2_OMEGA_new))
#    file.write("\ninc: ")
#    file.write(str(S2_inc_new))
#    file.write("\navg: ")
#    file.write(str(S2_avg_new))
#    file.write("\n\n")

































def the_mega_function3(data,n,name):
    xdata = data[1]
    ydata = data[2]
    r, theta = r_theta_data(data)

    avg_dist_list = []
    a_list = []
    e_list = []
    omega_list = []
    OMEGA_list = []
    inc_list = []
    
    nummer = 0
    a = 0.1254
    e = 0.88
    
    for a in np.linspace(0.1,0.2,101):
        for e in np.linspace(0.01,0.99,99):
            for omega in np.linspace(0,2*np.pi,n):
                for OMEGA in np.linspace(0,2*np.pi,n):
                    for inc in np.linspace(0,2*np.pi,n):
                        a_list.append(a)
                        e_list.append(e)
                        omega_list.append(omega)
                        OMEGA_list.append(OMEGA)
                        inc_list.append(inc)
                                
                        A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) 
                        C = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) 
                        B = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) 
                        D = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc)) 
        
                        nummer = nummer + 1
                        print("We are now at:", nummer, "out of:", 101*99*n**3, " ", round(nummer/(101*99*n**3) * 100,2),"percent done, now on:", name)
                                
                        dist_all_points = []
                        for i in range(len(theta)):
                            X0 = xdata[i]
                            Y0 = ydata[i]
                            y = (Y0 - X0*C/A)/(D - B*C/A)
                            ratio = A*y/(X0-B*y)
                            t = np.arctan(ratio)%(2*np.pi)
                                    
                            X,Y = f(t, a, e, omega, OMEGA, inc)
                                    
                                    
                            dist = np.sqrt( (X0 - X)**2 + (Y0 - Y)**2 )
                            dist_all_points.append(dist**2)
        
                                
                        avg_dist = sum(dist_all_points)/len(dist_all_points)
                        avg_dist_list.append(avg_dist)

    index = avg_dist_list.index(min(avg_dist_list))
    return a_list[index], e_list[index], omega_list[index], OMEGA_list[index], inc_list[index], min(avg_dist_list)
    
#n = 15
#a_test, e_test, omega_test, OMEGA_test, inc_test, avg_test = the_mega_function3(data_S2,n,"S2")

#plot_fit_method4(data_S2, a_test, e_test, omega_test, OMEGA_test, inc_test)







def error_test_2(data, a0=0.1, e0=0.5, omega0=np.pi, OMEGA0=np.pi, inc0=np.pi):
#    a, e, omega, OMEGA, inc = fit_method4(data, a0, e0, omega0, OMEGA0, inc0)[0]
    a, e, omega, OMEGA, inc = a0, e0, omega0, OMEGA0, inc0
    
    t = np.linspace(0,2*np.pi, 1080)
    x, y = f(t,a,e,omega,OMEGA,inc)
    
    num = 0
    for i in range(len(data[1])):
#        dist = []
        distx = []
        disty = []
        for j in range(len(x)):
#            dist.append( np.sqrt( (data[1][i] - x[j])**2 + (data[2][i] - y[j])**2 ) )
            distx = abs(data[1][i] - x[j])
            disty = abs(data[2][i] - y[j])
            
#        if min(dist) <= np.sqrt( (data[3][i])**2 + (data[4][i])**2 ):
#            num = num + 1

            if distx <= data[3][i]*1.2 and disty <= data[4][i]*1.2:
                num = num + 1
                break 
            
    return num




def the_mega_function4(data,n,name):
    xdata = data[1]
    ydata = data[2]
    r, theta = r_theta_data(data)

    avg_dist_list = []
    a_list = []
    e_list = []
    omega_list = []
    OMEGA_list = []
    inc_list = []
    
    nummer = 0
    
    num_max = 0
    
    n_a = 15
    n_e = 10
    
#    for a0 in np.linspace(0.05,0.2,30):
    for a0 in np.linspace(0.08,0.2,n_a):
#        for e0 in np.linspace(0.3,0.99,20):
        for e0 in np.linspace(0.5,0.99,n_e):
#            for omega0 in [0, 2/3*np.pi, 4/3*np.pi]:
            for omega0 in [0.001, 1/3*np.pi+0.001, 2/3*np.pi+0.001, np.pi+0.001, 4/3*np.pi+0.001, 5/3*np.pi+0.001]:
#                for OMEGA0 in [0, 2/3*np.pi, 4/3*np.pi]:
                for OMEGA0 in [0.001, 1/3*np.pi+0.001, 2/3*np.pi+0.001, np.pi+0.001, 4/3*np.pi+0.001, 5/3*np.pi+0.001]:
#                    for inc0 in [0, 2/3*np.pi, 4/3*np.pi]:
                    for inc0 in [0.001, 1/3*np.pi+0.001, 2/3*np.pi+0.001, np.pi+0.001, 4/3*np.pi+0.001, 5/3*np.pi+0.001]:                        
                        nummer = nummer + 1
#                        print("We are now at:", nummer, "out of:", 101*70*n**3, " ", round(nummer/(101*70*n**3) * 100,2),"percent done, now on:", name)
                        print("We are now at:", nummer, "out of:", n_a*n_e*6**3, " ", round(nummer/(n_a*n_e*6**3) * 100,2),"percent done, now on:", name)
                        
                        
                        def residual_4(parameters):
                            a,e,omega,OMEGA,inc = parameters
    
                            A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) 
                            C = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) 
                            B = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) 
                            D = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc)) 
        
                               
                            dist_all_points = []
                            for i in range(len(theta)):
                                X0 = xdata[i]
                                Y0 = ydata[i]
                                y = (Y0 - X0*C/A)/(D - B*C/A)
                                ratio = A*y/(X0-B*y)
                                t = np.arctan(ratio)%(2*np.pi)
                                X,Y = f(t, a, e, omega, OMEGA, inc)
                                       
                                    
                                dist = np.sqrt( (X0 - X)**2 + (Y0 - Y)**2 )
                                if dist <= min(data[3][i],data[4][i]):
                                    dist = 0.2 * dist
                                dist_all_points.append(dist**2)
        
                                
                            avg_dist = sum(dist_all_points)/len(dist_all_points)
                            return avg_dist                     
                        
                        bnds = [(a0-0.0025, a0+0.0025),(e0-0.02, e0+0.02), (0,2*np.pi),(0,2*np.pi),(0,2*np.pi)]
                        result = opt.minimize(residual_4, [a0,e0,omega0, OMEGA0, inc0], bounds=bnds)
                        a, e, omega, OMEGA, inc = result.x
                        avg_dist = result.fun
                        

                        num = error_test_2(data, a, e, omega, OMEGA, inc)

                        if num >= n - 2:
                            a_list.append(a)
                            e_list.append(e)
                            omega_list.append(omega)
                            OMEGA_list.append(OMEGA)
                            inc_list.append(inc)
                            avg_dist_list.append(avg_dist)

    return a_list, e_list, omega_list, OMEGA_list, inc_list, avg_dist_list, num_max



a_S2, e_S2, omega_S2, OMEGA_S2, inc_S2, avg_S2, num_S2 = the_mega_function4(data_S2,33,"S2")
a_S62, e_S62, omega_S62, OMEGA_S62, inc_S62, avg_S62, num_S62 = the_mega_function4(data_S62,15,"S62")
#a_S38, e_S38, omega_S38, OMEGA_S38, inc_S38, avg_S38, num_S38 = the_mega_function4(data_S38,16,"S38")
#a_S55, e_S55, omega_S55, OMEGA_S55, inc_S55, avg_S55, num_S55 = the_mega_function4(data_S55,19,"S55")

#print(a_test, e_test, omega_test, OMEGA_test, inc_test, avg_test)
#plot_fit_method4(data_S2, a_test, e_test, omega_test, OMEGA_test, inc_test)



















# Funktion som viser at der er en faktor 10 i forskel imellem min (a=0.1108) og litteratur (a=0.1254)
a = 0.1254
e = 0.88466
OMEGA = np.radians(227.85)
inc = np.radians(133.818)

def testything(omega):
  
    A = (np.cos(omega)*np.cos(OMEGA) - np.sin(omega)*np.sin(OMEGA)*np.cos(inc)) 
    C = (np.cos(omega)*np.sin(OMEGA) + np.sin(omega)*np.cos(OMEGA)*np.cos(inc)) 
    B = (-np.sin(omega)*np.cos(OMEGA) - np.cos(omega)*np.sin(OMEGA)*np.cos(inc)) 
    D = (-np.sin(omega)*np.sin(OMEGA) + np.cos(omega)*np.cos(OMEGA)*np.cos(inc)) 
        
                               
    dist_all_points = []
    for i in range(len(data_S2[1])):
        X0 = data_S2[1][i]
        Y0 = data_S2[2][i]
        y = (Y0 - X0*C/A)/(D - B*C/A)
        ratio = A*y/(X0-B*y)
        t = np.arctan(ratio)%(2*np.pi)
                                    
        X,Y = f(t, a, e, omega, OMEGA, inc)
                                       
                                    
        dist = np.sqrt( (X0 - X)**2 + (Y0 - Y)**2 )
        dist_all_points.append(dist**2)  
    return sum(dist_all_points)/len(dist_all_points)

#result = opt.minimize(testything, 0)

#print(result.x, result.fun)



print("Program done")







